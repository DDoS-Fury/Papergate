"""Serving and persistence layer of the streaming TGN: the real-time code path.

Scoring and memory:
- :func:`infer_logit`: score one edge (read-only), the per-edge reference.
- :func:`chain_edge_logits` / :func:`combine_edge_logits` / :func:`chain_logits`: score all
  edges of a block of events from one shared expansion; the event logit is the max of the
  per-edge logits, each calibrated on its own benign reference when the artifact has one.
- :func:`update_memory`: commit one edge into memory, neighbours and history counters.
- :func:`event_alarm`: per-event alarm rule (flagged, armed or sensor), shared by the
  offline replay and every serving path.

Online API (entity keys mapped through a :class:`NodeRegistry`):
- :func:`score_event`: score; with ``update``, commit only events judged benign.
- :func:`commit_event`: unconditional commit when the decision is taken outside (OPA ALLOW).
- :func:`deny_event`: OPA DENY: record the alarm only, never the baseline.

Persistence: :func:`save_model` / :func:`load_model`.

Parity contract: ``train_tgn._replay`` shares :func:`chain_edge_logits` and re-implements
the commits in batch; ``tests/verify_replay_batching.py`` and ``tests/test_fusion_parity.py``
check the replay and :func:`score_event` against the per-edge reference (keep them green).
Deliberate divergence: the offline replay commits on the OPA-ALLOW proxy, while
``score_event(update=True)`` commits on the model's own score.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Hashable

import numpy as np
import torch
from scipy.special import expit

from graphagate.config import TGNConfig
from graphagate.model.registry import NodeRegistry
from graphagate.model.tgn import ZTATemporalGraphNetwork, stable_hash
from graphagate.netclass import ip_is_internal, to_guest_device


SCHEMA_VERSION = 4  # 5-node schema: source→config→device→user→resource (+ config→user)


def build_model(hp: dict, device: torch.device) -> ZTATemporalGraphNetwork:
    """Instantiate the model from a hyper-parameter dict (see :func:`save_model`)."""
    found = int(hp.get("schema_version", 1))
    if found != SCHEMA_VERSION:
        raise RuntimeError(
            f"checkpoint schema_version={found} is incompatible with this code "
            f"(expected {SCHEMA_VERSION}: 5-node schema with the config/JA3 node — "
            "source→config→device→user→resource plus a config→user binding). "
            "Retrain with `graphagate.train_tgn` to regenerate the artifact."
        )
    model = ZTATemporalGraphNetwork(
        num_nodes=int(hp["capacity"]),
        node_feat_dim=int(hp["node_feat_dim"]),
        msg_dim=int(hp["msg_dim"]),
        memory_dim=int(hp["memory_dim"]),
        time_dim=int(hp["time_dim"]),
        num_hops=int(hp.get("num_hops", 2)),
        hash_buckets=int(hp.get("hash_buckets", 10000)),
        hash_dim=int(hp.get("hash_dim", 16)),
        hist_feat_dim=int(hp.get("hist_feat_dim", 6)),
        gnn_heads=int(hp.get("gnn_heads", 4)),
        link_pred_hidden_layers=int(hp.get("link_pred_hidden_layers", 2)),
    ).to(device)
    # Kill-chain precursor knobs (serving-time, not in the state_dict); fallbacks = TGNConfig.
    model.precursor_half_life = float(hp.get("precursor_half_life", TGNConfig.precursor_half_life))
    model.precursor_max_shift = float(hp.get("precursor_max_shift", TGNConfig.precursor_max_shift))
    # Plain-attribute toggles are not in the state_dict: restore them from hp
    # (default True = the training default).
    model.use_precursor = bool(hp.get("use_precursor", True))
    # Must match training, else source nodes carry a feature the model never learned.
    model.use_source_internal = bool(hp.get("use_source_internal", False))
    # Neighbour loader lives outside the state_dict; load_model restores its contents.
    model.init_neighbor_loader(int(hp.get("neighbor_size", 10)), device)
    # eval() before the caller restores buffers: TGNMemory flushes its message store into
    # memory on train->eval, so a later switch would apply the restored messages twice.
    model.eval()
    return model


def _event_tensors(src_idx: int, dst_idx: int, t_val: int, msg_vec, device):
    """Single-event ``(src, dst, t, msg)`` tensors on ``device``."""
    b_src = torch.tensor([src_idx], dtype=torch.long, device=device)
    b_dst = torch.tensor([dst_idx], dtype=torch.long, device=device)
    # TGNMemory.last_update is int64 — timestamps stay integer end-to-end.
    b_t = torch.tensor([int(t_val)], dtype=torch.long, device=device)
    b_msg = torch.as_tensor(msg_vec, dtype=torch.float, device=device).reshape(1, -1)
    return b_src, b_dst, b_t, b_msg


@torch.no_grad()
def infer_logit(model, src_idx: int, dst_idx: int, t_val: int, msg_vec, device,
                aux_src_idx: int | None = None) -> float:
    """Anomaly *logit* (``-logit P(benign)``) of a single edge; mutates nothing.

    A logit because ``1 - sigmoid`` saturates to 1.0 in float32 below -16, tying the head
    of the ranking; :func:`anomaly_score` converts it to the thresholds' probability
    (monotone). Both endpoints are expanded to their stored temporal neighbourhood; a cold
    node falls back to its memory. ``aux_src_idx`` feeds the per-device history triplet of
    the access edge (``compute_hist_feats``); ``None`` zero-pads it.
    """
    b_src, b_dst, _b_t, b_msg = _event_tensors(src_idx, dst_idx, t_val, msg_vec, device)
    nodes = torch.unique(torch.cat([b_src, b_dst]))
    n_id, edge_index, hist_t, hist_msg = model.neighbor_loader(nodes)
    assoc = model.neighbor_loader._assoc

    # Both recencies go through the model's shared encoder (cap + never-seen sentinel),
    # which is what keeps the serving path on the same Δt distribution as training.
    delta_t = model.pair_delta_t([src_idx], [dst_idx], [t_val], device)
    delta_t_src = model.src_delta_t([src_idx], [float(t_val)], device)

    # Interaction-history features of the pair (read-only; update_memory advances them).
    aux_ids = None if aux_src_idx is None else [aux_src_idx]
    hist_feats = model.compute_hist_feats([src_idx], [dst_idx], device, aux_src_ids=aux_ids)

    out = model(
        n_id, edge_index, hist_t, hist_msg, assoc[b_src], assoc[b_dst], b_msg, delta_t, delta_t_src, hist_feats
    ).squeeze(-1)
    return -float(out.item())


# Edge kinds of the causal chain. Every edge group carries its kind explicitly: the
# serving and replay paths list the groups in different orders, and the per-edge benign
# calibration below is keyed by kind, not by position.
EDGE_ACCESS = "user>res"
EDGE_DEV_USER = "dev>user"
EDGE_CFG_USER = "cfg>user"
EDGE_CFG_DEV = "cfg>dev"
EDGE_SRC_CFG = "src>cfg"
EDGE_SRC_DEV = "src>dev"  # config-node ablation only


def fit_edge_calibration(benign_logits, *, tail_q: float) -> dict:
    """Benign reference distribution of one edge kind's anomaly logits.

    A quantile grid up to ``tail_q`` plus an exponential tail fitted on the exceedances
    above it (so the calibrated score keeps ranking beyond the largest benign value seen,
    instead of tying there). Plain floats / lists: persisted in the checkpoint as is.
    """
    x = np.sort(np.asarray(benign_logits, dtype=np.float64))
    if x.size < 2:
        raise ValueError("edge calibration needs at least two benign logits")
    probs = np.linspace(0.0, tail_q, 513)
    q = np.quantile(x, probs)
    exc = x[x > q[-1]] - q[-1]
    scale = float(exc.mean()) if exc.size else float(x.std() or 1.0)
    return {"probs": probs.tolist(), "q": q.tolist(), "tail_q": float(tail_q),
            "tail_scale": max(scale, 1e-6), "n": int(x.size)}


def calibrated_edge_logit(model, kind: str, logit: torch.Tensor) -> torch.Tensor:
    """Map raw anomaly logits of edge ``kind`` onto ``log((1-p)/p)``, ``p`` their upper-tail
    p-value under the benign reference of that kind (float64).

    Raw edge logits live on different scales (benign access ≈ -10, bindings ≈ -15); on the
    p-value scale every edge is equally surprising at the same benign quantile. ``p`` is
    floored at ``1/(n+1)`` and extrapolated in log space above the grid, so the map is
    monotone and finite; a kind without a reference passes through unchanged.
    """
    spec = (getattr(model, "edge_calib", None) or {}).get(kind)
    if spec is None:
        return logit
    cache = model.__dict__.setdefault("_edge_calib_t", {})
    key = (kind, logit.device)
    if key not in cache:
        cache[key] = (torch.tensor(spec["q"], dtype=torch.float64, device=logit.device),
                      torch.tensor(spec["probs"], dtype=torch.float64, device=logit.device))
    q, probs = cache[key]
    x = logit.to(torch.float64)
    i = torch.searchsorted(q, x.contiguous(), right=True).clamp(1, q.numel() - 1)
    q0, q1, p0, p1 = q[i - 1], q[i], probs[i - 1], probs[i]
    cdf = p0 + (x - q0).clamp(min=0) / (q1 - q0).clamp(min=1e-12) * (p1 - p0)
    cdf = torch.minimum(cdf, p1).clamp(min=1.0 / (spec["n"] + 1))
    log_p = torch.log1p(-cdf)
    tail = x > q[-1]
    log_p_tail = np.log1p(-spec["tail_q"]) - (x - q[-1]) / spec["tail_scale"]
    log_p = torch.where(tail, log_p_tail, log_p)
    return torch.log(-torch.expm1(log_p)) - log_p


def set_edge_calibration(model, calib: dict | None) -> None:
    """Install (or clear, with ``None``) the per-edge benign references on ``model``."""
    model.edge_calib = calib
    model.__dict__.pop("_edge_calib_t", None)


@torch.no_grad()
def chain_edge_logits(model, groups, t, device) -> dict:
    """Raw per-edge anomaly logits of a block of events, from one shared expansion.

    ``groups`` lists the edge groups of a block of ``B`` events, one row per event, as
    ``(kind, src, dst, msg, aux_src)``: the edge kind (``EDGE_*``), global node-id tensors
    ``[B]``, the ``[B, msg_dim]`` edge messages, and the ``aux_src`` ids for the second
    history triplet (``None`` zero-pads it, see ``compute_hist_feats``). ``t`` holds the
    ``[B]`` event times. Returns ``{kind: [B] anomaly logits}`` (``-logit P(benign)``, see
    :func:`infer_logit`).

    One neighbour expansion over all endpoints and one GNN forward replace one of each per
    edge. Exact: an endpoint's embedding depends only on its own k-hop neighbourhood, which
    the shared expansion contains in full. Equality with per-edge :func:`infer_logit` is
    checked by ``tests/verify_replay_batching.py`` (offline replay) and
    ``tests/test_fusion_parity.py`` (``score_event``).

    Each group is still scored in its own ``model.score`` call. Concatenating them is the
    same maths, but it changes the row count of the time encoder's ``Linear(1, T)``, whose
    kernel path (FMA or not) then rounds ``w·Δt`` differently; at the never-seen sentinel
    ``Δt = delta_t_cap`` the argument of the cosine reaches ~1e5 rad, where one float32 ulp
    is ~0.06 rad, so the logit moved by ~1e-4 against the per-edge reference.
    """
    srcs = torch.cat([g[1] for g in groups])
    dsts = torch.cat([g[2] for g in groups])
    n_id, edge_index, hist_t, hist_msg = model.neighbor_loader(torch.cat([srcs, dsts]).unique())
    z = model.embed(n_id, edge_index, hist_t, hist_msg)
    assoc = model.neighbor_loader._assoc
    nf, h_idx = model.node_feat[n_id], model.node_hash[n_id]
    t_list = t.tolist()
    out = {}
    for kind, src, dst, msg, aux in groups:
        s_list, d_list = src.tolist(), dst.tolist()
        d_pair = model.pair_delta_t(s_list, d_list, t_list, device)
        d_src = model.src_delta_t(src, t, device)
        hist = model.compute_hist_feats(
            s_list, d_list, device, aux_src_ids=None if aux is None else aux.tolist()
        )
        out[kind] = -model.score(z, nf, h_idx, assoc[src], assoc[dst], msg, d_pair, d_src, hist)
    return out


def combine_edge_logits(model, edge_logits: dict) -> torch.Tensor:
    """Event anomaly logit: the max over its edges of the (benign-calibrated) edge logits.

    Without ``model.edge_calib`` (``edge_calibration=False``) this is the plain max.
    """
    out = None
    for kind, logit in edge_logits.items():
        c = calibrated_edge_logit(model, kind, logit)
        out = c if out is None else torch.maximum(out, c)
    return out


@torch.no_grad()
def chain_logits(model, groups, t, device) -> torch.Tensor:
    """Per-event anomaly logit of a block of events: :func:`chain_edge_logits` combined by
    :func:`combine_edge_logits`."""
    return combine_edge_logits(model, chain_edge_logits(model, groups, t, device))


@torch.no_grad()
def update_memory(model, src_idx: int, dst_idx: int, t_val: int, msg_vec, device,
                  aux_pair: tuple[int, int] | None = None) -> None:
    """Commit one edge into memory, neighbour store, recency and history counters.

    Callers commit only benign events (anti-poisoning gate). ``aux_pair`` bumps one more
    ``pair_count`` entry without a temporal edge: ``(device, dst)`` on the access edge, the
    per-device habituality counter read by ``compute_hist_feats``.
    """
    b_src, b_dst, b_t, b_msg = _event_tensors(src_idx, dst_idx, t_val, msg_vec, device)
    model.memory.update_state(b_src, b_dst, b_t, b_msg)
    model.memory.detach()
    model.neighbor_loader.insert(b_src, b_dst, b_t, b_msg)
    if not hasattr(model, "last_contact"):
        model.last_contact = {}
    model.last_contact[(src_idx, dst_idx)] = t_val
    # Interaction-history counters.
    if not hasattr(model, "pair_count"):
        model.pair_count, model.src_count = {}, {}
    model.pair_count[(src_idx, dst_idx)] = model.pair_count.get((src_idx, dst_idx), 0) + 1
    model.src_count[src_idx] = model.src_count.get(src_idx, 0) + 1
    if aux_pair is not None:
        model.pair_count[aux_pair] = model.pair_count.get(aux_pair, 0) + 1


def anomaly_score(anom_logit) -> float:
    """Convert an anomaly logit to the ``1 - P(benign)`` probability the thresholds use.

    float64 (``scipy.special.expit``) rather than the float32 ``1 - torch.sigmoid``: the
    latter returns exactly 1.0 from ``logit < -16`` on, collapsing the whole head of the
    ranking into one tie. Monotone, so it changes no ordering and no AUC.

    With a per-edge benign calibration the logit is ``log((1-p)/p)`` of the most surprising
    edge's benign p-value, so the score reads ``1 - p`` (before the precursor prior).
    """
    return expit(np.asarray(anom_logit, dtype=np.float64))


def precursor_shift(model, src_idx: int, t_val: int) -> float:
    """Additive anomaly-*logit* shift (>= 0.0) from the kill-chain precursor.

    Lateral movement is signal-clean; its one tell is a recent recon alert on the same
    entity. The memory gate drops that alert, so it lives in the time-decayed
    ``recent_alert`` state and acts as a serving-time prior (not a trained input:
    benign-only training would leave it dead). Additive on the logit, i.e. a fixed amount
    of evidence on the odds. Returns ``max_shift * 0.5**(Δt / half_life)`` while an alert
    is recent, else ``0.0``.
    """
    if not getattr(model, "use_precursor", False):
        return 0.0
    last = getattr(model, "recent_alert", {}).get(src_idx)
    if last is None:
        return 0.0
    dt = max(0.0, float(t_val) - float(last))
    decay = 0.5 ** (dt / model.precursor_half_life)
    return model.precursor_max_shift * decay


def record_alert(model, src_idx: int, t_val: int) -> None:
    """Arm the kill-chain precursor: remember that ``src_idx`` alerted at ``t_val``.

    Keeps the latest alert time, never an older one: a retried or late ``/deny`` /
    ``/update`` cannot move an entity's alert back in time, so recording is idempotent.
    """
    if not hasattr(model, "recent_alert"):
        model.recent_alert = {}
    last = model.recent_alert.get(src_idx)
    model.recent_alert[src_idx] = t_val if last is None else max(last, t_val)


def sensor_alarm(features) -> bool:
    """Whether the event's Snort probe (``features[1]``) fired. Recon precedes lateral
    movement, so a sensor alert arms the precursor on its own. Datasets whose message has
    no probe slot (e.g. LANL) never raise it."""
    return len(features) > 1 and float(features[1]) > 0.5


def event_alarm(score: float, *, flagged: bool, threshold_arm: float | None, features) -> bool:
    """Per-event alarm that arms the kill-chain precursor (see :func:`precursor_shift`).

    True when the event was flagged (at or above its decision threshold), armed
    (``score >= threshold_arm``: recon sits below the decision threshold, hence a lower
    one) or its sensor fired; ``threshold_arm=None`` disables arming. One rule for the
    offline replay and every serving path.
    """
    armed = threshold_arm is not None and score >= threshold_arm
    return bool(flagged) or armed or sensor_alarm(features)


def signal_dirty(features) -> bool:
    """True when the edge signal already fires: broken TLS trust (``ja3 == 0``) or a sensor.

    Observable at serving time, unlike the anomaly class, so the decision threshold is
    routed on it: dirty events keep the conservative FPR threshold (the rule baseline
    catches them), signal-clean ones get the recall-oriented threshold. ``features`` is
    ``[ja3, s1, s2, s3, method, role, clearance]``; ``features[4]`` is not a sensor.
    """
    ja3, s1, s2, s3 = (float(features[i]) for i in range(4))
    return ja3 <= 0.5 or s1 > 0.5 or s2 > 0.5 or s3 > 0.5


def _reset_slot(model, idx: int) -> None:
    """Cold-start a reused memory slot after eviction."""
    with torch.no_grad():
        model.memory.memory[idx].zero_()
        model.memory.last_update[idx] = 0
        model.node_feat[idx].zero_()
        model.node_feat[idx, 14] = 1.0  # trust: neutral constant
        model.node_hash[idx].zero_()
    # Re-seed (do not delete) the slot's PyG message store: TGNMemory reads msg_store[i]
    # for every node it touches. Same format as TGNMemory._reset_message_store.
    mem = model.memory
    empty_i = mem.memory.new_empty((0,), dtype=torch.long)
    empty_msg = mem.memory.new_empty((0, mem.raw_msg_dim))
    mem.msg_s_store[idx] = (empty_i, empty_i, empty_i, empty_msg)
    mem.msg_d_store[idx] = (empty_i, empty_i, empty_i, empty_msg)
    if hasattr(model, "last_contact"):
        keys_to_delete = [k for k in model.last_contact if k[0] == idx or k[1] == idx]
        for k in keys_to_delete:
            del model.last_contact[k]
    # A reused index must not inherit the evicted entity's counters, alerts or neighbours.
    if hasattr(model, "pair_count"):
        for k in [k for k in model.pair_count if k[0] == idx or k[1] == idx]:
            del model.pair_count[k]
    if hasattr(model, "src_count"):
        model.src_count.pop(idx, None)
    if hasattr(model, "recent_alert"):
        model.recent_alert.pop(idx, None)
    model.neighbor_loader.reset_node(idx)


def _set_node_features(model, idx: int, feat, device) -> None:
    """Write a node's static features into its slot, keeping its trust column (14)."""
    with torch.no_grad():
        trust = model.node_feat[idx, 14].item()
        model.node_feat[idx] = torch.as_tensor(
            feat, dtype=model.node_feat.dtype, device=device
        )
        model.node_feat[idx, 14] = trust


def _set_source_network_feature(model, idx: int, key_source) -> None:
    """Set the internal/external bit (``node_feat[5]``) of a runtime-admitted source node.

    A model feature derived from the ``src:`` IP, never an authz gate; no-op when the
    checkpoint was trained without it.
    """
    if not getattr(model, "use_source_internal", False):
        return
    with torch.no_grad():
        model.node_feat[idx, 5] = 1.0 if ip_is_internal(key_source) else 0.0


def _admit(model, registry: NodeRegistry, key: Hashable) -> int:
    """Map an external key to its memory slot, admitting (and hashing) it if unseen."""
    idx, is_new = registry.get_or_add(
        key, recency=model.memory.last_update,
        on_evict=lambda i: _reset_slot(model, i),
    )
    if is_new:
        model.node_hash[idx] = stable_hash(key, model.hash_emb.num_embeddings)
    return idx


def score_event(
    model,
    registry: NodeRegistry,
    threshold: float,
    key_user: Hashable,
    key_device: Hashable | None,
    key_dst: Hashable,
    timestamp: int,
    features,
    device,
    *,
    key_source: Hashable | None = None,
    key_config: Hashable | None = None,
    threshold_dirty=None,
    user_feat=None,
    device_feat=None,
    dst_feat=None,
    update: bool = True,
    guest_device_fallback: bool = False,
) -> tuple[float, bool, float]:
    """Score one access event; with ``update``, commit it only if judged benign.

    Keys are mapped through ``registry`` (unseen ones admitted). Scored chain:
    ``source -> config -> device -> user -> dst`` plus ``config -> user``; binding edges
    carry zero messages, the access edge carries ``features``, and the event logit is the
    max of the benign-calibrated edge logits. ``key_source`` and ``key_device`` are optional
    (a missing one drops its own edges); ``key_config`` defaults to ``"conf:guest"``.

    ``user_feat`` / ``device_feat`` / ``dst_feat`` overwrite that node's static features
    (type-specific: the device tier lives on device nodes, user nodes carry none, role and
    clearance travel in the message); ``None`` keeps the slot's current features.

    Returns ``(anomaly_score, is_anomaly, effective_threshold)``: ``threshold_dirty`` for
    signal-dirty events, ``threshold`` otherwise.
    """
    model.eval()
    if key_config is None:
        key_config = "conf:guest"
    if guest_device_fallback:
        key_device = to_guest_device(key_device)
    user_idx = _admit(model, registry, key_user)
    device_idx = None if key_device is None else _admit(model, registry, key_device)
    dst_idx = _admit(model, registry, key_dst)
    source_idx = None if key_source is None else _admit(model, registry, key_source)
    config_idx = _admit(model, registry, key_config)

    if user_feat is not None:
        _set_node_features(model, user_idx, user_feat, device)
    if device_feat is not None and device_idx is not None:
        _set_node_features(model, device_idx, device_feat, device)
    if dst_feat is not None:
        _set_node_features(model, dst_idx, dst_feat, device)
    if source_idx is not None:
        _set_source_network_feature(model, source_idx, key_source)

    features_bind = [0.0] * len(features)
    # Causal chain source → config → device → user → resource, plus config → user.
    # The config node is always present; device / source bindings are skipped if absent
    # (config → user then bridges the chain when the device is missing).
    def _ids(idx):
        return torch.tensor([idx], dtype=torch.long, device=device)

    msg = torch.as_tensor(features, dtype=torch.float, device=device).reshape(1, -1)
    zeros = torch.zeros_like(msg)
    b_user, b_cfg = _ids(user_idx), _ids(config_idx)
    b_dev = None if device_idx is None else _ids(device_idx)
    groups = [(EDGE_ACCESS, b_user, _ids(dst_idx), msg, b_dev),
              (EDGE_CFG_USER, b_cfg, b_user, zeros, None)]
    if b_dev is not None:
        groups += [(EDGE_CFG_DEV, b_cfg, b_dev, zeros, None),
                   (EDGE_DEV_USER, b_dev, b_user, zeros, None)]
    if source_idx is not None:
        groups.append((EDGE_SRC_CFG, _ids(source_idx), b_cfg, zeros, None))

    raw_logit = float(chain_logits(model, groups, _ids(int(timestamp)), device))
    # Kill-chain precursor prior: keyed on the device node if present, else on the user
    boost_idx = device_idx if device_idx is not None else user_idx
    score = float(anomaly_score(raw_logit + precursor_shift(model, boost_idx, timestamp)))
    eff_threshold = threshold
    if threshold_dirty is not None and signal_dirty(features):
        eff_threshold = threshold_dirty
    is_anomaly = score >= eff_threshold

    if update and event_alarm(score, flagged=is_anomaly, threshold_arm=model.threshold_arm,
                              features=features):
        record_alert(model, boost_idx, timestamp)

    if update and not is_anomaly:
        # Commit order mirrors the causal chain: source→config, config→device,
        # config→user, device→user, user→resource.
        if source_idx is not None:
            update_memory(model, source_idx, config_idx, timestamp, features_bind, device)
        if device_idx is not None:
            update_memory(model, config_idx, device_idx, timestamp, features_bind, device)
        update_memory(model, config_idx, user_idx, timestamp, features_bind, device)
        if device_idx is not None:
            update_memory(model, device_idx, user_idx, timestamp, features_bind, device)
        update_memory(model, user_idx, dst_idx, timestamp, features, device,
                      aux_pair=(device_idx, dst_idx) if device_idx is not None else None)

    return score, is_anomaly, eff_threshold


def commit_event(
    model,
    registry: NodeRegistry,
    key_user: Hashable,
    key_device: Hashable | None,
    key_dst: Hashable,
    timestamp: int,
    features,
    device,
    *,
    key_source: Hashable | None = None,
    key_config: Hashable | None = None,
    user_feat=None,
    device_feat=None,
    dst_feat=None,
    guest_device_fallback: bool = False,
    alarm: bool = False,
) -> None:
    """Commit an event the caller judged benign (e.g. OPA ALLOW), without re-scoring.

    Admits keys (evicting LRU on overflow), refreshes the given static features and
    advances memory and neighbour history along the same chain as :func:`score_event`.
    ``alarm`` is the :func:`event_alarm` of the scoring step: OPA may ALLOW an event the
    model flagged, which is exactly the case the precursor exists for. Every scored event
    ends in this function or in :func:`deny_event`.
    """
    model.eval()
    if key_config is None:
        key_config = "conf:guest"
    if guest_device_fallback:
        key_device = to_guest_device(key_device)
    user_idx = _admit(model, registry, key_user)
    device_idx = None if key_device is None else _admit(model, registry, key_device)
    dst_idx = _admit(model, registry, key_dst)
    source_idx = None if key_source is None else _admit(model, registry, key_source)
    config_idx = _admit(model, registry, key_config)

    if user_feat is not None:
        _set_node_features(model, user_idx, user_feat, device)
    if device_feat is not None and device_idx is not None:
        _set_node_features(model, device_idx, device_feat, device)
    if dst_feat is not None:
        _set_node_features(model, dst_idx, dst_feat, device)
    if source_idx is not None:
        _set_source_network_feature(model, source_idx, key_source)

    if alarm or sensor_alarm(features):
        record_alert(model, device_idx if device_idx is not None else user_idx, timestamp)

    features_bind = [0.0] * len(features)
    # Commit order: source→config, config→device, config→user, device→user, user→resource.
    if source_idx is not None:
        update_memory(model, source_idx, config_idx, timestamp, features_bind, device)
    if device_idx is not None:
        update_memory(model, config_idx, device_idx, timestamp, features_bind, device)
    update_memory(model, config_idx, user_idx, timestamp, features_bind, device)
    if device_idx is not None:
        update_memory(model, device_idx, user_idx, timestamp, features_bind, device)
    update_memory(model, user_idx, dst_idx, timestamp, features, device,
                  aux_pair=(device_idx, dst_idx) if device_idx is not None else None)


def deny_event(
    model,
    registry: NodeRegistry,
    key_user: Hashable,
    key_device: Hashable | None,
    timestamp: int,
    features,
    *,
    guest_device_fallback: bool = False,
    alarm: bool = False,
) -> None:
    """Record an event the caller DENYed: alert state only, never the baseline.

    Without it a denied recon event would never arm the kill-chain precursor. The DENY is
    not evidence by itself: only ``alarm`` (from ``/infer``) or the sensor records an alert,
    and only the alerted entity (device, else user) is admitted.
    """
    if not (alarm or sensor_alarm(features)):
        return
    if guest_device_fallback:
        key_device = to_guest_device(key_device)
    actor = _admit(model, registry, key_device if key_device is not None else key_user)
    record_alert(model, actor, timestamp)


def save_model(model, registry: NodeRegistry, threshold: float, hp: dict,
               checkpoint_path, stats_path, *, threshold_dirty=None,
               calibration=None, operating_point=None) -> None:
    """Persist the deployable artifact: weights, memory and message stores, recency and
    history counters, alert state, neighbour buffers, edge calibration (checkpoint) and
    registry plus thresholds (stats JSON).

    ``threshold`` is the signal-clean (cost-sensitive) threshold, ``threshold_dirty`` the
    conservative one for signal-dirty events (defaults to ``threshold``).
    ``calibration`` / ``operating_point`` are optional provenance.
    """
    checkpoint_path = Path(checkpoint_path)
    stats_path = Path(stats_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # state_dict carries the `memory` / `last_update` / `_assoc` buffers; the raw
    # message store is a plain dict (not a buffer) so it is saved alongside.
    tmp_checkpoint = checkpoint_path.with_suffix('.pt.tmp')
    torch.save(
        {
            "model": model.state_dict(),
            "msg_s_store": model.memory.msg_s_store,
            "msg_d_store": model.memory.msg_d_store,
            "last_contact": getattr(model, "last_contact", {}),
            "pair_count": getattr(model, "pair_count", {}),
            "src_count": getattr(model, "src_count", {}),
            "recent_alert": getattr(model, "recent_alert", {}),
            "neighbor_loader": model.neighbor_loader.state(),
            "edge_calib": getattr(model, "edge_calib", None),
            "hyperparams": hp,
        },
        tmp_checkpoint,
    )
    tmp_checkpoint.replace(checkpoint_path)

    stats = {
        "threshold": float(threshold),
        "threshold_dirty": float(threshold_dirty if threshold_dirty is not None else threshold),
        # Kill-chain arm threshold (see event_alarm); read back onto the model by load_model.
        "threshold_arm": model.threshold_arm,
        "target_fpr": hp.get("target_fpr"),
        "capacity": int(hp["capacity"]),
        "registry": registry.to_dict(),
    }
    if calibration is not None:
        stats["calibration"] = calibration
    if operating_point is not None:
        stats["operating_point"] = operating_point
    tmp_stats = stats_path.with_suffix('.json.tmp')
    with open(tmp_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    tmp_stats.replace(stats_path)


def load_model(checkpoint_path, stats_path, device):
    """Reconstruct ``(model, registry, threshold, threshold_dirty, hp)`` for serving.

    ``threshold`` is the signal-clean (cost-sensitive) decision threshold and
    ``threshold_dirty`` the conservative one for signal-dirty events; old artifacts without
    a dirty threshold fall back to ``threshold`` (single-threshold behaviour). ``hp`` (the
    saved hyper-parameter dict) is returned so a long-running server can later re-persist the
    evolved state via :func:`save_model` without re-reading the checkpoint.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hp = ckpt["hyperparams"]
    model = build_model(hp, device)
    model.load_state_dict(ckpt["model"])
    # Restore pending raw messages so memory continuation is exact.
    model.memory.msg_s_store = ckpt.get("msg_s_store", {})
    model.memory.msg_d_store = ckpt.get("msg_d_store", {})
    model.last_contact = ckpt.get("last_contact", {})
    model.pair_count = ckpt.get("pair_count", {})
    model.src_count = ckpt.get("src_count", {})
    model.recent_alert = ckpt.get("recent_alert", {})
    # Per-edge benign references; absent in older artifacts -> plain max of raw logits.
    set_edge_calibration(model, ckpt.get("edge_calib"))
    # Restore the temporal neighbour buffers (map_location already placed the saved
    # tensors on ``device``); build_model created an empty loader of the right shape.
    if "neighbor_loader" in ckpt:
        model.neighbor_loader.load_state(ckpt["neighbor_loader"])
    model.eval()

    with open(stats_path, encoding="utf-8") as f:
        stats = json.load(f)
    registry = NodeRegistry.from_dict(stats["registry"])
    # Artifacts without an arm threshold keep arming on the decision alone (flagged / sensor).
    arm = stats.get("threshold_arm")
    model.threshold_arm = None if arm is None else float(arm)
    threshold = float(stats["threshold"])
    threshold_dirty = float(stats.get("threshold_dirty", threshold))
    return model, registry, threshold, threshold_dirty, hp
