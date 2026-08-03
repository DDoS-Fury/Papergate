"""Serving / persistence layer for the streaming TGN.

This module is the single source of truth for the *real-time* code path.

Relationship to the offline evaluation: ``train_tgn._replay`` does **not** call these
primitives — it is an independent vectorised re-implementation that scores a block of
events against one shared neighbour expansion. What ties the two together is a checked
equivalence, not shared code: ``tests/verify_replay_batching.py`` replays the same stream
through ``infer_score`` / ``update_memory`` event by event and asserts agreement with
``_replay(batch_size=1)`` to 1e-5, on both the v4 five-edge chain and the legacy v3 one.
Treat that harness as part of the contract: any change to one path must keep it green.

One divergence is deliberate and is *not* a bug: the offline replay commits on the
OPA-ALLOW proxy (``not signal_dirty``), whereas :func:`score_event` with ``update=True``
commits on the model's own score. See the ``_replay`` docstring for why measuring under
the OPA-in-the-loop gate is the conservative choice.

- :func:`infer_score` — score one event (read memory, no mutation).
- :func:`update_memory` — commit one event into the TGN memory.

- :func:`infer_score` — score one event (read memory, no mutation).
- :func:`update_memory` — commit one event into the TGN memory.
- :func:`score_event`  — the high-level online API: map external entity keys
  through a :class:`NodeRegistry`, score, and update memory **only for events that
  look benign** (anti-poisoning gate). Anomalous events are reported but never
  written into the baseline.
- :func:`commit_event` — the unconditional commit used when the benign/anomalous
  decision is made outside the model (e.g. OPA): map keys and advance memory +
  neighbour history without re-scoring.
- :func:`apply_feedback` — precursor arming + trust nudge, shared by both paths.
- :func:`save_model` / :func:`load_model` — persist and restore weights, the TGN
  memory buffers, the (non-state_dict) raw-message store, the entity registry and
  the calibrated decision threshold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Hashable

import numpy as np
import torch

from graphagate.config import TGNConfig
from graphagate.model.registry import NodeRegistry
from graphagate.model.tgn import ZTATemporalGraphNetwork, stable_hash
from graphagate.netclass import ip_is_internal, to_guest_device


SCHEMA_VERSION = 4  # 5-node schema: config (JA3) node — source→config→device→user→resource (+ config→user)


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
    # Kill-chain precursor prior knobs (serving-time; not in the state_dict).
    # Fallbacks track TGNConfig, not the historical 100000.0 / 3.0 that saturated scores.
    model.precursor_half_life = float(hp.get("precursor_half_life", TGNConfig.precursor_half_life))
    model.precursor_max_boost = float(hp.get("precursor_max_boost", TGNConfig.precursor_max_boost))
    # v3 static-feature toggles: must match the values the checkpoint was trained with,
    # else the source node would carry a feature the model never learned from.
    model.use_source_internal = bool(hp.get("use_source_internal", False))
    # The bounded neighbour loader lives outside the state_dict; build it on the
    # serving device so its buffers match the model's. Its contents are restored
    # from the checkpoint in load_model.
    model.init_neighbor_loader(int(hp.get("neighbor_size", 10)), device)
    # Switch to eval *before* the caller restores buffers. PyG's TGNMemory flushes
    # its pending message store into the memory buffer on the train->eval transition
    # (see TGNMemory.train). If we let that happen *after* load_state_dict has already
    # restored the (fully materialised) memory + the saved message store, the messages
    # are applied a second time and the reloaded memory diverges from the original.
    # Evaluating here, while the store is still empty, makes the later restore exact.
    model.eval()
    return model


def _event_tensors(src_idx: int, dst_idx: int, t_val: int, msg_vec, device):
    b_src = torch.tensor([src_idx], dtype=torch.long, device=device)
    b_dst = torch.tensor([dst_idx], dtype=torch.long, device=device)
    # TGNMemory.last_update is int64 — timestamps stay integer end-to-end.
    b_t = torch.tensor([int(t_val)], dtype=torch.long, device=device)
    b_msg = torch.as_tensor(msg_vec, dtype=torch.float, device=device).reshape(1, -1)
    return b_src, b_dst, b_t, b_msg


@torch.no_grad()
def infer_score(model, src_idx: int, dst_idx: int, t_val: int, msg_vec, device,
                aux_src_idx: int | None = None) -> float:
    """Return the anomaly score (1 - P(benign)) for a single edge.

    Does **not** mutate memory or the neighbour loader. The two endpoints are
    expanded to their stored temporal neighbourhood so the embedding reflects each
    entity's recent interaction history (the structural signal for lateral movement);
    a cold-start node with no neighbours falls back to its memory state alone.

    ``aux_src_idx`` feeds the second interaction-history triplet (the per-device
    habituality counters on the access edge — see ``compute_hist_feats``); ``None``
    zero-pads it (binding edges / device-less datasets).
    """
    b_src, b_dst, b_t, b_msg = _event_tensors(src_idx, dst_idx, t_val, msg_vec, device)
    nodes = torch.unique(torch.cat([b_src, b_dst]))
    n_id, edge_index, hist_t, hist_msg = model.neighbor_loader(nodes)
    assoc = model.neighbor_loader._assoc
    
    # Both recencies go through the model's shared encoder (cap + never-seen sentinel),
    # which is what keeps the serving path on the same Δt distribution as training.
    delta_t = model.pair_delta_t([src_idx], [dst_idx], [t_val], device)
    delta_t_src = model.src_delta_t([src_idx], [float(t_val)], device)

    # Explicit interaction-history features for this src→dst pair (read-only; counts are
    # advanced in update_memory, exactly mirroring the train-time predict-then-update order).
    aux_ids = None if aux_src_idx is None else [aux_src_idx]
    hist_feats = model.compute_hist_feats([src_idx], [dst_idx], device, aux_src_ids=aux_ids)

    out = model(
        n_id, edge_index, hist_t, hist_msg, assoc[b_src], assoc[b_dst], b_msg, delta_t, delta_t_src, hist_feats
    ).squeeze(-1)
    prob_benign = torch.sigmoid(out).item()
    return 1.0 - prob_benign


@torch.no_grad()
def update_memory(model, src_idx: int, dst_idx: int, t_val: int, msg_vec, device,
                  aux_pair: tuple[int, int] | None = None) -> None:
    """Commit a single edge into the TGN memory and the neighbour store.

    Inserting into the neighbour loader only here (and only for events the caller
    has judged benign) keeps the anti-poisoning gate intact: anomalous events never
    enter an entity's history.

    ``aux_pair`` advances an extra ``pair_count`` entry without a temporal edge: on
    the access-edge commit it is ``(device_idx, dst_idx)``, the per-device resource
    habituality counter read back by ``compute_hist_feats``'s second triplet (the
    device's own activity count is kept by its binding-edge commit, so only the pair
    counter is bumped here).
    """
    b_src, b_dst, b_t, b_msg = _event_tensors(src_idx, dst_idx, t_val, msg_vec, device)
    model.memory.update_state(b_src, b_dst, b_t, b_msg)
    model.memory.detach()
    model.neighbor_loader.insert(b_src, b_dst, b_t, b_msg)
    if not hasattr(model, "last_contact"):
        model.last_contact = {}
    model.last_contact[(src_idx, dst_idx)] = t_val
    # Advance the interaction-history counters (benign-gated: update_memory is only
    # called for events judged benign, so anomalies never inflate an entity's history).
    if not hasattr(model, "pair_count"):
        model.pair_count, model.src_count = {}, {}
    model.pair_count[(src_idx, dst_idx)] = model.pair_count.get((src_idx, dst_idx), 0) + 1
    model.src_count[src_idx] = model.src_count.get(src_idx, 0) + 1
    if aux_pair is not None:
        model.pair_count[aux_pair] = model.pair_count.get(aux_pair, 0) + 1


def precursor_boost(model, src_idx: int, t_val: int) -> float:
    """Multiplicative anomaly-score factor (>= 1.0) from the kill-chain precursor.

    Lateral movement is signal-clean and feature-identical to a benign non-habitual
    access; the one tell is that it follows a recon alert (Snort / detected anomaly) on
    the SAME entity. The predict-then-update memory gate drops that anomalous precursor
    from the TGN memory, so we carry it in a separate, time-decayed per-entity
    ``recent_alert`` state and apply it as a SERVING-TIME prior — never a trained input
    (benign-only training would leave it a dead feature).

    Multiplicative (not additive) on purpose: it lifts an already-suspicious score over
    the threshold without turning near-zero benign scores into false positives. Returns
    ``1 + max_boost * 0.5**(Δt / half_life)`` while an alert is recent, else ``1.0``.
    """
    if not getattr(model, "use_precursor", False):
        return 1.0
    last = getattr(model, "recent_alert", {}).get(src_idx)
    if last is None:
        return 1.0
    dt = max(0.0, float(t_val) - float(last))
    decay = 0.5 ** (dt / model.precursor_half_life)
    return 1.0 + model.precursor_max_boost * decay


def record_alert(model, src_idx: int, t_val: int) -> None:
    """Arm the kill-chain precursor: remember that ``src_idx`` just alerted at ``t_val``."""
    if not hasattr(model, "recent_alert"):
        model.recent_alert = {}
    model.recent_alert[src_idx] = t_val


TRUST = 14  # runtime trust slot in node_feat


def apply_feedback(model, user_idx: int, device_idx: int | None, timestamp: int, features,
                   *, flagged: bool, boost_idx: int | None = None) -> None:
    """Per-event alarm feedback: arm the kill-chain precursor and nudge the trust feature.

    ``flagged`` is the decision the *system* reached for this event — the model's own
    ``is_anomaly`` on the internal-gate path, or the caller's verdict on the
    ``/infer`` → OPA → ``/update`` path. A sensor alert in ``features`` (Snort probe)
    counts as an alarm on its own, since recon precedes lateral movement.

    Both serving paths must call this. Previously only ``score_event(update=True)`` did,
    so on the documented two-step flow — which is ``score_event(update=False)`` followed
    by ``commit_event`` — ``recent_alert`` stayed empty, ``precursor_boost`` always
    returned 1.0 and the trust column stayed frozen at its checkpoint value: two inputs
    that contribute offline were simply absent in deployment.
    """
    if boost_idx is None:
        boost_idx = device_idx if device_idx is not None else user_idx
    alarm = bool(flagged) or (len(features) > 1 and features[1] > 0.5)
    targets = [user_idx] if device_idx is None else [device_idx, user_idx]
    # Datasets whose node_feat is narrower than the ZTA schema (e.g. the LANL adapter)
    # have no trust column; the precursor still applies.
    if model.node_feat.size(1) <= TRUST:
        if alarm:
            record_alert(model, boost_idx, timestamp)
        return
    if alarm:
        record_alert(model, boost_idx, timestamp)
        for idx in targets:
            model.node_feat[idx, TRUST] = max(0.0, model.node_feat[idx, TRUST].item() - 0.5)
    else:
        for idx in targets:
            model.node_feat[idx, TRUST] = min(1.0, model.node_feat[idx, TRUST].item() + 0.01)


def signal_dirty(features) -> bool:
    """Whether an event's edge signal already fires (the observable, class-free split).

    ``features`` is the message vector ``[ja3, s1, s2, s3, method, role, clearance]``:
    the signal is "dirty" when TLS trust is broken (``ja3==0``) or any Snort/sensor probe
    fires — the same condition as the rule baseline. (``features[4]`` is the HTTP method,
    NOT a sensor: including it — as an earlier revision did — silently routed every
    non-GET request to the conservative threshold.) The true anomaly class is unknown at
    serving time, but this *is* observable, so the decision threshold is routed on it:
    dirty events keep the conservative FPR threshold (the cheap rule already catches
    them), while signal-clean events — where lateral movement is indistinguishable from
    benign except by temporal pattern — get the recall-oriented cost-sensitive
    threshold. See ``score_event``.
    """
    ja3, s1, s2, s3 = (float(features[i]) for i in range(4))
    return ja3 <= 0.5 or s1 > 0.5 or s2 > 0.5 or s3 > 0.5


def _reset_slot(model, idx: int) -> None:
    """Cold-start a reused memory slot after eviction."""
    with torch.no_grad():
        model.memory.memory[idx].zero_()
        model.memory.last_update[idx] = 0
        model.node_feat[idx].zero_()
        model.node_feat[idx, 14] = 1.0  # Reset Trust Score to max
        model.node_hash[idx].zero_()
    # Reset (do NOT delete) the slot's PyG message store. TGNMemory seeds *every*
    # node id with an empty-message tuple and `_compute_msg` does `msg_store[i]` for
    # every node it touches — so popping the entry makes a reused (post-eviction)
    # slot raise `KeyError: idx` on its first event. Re-seed the empty tuple in the
    # exact format of TGNMemory._reset_message_store: (src, dst, t, msg).
    mem = model.memory
    empty_i = mem.memory.new_empty((0,), dtype=torch.long)
    empty_msg = mem.memory.new_empty((0, mem.raw_msg_dim))
    mem.msg_s_store[idx] = (empty_i, empty_i, empty_i, empty_msg)
    mem.msg_d_store[idx] = (empty_i, empty_i, empty_i, empty_msg)
    if hasattr(model, "last_contact"):
        keys_to_delete = [k for k in model.last_contact.keys() if k[0] == idx or k[1] == idx]
        for k in keys_to_delete:
            del model.last_contact[k]
    # Purge the recycled slot's interaction-history counters too, so a reused index
    # cannot inherit the evicted entity's habituality.
    if hasattr(model, "pair_count"):
        for k in [k for k in model.pair_count if k[0] == idx or k[1] == idx]:
            del model.pair_count[k]
    if hasattr(model, "src_count"):
        model.src_count.pop(idx, None)
    if hasattr(model, "recent_alert"):
        model.recent_alert.pop(idx, None)
    # Scrub the reused slot's temporal neighbourhood so a recycled index can't
    # inherit the evicted entity's interaction history.
    model.neighbor_loader.reset_node(idx)


def _set_node_features(model, idx: int, feat, device) -> None:
    """Write a node's static attributes (role / clearance / tier) into its slot."""
    with torch.no_grad():
        trust = model.node_feat[idx, 14].item()
        model.node_feat[idx] = torch.as_tensor(
            feat, dtype=model.node_feat.dtype, device=device
        )
        model.node_feat[idx, 14] = trust


def _set_source_network_feature(model, idx: int, key_source) -> None:
    """Write the source network's internal/external bit into node_feat index 5.

    Derived from the (``src:``-namespaced) client IP — a model feature, never an authz
    gate. Source nodes are admitted at runtime, so unlike the preregistered resources
    their static features must be set here on each event rather than baked at train time.
    No-op when the checkpoint was trained without the source-internal feature.
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
) -> tuple[float, bool]:
    """Score one streaming access event (v4 schema: up to 5 edges per request).

    Maps the (possibly unseen) entity keys through ``registry`` and scores the causal
    chain ``key_source -> key_config -> key_device -> key_user -> key_dst`` plus the
    ``key_config -> key_user`` binding: the four binding edges carry zero messages, the
    access edge carries ``features``; the anomaly score is the max over the edges.
    ``key_source`` (the client IP) and ``key_device`` are optional — a missing one skips
    only its own binding edges (a key is never aliased onto two roles). ``key_config``
    (the client's TLS/JA3 fingerprint) defaults to the generic ``"conf:guest"`` when the
    collector cannot resolve it, so the config node is always present. When ``update`` is
    set the event is written into memory **only if it is classified benign**
    (``score < threshold``), keeping the baseline free of attacker-controlled events.

    ``user_feat`` / ``device_feat`` / ``dst_feat`` are the per-node static
    attributes, written into the matching node slot. They are kept **separate by
    node type** because the training feature contract is type-specific: the device
    tier lives in ``node_feat[2]`` of DEVICE nodes only, while user nodes carry no
    static attributes (role / clearance travel in the edge message, never the node
    features). Collapsing them onto a single ``src_feat`` applied to both endpoints
    pushes the user node out of distribution — the train/serve skew that this split
    fixes. In a ZTA deployment the orchestrator/OPA already holds these per request,
    so they are supplied per event — no extra data store is required. When a vector
    is omitted, that slot keeps whatever features it already has (e.g. those learned
    for preregistered entities at train time).

    Returns ``(anomaly_score, is_anomaly)``.
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
    edge_scores = [
        infer_score(model, user_idx, dst_idx, timestamp, features, device,
                    aux_src_idx=device_idx),
        infer_score(model, config_idx, user_idx, timestamp, features_bind, device),
    ]
    if device_idx is not None:
        edge_scores.append(
            infer_score(model, config_idx, device_idx, timestamp, features_bind, device)
        )
        edge_scores.append(
            infer_score(model, device_idx, user_idx, timestamp, features_bind, device)
        )
    if source_idx is not None:
        edge_scores.append(
            infer_score(model, source_idx, config_idx, timestamp, features_bind, device)
        )

    raw_score = max(edge_scores)
    # Kill-chain precursor prior — keyed on the DEVICE node (if present), else on USER
    boost_idx = device_idx if device_idx is not None else user_idx
    score = min(1.0, raw_score * precursor_boost(model, boost_idx, timestamp))
    eff_threshold = threshold
    if threshold_dirty is not None and signal_dirty(features):
        eff_threshold = threshold_dirty
    is_anomaly = score >= eff_threshold

    if update:
        apply_feedback(model, user_idx, device_idx, timestamp, features,
                       flagged=is_anomaly, boost_idx=boost_idx)

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

    return score, is_anomaly


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
    flagged: bool = False,
) -> None:
    """Commit an event the caller has already judged benign (e.g. OPA returned ALLOW).

    This is the *unconditional* counterpart of :func:`score_event`: it maps the entity
    keys through ``registry`` (admitting unseen entities, evicting LRU on overflow),
    optionally refreshes their static attributes, and advances both the TGN memory and
    the neighbour history along the same up-to-5-edge chain (``key_config`` defaults to
    ``"conf:guest"`` when omitted). Use it for the two-step anti-poisoning flow where the
    benign/anomalous decision is made *outside* the model (score with
    :func:`infer_score` / :func:`score_event` ``update=False`` first, then commit here
    only on approval).

    ``flagged`` carries back what the scoring step decided: OPA may ALLOW an event the
    model still flagged, and that is precisely the case the kill-chain precursor exists
    for. It drives :func:`apply_feedback` (precursor + trust), which this function used
    not to run at all — leaving both signals inert on the recommended flow.
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

    apply_feedback(model, user_idx, device_idx, timestamp, features, flagged=flagged)

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


def save_model(model, registry: NodeRegistry, threshold: float, hp: dict,
               checkpoint_path, stats_path, *, threshold_dirty=None,
               calibration=None, operating_point=None) -> None:
    """Persist the deployable artifact (weights + memory + registry + threshold).

    ``threshold`` is the primary (signal-clean / cost-sensitive) decision threshold;
    ``threshold_dirty`` is the conservative threshold for events whose edge signal already
    fires (defaults to ``threshold`` when omitted, so old single-threshold artifacts stay
    valid). ``calibration`` / ``operating_point`` are optional provenance for observability.
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
            "hyperparams": hp,
        },
        tmp_checkpoint,
    )
    tmp_checkpoint.replace(checkpoint_path)

    stats = {
        "threshold": float(threshold),
        "threshold_dirty": float(threshold_dirty if threshold_dirty is not None else threshold),
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
    # Restore the temporal neighbour buffers (map_location already placed the saved
    # tensors on ``device``); build_model created an empty loader of the right shape.
    if "neighbor_loader" in ckpt:
        model.neighbor_loader.load_state(ckpt["neighbor_loader"])
    model.eval()

    with open(stats_path, encoding="utf-8") as f:
        stats = json.load(f)
    registry = NodeRegistry.from_dict(stats["registry"])
    threshold = float(stats["threshold"])
    threshold_dirty = float(stats.get("threshold_dirty", threshold))
    return model, registry, threshold, threshold_dirty, hp
