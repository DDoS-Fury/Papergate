"""Serving / persistence layer for the streaming TGN.

This module is the single source of truth for the *real-time* code path. Both the
offline evaluation in :mod:`graphagate.train_tgn` and any online deployment call
the same primitives here, so the behaviour that is measured is exactly the
behaviour that is served:

- :func:`infer_score` — score one event (read memory, no mutation).
- :func:`update_memory` — commit one event into the TGN memory.
- :func:`score_event`  — the high-level online API: map external entity keys
  through a :class:`NodeRegistry`, score, and update memory **only for events that
  look benign** (anti-poisoning gate). Anomalous events are reported but never
  written into the baseline.
- :func:`commit_event` — the unconditional commit used when the benign/anomalous
  decision is made outside the model (e.g. OPA): map keys and advance memory +
  neighbour history without re-scoring.
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

from graphagate.model.registry import NodeRegistry
from graphagate.model.tgn import ZTATemporalGraphNetwork, stable_hash


def build_model(hp: dict, device: torch.device) -> ZTATemporalGraphNetwork:
    """Instantiate the model from a hyper-parameter dict (see :func:`save_model`)."""
    model = ZTATemporalGraphNetwork(
        num_nodes=int(hp["capacity"]),
        node_feat_dim=int(hp["node_feat_dim"]),
        msg_dim=int(hp["msg_dim"]),
        memory_dim=int(hp["memory_dim"]),
        time_dim=int(hp["time_dim"]),
        num_hops=int(hp.get("num_hops", 2)),
        hash_buckets=int(hp.get("hash_buckets", 10000)),
        hash_dim=int(hp.get("hash_dim", 16)),
        hist_feat_dim=int(hp.get("hist_feat_dim", 3)),
    ).to(device)
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
def infer_score(model, src_idx: int, dst_idx: int, t_val: int, msg_vec, device) -> float:
    """Return the anomaly score (1 - P(benign)) for a single event.

    Does **not** mutate memory or the neighbour loader. The two endpoints are
    expanded to their stored temporal neighbourhood so the embedding reflects each
    entity's recent interaction history (the structural signal for lateral movement);
    a cold-start node with no neighbours falls back to its memory state alone.
    """
    b_src, b_dst, b_t, b_msg = _event_tensors(src_idx, dst_idx, t_val, msg_vec, device)
    nodes = torch.unique(torch.cat([b_src, b_dst]))
    n_id, edge_index, hist_t, hist_msg = model.neighbor_loader(nodes)
    assoc = model.neighbor_loader._assoc
    
    last_t = getattr(model, "last_contact", {}).get((src_idx, dst_idx), 0)
    delta_t_val = float(t_val - last_t)
    delta_t = torch.tensor([delta_t_val], dtype=torch.float, device=device)
    
    delta_t_src_val = float(t_val - model.memory.last_update[src_idx].item())
    delta_t_src = torch.tensor([delta_t_src_val], dtype=torch.float, device=device)

    # Explicit interaction-history features for this src→dst pair (read-only; counts are
    # advanced in update_memory, exactly mirroring the train-time predict-then-update order).
    hist_feats = model.compute_hist_feats([src_idx], [dst_idx], device)

    out = model(
        n_id, edge_index, hist_t, hist_msg, assoc[b_src], assoc[b_dst], b_msg, delta_t, delta_t_src, hist_feats
    ).squeeze(-1)
    prob_benign = torch.sigmoid(out).item()
    return 1.0 - prob_benign


@torch.no_grad()
def update_memory(model, src_idx: int, dst_idx: int, t_val: int, msg_vec, device) -> None:
    """Commit a single event into the TGN memory and the neighbour store.

    Inserting into the neighbour loader only here (and only for events the caller
    has judged benign) keeps the anti-poisoning gate intact: anomalous events never
    enter an entity's history.
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


def _reset_slot(model, idx: int) -> None:
    """Cold-start a reused memory slot after eviction."""
    with torch.no_grad():
        model.memory.memory[idx].zero_()
        model.memory.last_update[idx] = 0
        model.node_feat[idx].zero_()
        model.node_feat[idx, 14] = 1.0  # Reset Trust Score to max
        model.node_hash[idx].zero_()
    for store in (model.memory.msg_s_store, model.memory.msg_d_store):
        store.pop(idx, None)
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


def score_event(
    model,
    registry: NodeRegistry,
    threshold: float,
    key_src: Hashable,
    key_dst: Hashable,
    timestamp: int,
    features,
    device,
    *,
    src_feat=None,
    dst_feat=None,
    update: bool = True,
) -> tuple[float, bool]:
    """Score one streaming access event.

    Maps the (possibly unseen) entity keys through ``registry``, computes the
    anomaly score, and — when ``update`` is set — writes the event into memory
    **only if it is classified benign** (``score < threshold``). This keeps the
    memory baseline free of attacker-controlled events.

    ``src_feat`` / ``dst_feat`` are the endpoints' static attributes (role /
    clearance / device tier). In a ZTA deployment the orchestrator/OPA already
    holds these for every request, so they are supplied per event — no extra data
    store is required. When omitted, the slot keeps whatever features it already
    has (e.g. those learned for preregistered entities at train time).

    Returns ``(anomaly_score, is_anomaly)``.
    """
    model.eval()
    recency = model.memory.last_update
    src_idx, is_new_src = registry.get_or_add(
        key_src, recency=recency, on_evict=lambda i: _reset_slot(model, i)
    )
    if is_new_src:
        model.node_hash[src_idx] = stable_hash(key_src, model.hash_emb.num_embeddings)

    dst_idx, is_new_dst = registry.get_or_add(
        key_dst, recency=recency, on_evict=lambda i: _reset_slot(model, i)
    )
    if is_new_dst:
        model.node_hash[dst_idx] = stable_hash(key_dst, model.hash_emb.num_embeddings)

    if src_feat is not None:
        _set_node_features(model, src_idx, src_feat, device)
    if dst_feat is not None:
        _set_node_features(model, dst_idx, dst_feat, device)

    score = infer_score(model, src_idx, dst_idx, timestamp, features, device)
    is_anomaly = score >= threshold

    snort_alert = features[1] > 0.5
    if is_anomaly or snort_alert:
        model.node_feat[src_idx, 14] = max(0.0, model.node_feat[src_idx, 14].item() - 0.5)
    else:
        model.node_feat[src_idx, 14] = min(1.0, model.node_feat[src_idx, 14].item() + 0.01)

    if update and not is_anomaly:
        update_memory(model, src_idx, dst_idx, timestamp, features, device)

    return score, is_anomaly


def commit_event(
    model,
    registry: NodeRegistry,
    key_src: Hashable,
    key_dst: Hashable,
    timestamp: int,
    features,
    device,
    *,
    src_feat=None,
    dst_feat=None,
) -> None:
    """Commit an event the caller has already judged benign (e.g. OPA returned ALLOW).

    This is the *unconditional* counterpart of :func:`score_event`: it maps the entity
    keys through ``registry`` (admitting unseen entities, evicting LRU on overflow),
    optionally refreshes their static attributes, and advances both the TGN memory and
    the neighbour history. Use it for the two-step anti-poisoning flow where the
    benign/anomalous decision is made *outside* the model (score with
    :func:`infer_score` / :func:`score_event` ``update=False`` first, then commit here
    only on approval).
    """
    model.eval()
    recency = model.memory.last_update
    src_idx, is_new_src = registry.get_or_add(
        key_src, recency=recency, on_evict=lambda i: _reset_slot(model, i)
    )
    if is_new_src:
        model.node_hash[src_idx] = stable_hash(key_src, model.hash_emb.num_embeddings)

    dst_idx, is_new_dst = registry.get_or_add(
        key_dst, recency=recency, on_evict=lambda i: _reset_slot(model, i)
    )
    if is_new_dst:
        model.node_hash[dst_idx] = stable_hash(key_dst, model.hash_emb.num_embeddings)

    if src_feat is not None:
        _set_node_features(model, src_idx, src_feat, device)
    if dst_feat is not None:
        _set_node_features(model, dst_idx, dst_feat, device)

    update_memory(model, src_idx, dst_idx, timestamp, features, device)


def save_model(model, registry: NodeRegistry, threshold: float, hp: dict,
               checkpoint_path, stats_path) -> None:
    """Persist the deployable artifact (weights + memory + registry + threshold)."""
    checkpoint_path = Path(checkpoint_path)
    stats_path = Path(stats_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # state_dict carries the `memory` / `last_update` / `_assoc` buffers; the raw
    # message store is a plain dict (not a buffer) so it is saved alongside.
    torch.save(
        {
            "model": model.state_dict(),
            "msg_s_store": model.memory.msg_s_store,
            "msg_d_store": model.memory.msg_d_store,
            "last_contact": getattr(model, "last_contact", {}),
            "pair_count": getattr(model, "pair_count", {}),
            "src_count": getattr(model, "src_count", {}),
            "neighbor_loader": model.neighbor_loader.state(),
            "hyperparams": hp,
        },
        checkpoint_path,
    )

    stats = {
        "threshold": float(threshold),
        "target_fpr": hp.get("target_fpr"),
        "capacity": int(hp["capacity"]),
        "registry": registry.to_dict(),
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def load_model(checkpoint_path, stats_path, device):
    """Reconstruct ``(model, registry, threshold, hp)`` for serving.

    ``hp`` (the saved hyper-parameter dict) is returned so a long-running server can
    later re-persist the evolved state via :func:`save_model` without re-reading the
    checkpoint.
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
    # Restore the temporal neighbour buffers (map_location already placed the saved
    # tensors on ``device``); build_model created an empty loader of the right shape.
    if "neighbor_loader" in ckpt:
        model.neighbor_loader.load_state(ckpt["neighbor_loader"])
    model.eval()

    with open(stats_path, encoding="utf-8") as f:
        stats = json.load(f)
    registry = NodeRegistry.from_dict(stats["registry"])
    threshold = float(stats["threshold"])
    return model, registry, threshold, hp
