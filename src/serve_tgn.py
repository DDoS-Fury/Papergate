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
from graphagate.model.tgn import ZTATemporalGraphNetwork


def build_model(hp: dict, device: torch.device) -> ZTATemporalGraphNetwork:
    """Instantiate the model from a hyper-parameter dict (see :func:`save_model`)."""
    model = ZTATemporalGraphNetwork(
        num_nodes=int(hp["capacity"]),
        node_feat_dim=int(hp["node_feat_dim"]),
        msg_dim=int(hp["msg_dim"]),
        memory_dim=int(hp["memory_dim"]),
        time_dim=int(hp["time_dim"]),
    ).to(device)
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

    Does **not** mutate memory. ``torch.unique`` naturally collapses the
    ``src == dst`` self-loop into a single node, yielding a valid edge index.
    """
    b_src, b_dst, b_t, b_msg = _event_tensors(src_idx, dst_idx, t_val, msg_vec, device)
    n_id, inv = torch.unique(torch.cat([b_src, b_dst]), return_inverse=True)
    edge_index = torch.stack([inv[:1], inv[1:]], dim=0)
    out = model(n_id, edge_index, b_t, b_msg).squeeze(-1)
    prob_benign = torch.sigmoid(out).item()
    return 1.0 - prob_benign


@torch.no_grad()
def update_memory(model, src_idx: int, dst_idx: int, t_val: int, msg_vec, device) -> None:
    """Commit a single event into the TGN memory and detach the graph."""
    b_src, b_dst, b_t, b_msg = _event_tensors(src_idx, dst_idx, t_val, msg_vec, device)
    model.memory.update_state(b_src, b_dst, b_t, b_msg)
    model.memory.detach()


def _reset_slot(model, idx: int) -> None:
    """Cold-start a reused memory slot after eviction."""
    with torch.no_grad():
        model.memory.memory[idx].zero_()
        model.memory.last_update[idx] = 0
        model.node_feat[idx].zero_()
    for store in (model.memory.msg_s_store, model.memory.msg_d_store):
        store.pop(idx, None)


def _set_node_features(model, idx: int, feat, device) -> None:
    """Write a node's static attributes (role / clearance / tier) into its slot."""
    with torch.no_grad():
        model.node_feat[idx] = torch.as_tensor(
            feat, dtype=model.node_feat.dtype, device=device
        )


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
    src_idx, _ = registry.get_or_add(
        key_src, recency=recency, on_evict=lambda i: _reset_slot(model, i)
    )
    dst_idx, _ = registry.get_or_add(
        key_dst, recency=recency, on_evict=lambda i: _reset_slot(model, i)
    )

    if src_feat is not None:
        _set_node_features(model, src_idx, src_feat, device)
    if dst_feat is not None:
        _set_node_features(model, dst_idx, dst_feat, device)

    score = infer_score(model, src_idx, dst_idx, timestamp, features, device)
    is_anomaly = score >= threshold

    if update and not is_anomaly:
        update_memory(model, src_idx, dst_idx, timestamp, features, device)

    return score, is_anomaly


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
    """Reconstruct ``(model, registry, threshold)`` for serving."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hp = ckpt["hyperparams"]
    model = build_model(hp, device)
    model.load_state_dict(ckpt["model"])
    # Restore pending raw messages so memory continuation is exact.
    model.memory.msg_s_store = ckpt.get("msg_s_store", {})
    model.memory.msg_d_store = ckpt.get("msg_d_store", {})
    model.eval()

    with open(stats_path, encoding="utf-8") as f:
        stats = json.load(f)
    registry = NodeRegistry.from_dict(stats["registry"])
    threshold = float(stats["threshold"])
    return model, registry, threshold
