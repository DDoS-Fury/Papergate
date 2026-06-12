"""Verification harness for the batched offline replay (train_tgn._replay).

Two checks, both on CPU for bit-level determinism:

  1. PARITY: the new ``_replay(batch_size=1)`` must reproduce the *old* per-event replay
     (kept verbatim below as ``_replay_ref``, which calls the unchanged serving primitives
     ``serve_tgn.infer_score`` / ``update_memory`` / ``precursor_boost``). This proves the
     refactor did not change the semantics at batch size 1.

  2. DRIFT: ``_replay(batch_size=B)`` vs ``batch_size=1`` on identical warmed state, to
     quantify the standard batched-TGN within-batch staleness on the synthetic stream.

Run (in the project image):
    docker run --rm -v "$PWD/src:/app/src" -v "$PWD/tests:/app/tests" \
        --entrypoint python graphagate /app/tests/verify_replay_batching.py
"""

from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graphagate.config import TGNConfig
from graphagate.model.registry import NodeRegistry
from graphagate.model.tgn import ZTATemporalGraphNetwork, stable_hash
from graphagate.serve_tgn import (
    infer_score,
    precursor_boost,
    record_alert,
    signal_dirty,
    update_memory,
)
from graphagate.train_tgn import _replay, _synthetic_stream_data


def _replay_ref(model, source_nodes, device_nodes, user, dst, t, msg, y, device, *,
                threshold=None, threshold_dirty=None, gate_by_label=False):
    """Verbatim copy of the ORIGINAL per-event _replay — the parity ground truth."""
    model.eval()
    scores = np.empty(user.shape[0], dtype=np.float64)
    labels = np.empty(user.shape[0], dtype=np.int64)
    u_l, d_l, t_l, y_l = user.tolist(), dst.tolist(), t.tolist(), y.tolist()
    dev_l = device_nodes.tolist() if device_nodes is not None else None
    src_l = source_nodes.tolist() if source_nodes is not None else None

    for i in range(len(u_l)):
        u, d, tv, lab = u_l[i], d_l[i], t_l[i], y_l[i]
        dev = dev_l[i] if dev_l is not None else None
        src_ip = src_l[i] if src_l is not None else None
        msg_vec = msg[i]
        features_bind = [0.0] * len(msg_vec)
        edge_scores = []
        if src_ip is not None and dev is not None:
            edge_scores.append(infer_score(model, src_ip, dev, tv, features_bind, device))
        if dev is not None:
            edge_scores.append(infer_score(model, dev, u, tv, features_bind, device))
        edge_scores.append(infer_score(model, u, d, tv, msg_vec, device, aux_src_idx=dev))
        raw_score = max(edge_scores)
        actor = dev if dev is not None else u
        score = min(1.0, raw_score * precursor_boost(model, actor, tv))
        scores[i] = score
        labels[i] = lab

        eff_thr = threshold
        if threshold_dirty is not None and signal_dirty(msg_vec):
            eff_thr = threshold_dirty

        do_update = (lab == 0) if gate_by_label else (score < eff_thr)
        if do_update:
            if src_ip is not None and dev is not None:
                update_memory(model, src_ip, dev, tv, features_bind, device)
            if dev is not None:
                update_memory(model, dev, u, tv, features_bind, device)
            update_memory(model, u, d, tv, msg_vec, device,
                          aux_pair=(dev, d) if dev is not None else None)

        snort_alert = msg_vec[1] > 0.5
        is_anomaly = (score >= eff_thr) if not gate_by_label else (lab == 1)
        if is_anomaly or snort_alert:
            record_alert(model, actor, tv)
            model.node_feat[actor, 14] = max(0.0, model.node_feat[actor, 14].item() - 0.5)
            if actor != u:
                model.node_feat[u, 14] = max(0.0, model.node_feat[u, 14].item() - 0.5)
        else:
            model.node_feat[actor, 14] = min(1.0, model.node_feat[actor, 14].item() + 0.01)
            if actor != u:
                model.node_feat[u, 14] = min(1.0, model.node_feat[u, 14].item() + 0.01)

    return scores, labels


def _build_model(data, cfg, device):
    torch.manual_seed(cfg.seed)
    total_nodes = data.num_nodes
    capacity = total_nodes + cfg.capacity_headroom
    registry = NodeRegistry(capacity=capacity)
    registry.preregister(data.keys)
    m = ZTATemporalGraphNetwork(
        num_nodes=capacity, node_feat_dim=cfg.node_feat_dim, msg_dim=cfg.msg_dim,
        memory_dim=cfg.memory_dim, time_dim=cfg.time_dim, num_hops=cfg.num_hops,
        hash_buckets=cfg.hash_buckets, hash_dim=cfg.hash_dim, hist_feat_dim=cfg.hist_feat_dim,
    ).to(device)
    with torch.no_grad():
        m.node_feat[:total_nodes] = data.node_features.to(device)
        hashes = [stable_hash(registry._idx_to_key[i], cfg.hash_buckets) for i in range(total_nodes)]
        m.node_hash[:total_nodes] = torch.tensor(hashes, dtype=torch.long, device=device)
    m.init_neighbor_loader(cfg.neighbor_size, device)
    m.use_struct_head = m.use_hash_identity = m.use_hist_feats = m.use_precursor = True
    m.precursor_half_life = cfg.precursor_half_life
    m.precursor_max_boost = cfg.precursor_max_boost
    m.eval()
    return m


def _warm(model, data, lo, hi, device):
    """Deterministically populate memory/history via the unchanged serving primitive."""
    u, d, t = data.user.tolist(), data.dst.tolist(), data.t.tolist()
    y = data.y.tolist()
    dev = data.device_nodes.tolist() if data.device_nodes is not None else None
    src = data.source_nodes.tolist() if data.source_nodes is not None else None
    for i in range(lo, hi):
        if y[i] != 0:
            continue
        msg_vec = data.msg[i]
        fb = [0.0] * len(msg_vec)
        dv = dev[i] if dev is not None else None
        s_ = src[i] if src is not None else None
        if s_ is not None and dv is not None:
            update_memory(model, s_, dv, t[i], fb, device)
        if dv is not None:
            update_memory(model, dv, u[i], t[i], fb, device)
        update_memory(model, u[i], d[i], t[i], msg_vec, device,
                      aux_pair=(dv, d[i]) if dv is not None else None)


def _slice_data(data, lo, hi):
    return dict(
        source_nodes=data.source_nodes[lo:hi] if data.source_nodes is not None else None,
        device_nodes=data.device_nodes[lo:hi] if data.device_nodes is not None else None,
        user=data.user[lo:hi], dst=data.dst[lo:hi], t=data.t[lo:hi],
        msg=data.msg[lo:hi], y=data.y[lo:hi],
    )


def main() -> int:
    device = torch.device("cpu")
    cfg = dataclasses.replace(TGNConfig(), num_events=4000, capacity_headroom=2000)
    data = _synthetic_stream_data(cfg)
    n = len(data.dst)
    warm_hi = n // 2
    test_lo, test_hi = warm_hi, min(warm_hi + 800, n)
    sl = _slice_data(data, test_lo, test_hi)
    thr, thr_dirty = 0.5, 0.7

    print(f"events={n} warm=[0,{warm_hi}) test=[{test_lo},{test_hi}) "
          f"has_dev={data.device_nodes is not None} has_src={data.source_nodes is not None}")

    def run(replay_fn, bs=None, **kw):
        m = _build_model(data, cfg, device)
        _warm(m, data, 0, warm_hi, device)
        extra = {} if bs is None else {"batch_size": bs, "desc": f"bs={bs}"}
        return replay_fn(m, sl["source_nodes"], sl["device_nodes"], sl["user"], sl["dst"],
                         sl["t"], sl["msg"], sl["y"], device, **kw, **extra)[0]

    ok = True
    for gate, kw in (("calib(gate_by_label)", dict(gate_by_label=True)),
                     ("eval(routed)", dict(threshold=thr, threshold_dirty=thr_dirty,
                                           gate_by_label=False))):
        s_ref = run(_replay_ref, **kw)
        s_b1 = run(_replay, bs=1, **kw)
        d1 = float(np.max(np.abs(s_ref - s_b1)))
        parity = d1 <= 1e-5
        ok = ok and parity
        print(f"\n[{gate}] PARITY ref vs batched(bs=1): max|Δ|={d1:.3e}  -> "
              f"{'PASS' if parity else 'FAIL'}")
        for bs in (64, 256):
            s_bn = run(_replay, bs=bs, **kw)
            dn = float(np.max(np.abs(s_b1 - s_bn)))
            mn = float(np.mean(np.abs(s_b1 - s_bn)))
            print(f"    DRIFT bs={bs:4d} vs bs=1: max|Δ|={dn:.3e} mean|Δ|={mn:.3e}")

    print(f"\nOVERALL PARITY: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
