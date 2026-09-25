"""Exact-fusion diff: dump replay / serving (/ train-mode) outputs of ONE source tree.

Run the same command against two trees (HEAD in a `git worktree` and the working tree) and
compare the dumps with `fusion_cmp.py`. `graphagate` must resolve to <tree>/src: for the
worktree, put a directory containing a `graphagate -> <worktree>/src` symlink first on
PYTHONPATH (the editable install otherwise wins).

Seeded-init mode (what ran on the Mac, CPU, 2026-09-25): untrained model, 4000-event stream
at the TGNConfig default seed (42), 400 scored events. Also dumps train-mode logits and all
parameter gradients.

    python tasks/tmp/fusion_e2e.py <tree> <out.npz> [--ref-state head_state.pt]

Checkpoint mode (step 0 on the workstation, before trusting any branch run): the trained
checkpoint + stats, events [val_end, val_end + n) of a dev-seed stream, on --device.

    python tasks/tmp/fusion_e2e.py <tree> <out.npz> --ckpt artifacts/tgn_checkpoint.pt \
        --stats artifacts/tgn_stats.json --seed 2000 --device cuda --n 5000

Pass criterion (fusion_cmp.py): |Δ| <= 1e-7 + 1e-5·max|ref| per array.
"""
import argparse
import dataclasses
import os
import sys

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("tree")
p.add_argument("out")
p.add_argument("--ref-state", help="HEAD state_dict to load strictly (seeded-init mode)")
p.add_argument("--ckpt")
p.add_argument("--stats")
p.add_argument("--seed", type=int, default=2000)
p.add_argument("--n", type=int, default=5000)
p.add_argument("--num-events", type=int, help="shorter stream (smoke test only)")
p.add_argument("--device", default="cpu")
p.add_argument("--threads", type=int, default=2)
args = p.parse_args()
torch.set_num_threads(args.threads)
sys.path.insert(0, os.path.join(args.tree, "tests"))

import graphagate  # noqa: E402

assert os.path.samefile(os.path.dirname(graphagate.__file__), os.path.join(args.tree, "src")), \
    graphagate.__file__
from graphagate.config import TGNConfig  # noqa: E402
from graphagate.model.registry import NodeRegistry  # noqa: E402
from graphagate.serve_tgn import load_model, score_event  # noqa: E402
from graphagate.train_tgn import _replay, _synthetic_stream_data  # noqa: E402

device = torch.device(args.device)
res = {}


def replay_all(fresh, sl, bss, thr, thr_dirty):
    for bs in bss:
        for gate in ("label", "routed"):
            kw = {"gate_by_label": True} if gate == "label" else {
                "threshold": thr, "threshold_dirty": thr_dirty, "gate_by_label": False}
            m = fresh()
            res[f"replay_{gate}_bs{bs}"] = _replay(
                m, sl["source_nodes"], sl["device_nodes"], sl["user"], sl["dst"], sl["t"],
                sl["msg"], sl["y"], device, config_nodes=sl["config_nodes"], batch_size=bs,
                desc=f"{gate} bs={bs}", **kw)[0]


def serve_all(m, reg, keys, data, lo, hi, thr, thr_dirty, guest_fallback=False):
    u, d, t = data.user.tolist(), data.dst.tolist(), data.t.tolist()
    dev, src, cfn = data.device_nodes.tolist(), data.source_nodes.tolist(), data.config_nodes.tolist()
    out = []
    for i in range(lo, hi):
        s, _flag, _thr = score_event(
            m, reg, thr, keys[u[i]], keys[dev[i]], keys[d[i]], int(t[i]), data.msg[i].tolist(),
            device, key_source=keys[src[i]], key_config=keys[cfn[i]], threshold_dirty=thr_dirty,
            update=True, guest_device_fallback=guest_fallback)
        out.append(s)
    res["serve"] = np.array(out)


if args.ckpt:
    # ---- checkpoint mode: trained weights, real memory state, dev-seed stream ----------
    cfg = dataclasses.replace(TGNConfig(), seed=args.seed)
    if args.num_events:
        cfg = dataclasses.replace(cfg, num_events=args.num_events)
    assert args.seed not in range(1000, 1010), "pre-registered seeds are off limits"
    data = _synthetic_stream_data(cfg)
    n = len(data.dst)
    lo = int(n * cfg.train_frac) + int(n * cfg.val_frac)
    hi = min(lo + args.n, n)
    _m, reg, thr, thr_dirty, _hp = load_model(args.ckpt, args.stats, device)
    idx = [reg.get(k) for k in data.keys]
    assert None not in idx, "checkpoint registry does not cover this stream (other seed?)"
    idx = torch.tensor(idx, dtype=torch.long)

    def col(x):
        return idx[x[lo:hi]]

    sl = {"source_nodes": col(data.source_nodes), "device_nodes": col(data.device_nodes),
          "config_nodes": col(data.config_nodes), "user": col(data.user), "dst": col(data.dst),
          "t": data.t[lo:hi], "msg": data.msg[lo:hi], "y": data.y[lo:hi]}
    replay_all(lambda: load_model(args.ckpt, args.stats, device)[0], sl, (1, 256), thr, thr_dirty)
    m, reg, _t, _td, hp = load_model(args.ckpt, args.stats, device)
    serve_all(m, reg, list(data.keys), data, lo, hi, thr, thr_dirty,
              guest_fallback=bool(hp.get("guest_device_fallback", False)))
    res["sd_keys"] = np.array(sorted(f"{k}:{tuple(v.shape)}" for k, v in m.state_dict().items()))
else:
    # ---- seeded-init mode --------------------------------------------------------------
    from verify_replay_batching import _build_model, _slice_data, _warm

    cfg = dataclasses.replace(TGNConfig(), num_events=4000, capacity_headroom=2000)
    data = _synthetic_stream_data(cfg)
    n = len(data.dst)
    warm = n // 2
    lo, hi = warm, min(warm + 400, n)

    def fresh():
        m = _build_model(data, cfg, device)
        _warm(m, data, 0, warm, device)
        return m

    m = _build_model(data, cfg, device)
    sd = m.state_dict()
    res["sd_keys"] = np.array(sorted(f"{k}:{tuple(v.shape)}" for k, v in sd.items()))
    if args.ref_state is None:
        torch.save(sd, args.out.replace(".npz", "_state.pt"))
    else:
        ref = torch.load(args.ref_state)
        m.load_state_dict(ref, strict=True)
        res["sd_maxdiff"] = np.array(max(float((sd[k].float() - ref[k].float()).abs().max()) for k in ref))

    replay_all(fresh, _slice_data(data, lo, hi), (1, 64), 0.5, 0.7)
    reg = NodeRegistry(capacity=data.num_nodes + cfg.capacity_headroom)
    reg.preregister(data.keys)
    serve_all(fresh(), reg, reg._idx_to_key, data, lo, hi, 0.5, 0.7)

    # Train mode: score + gradients (E2 exact in training; E3 keeps per-row dropout). The
    # embedding is taken in eval: E5 changes the attention-dropout draw count in training.
    sl = _slice_data(data, lo, hi)
    m = fresh()
    nodes = torch.cat([sl["user"][:64], sl["dst"][:64], sl["device_nodes"][:64]]).unique()
    n_id, ei, ht, hm = m.neighbor_loader(nodes)
    z = m.embed(n_id, ei, ht, hm)
    m.train()
    assoc = m.neighbor_loader._assoc
    g = torch.Generator().manual_seed(7)
    P = 64
    src_l = assoc[sl["user"][:P]].repeat(3)
    dst_l = assoc[torch.cat([sl["dst"][:P], sl["device_nodes"][:P], sl["dst"][:P].flip(0)])]
    msg = torch.randn(3 * P, cfg.msg_dim, generator=g)
    dt = torch.rand(3 * P, generator=g) * 1e5
    dts = torch.rand(3 * P, generator=g) * 1e5
    hist = torch.rand(3 * P, cfg.hist_feat_dim, generator=g)
    torch.manual_seed(123)  # struct_proj dropout draws
    logit = m.score(z, m.node_feat[n_id], m.node_hash[n_id], src_l, dst_l, msg, dt, dts, hist)
    logit.square().mean().backward()
    res["train_logit"] = logit.detach().numpy()
    for name, prm in m.named_parameters():
        if prm.grad is not None:
            res[f"grad:{name}"] = prm.grad.numpy().copy()

np.savez(args.out, **res)
print("saved", args.out, {k: v.shape for k, v in res.items() if not k.startswith("grad:")})
