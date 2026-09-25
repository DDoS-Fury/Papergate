"""Instrumented train_tgn run: per-event, per-edge score decomposition on val pass B + test.

No change to src/: chain_logits / _replay / _synthetic_stream_data are wrapped in the
train_tgn module namespace. The main logit computation is the same calls in the same
order as serve_tgn.chain_logits (extra score calls are side-effect free in eval), so the
reported scores are the pipeline's own. Writes /diag/out/diag_<phase>.npz.
"""
import sys, time
import numpy as np
import torch

from graphagate import train_tgn as T
from graphagate.config import TGNConfig

OUT = "/diag/out"  # overridden by --out
GROUPS = ["access", "dev>user", "cfg>user", "cfg>dev", "src>cfg"]
K = 5
BENIGN_SAMPLE = 0.03
rng = np.random.default_rng(0)

S = {"data": None, "phase": None, "offset": 0, "i": 0, "rec": None, "train_end": 0, "val_end": 0}
_orig_chain = T.chain_logits
_orig_replay = T._replay
_orig_synth = T._synthetic_stream_data


def _synth(cfg):
    d = _orig_synth(cfg)
    S["data"] = d
    n = len(d.dst)
    S["train_end"] = int(n * cfg.train_frac)
    S["val_end"] = S["train_end"] + int(n * cfg.val_frac)
    return d


def _score_groups(model, z, nf, h_idx, assoc, groups, t, device, *, zero_msg=False):
    """Per-group anomaly logits (same calls as serve_tgn.chain_logits) + inputs used."""
    t_list = t.tolist()
    rows = []
    for src, dst, msg, aux in groups:
        s_list, d_list = src.tolist(), dst.tolist()
        d_pair = model.pair_delta_t(s_list, d_list, t_list, device)
        d_src = model.src_delta_t(src, t, device)
        hist = model.compute_hist_feats(s_list, d_list, device,
                                        aux_src_ids=None if aux is None else aux.tolist())
        m = torch.zeros_like(msg) if zero_msg else msg
        tot = model.score(z, nf, h_idx, assoc[src], assoc[dst], m, d_pair, d_src, hist)
        rows.append((tot, d_pair, d_src, hist, src, dst, m, aux))
    return rows


def _chain(model, groups, t, device):
    if S["phase"] is None:
        return _orig_chain(model, groups, t, device)
    assert t.numel() == 1, "diag requires eval_batch_size=1"
    gi = S["offset"] + S["i"]
    S["i"] += 1
    R = S["rec"]
    srcs = torch.cat([g[0] for g in groups])
    dsts = torch.cat([g[1] for g in groups])
    n_id, edge_index, hist_t, hist_msg = model.neighbor_loader(torch.cat([srcs, dsts]).unique())
    z = model.embed(n_id, edge_index, hist_t, hist_msg)
    assoc = model.neighbor_loader._assoc
    nf, h_idx = model.node_feat[n_id], model.node_hash[n_id]
    rows = _score_groups(model, z, nf, h_idx, assoc, groups, t, device)
    out = None
    per = np.zeros((len(GROUPS), 11), dtype=np.float32)
    for g, (tot, d_pair, d_src, hist, src, dst, m, aux) in enumerate(rows):
        logit = -tot
        out = logit if out is None else torch.maximum(out, logit)
        model.use_struct_head = False
        feat = model.score(z, nf, h_idx, assoc[src], assoc[dst], m, d_pair, d_src, hist)
        model.use_struct_head = True
        per[g, 0] = float(logit)            # anomaly logit of this edge
        per[g, 1] = float(-feat)            # feature-head part (anomaly sign)
        per[g, 2] = float(-(tot - feat))    # structural-head part (anomaly sign)
        per[g, 3:9] = hist[0].cpu().numpy()
        per[g, 9] = float(torch.log1p(d_pair.float().view(-1)[0].clamp(min=0)))
        per[g, 10] = float(torch.log1p(d_src.float().view(-1)[0].clamp(min=0)))
    R["per"].append(per)
    R["raw"].append(float(out))
    R["idx"].append(gi)
    ent = [groups[0][0], groups[0][1], groups[1][0], groups[2][0]]  # user, dst, device, config
    R["z"].append(torch.cat([z[assoc[e]].view(-1) for e in ent]).half().cpu().numpy())

    etype = int(S["data"].types[gi])
    if etype in (3, 4) or (etype == 0 and rng.random() < BENIGN_SAMPLE):
        R["x_idx"].append(gi)
        R["x_abl"].append(_ablations(model, z, nf, h_idx, assoc, groups, t, device))
        R["x_neg"].append(_negatives(model, groups, t, device))
    return out


def _max_logit(rows):
    return float(torch.stack([-r[0] for r in rows]).max())


def _ablations(model, z, nf, h_idx, assoc, groups, t, device):
    """Event anomaly logit with one input block removed at a time."""
    res = []
    for flag in ("use_hist_feats", "use_hash_identity", "use_struct_head"):
        setattr(model, flag, False)
        res.append(_max_logit(_score_groups(model, z, nf, h_idx, assoc, groups, t, device)))
        setattr(model, flag, True)
    res.append(_max_logit(_score_groups(model, z, nf, h_idx, assoc, groups, t, device, zero_msg=True)))
    res.append(_max_logit(_score_groups(model, torch.zeros_like(z), nf, h_idx, assoc, groups, t, device)))
    return res  # [no_hist, no_hash, no_struct, zero_msg, zero_z]


def _negatives(model, groups, t, device):
    """Anomaly logits of K training-style structural negatives per edge group."""
    d = S["data"]
    ranges = [(d.neg_lo, d.neg_num), (d.usr_lo, d.usr_num), (d.usr_lo, d.usr_num),
              (d.dev_lo, d.dev_num), (d.cfg_lo, d.cfg_num)]
    neg_groups = []
    for (src, dst, msg, aux), (lo, num) in zip(groups, ranges):
        nd = T._sample_structural_negatives(K, num, lo, device, avoid=dst.repeat(K))
        neg_groups.append((src.repeat(K), nd, msg.repeat(K, 1),
                           None if aux is None else aux.repeat(K)))
    nodes = torch.cat([torch.cat([g[0], g[1]]) for g in neg_groups]).unique()
    n_id, edge_index, hist_t, hist_msg = model.neighbor_loader(nodes)
    z = model.embed(n_id, edge_index, hist_t, hist_msg)
    assoc = model.neighbor_loader._assoc
    nf, h_idx = model.node_feat[n_id], model.node_hash[n_id]
    rows = _score_groups(model, z, nf, h_idx, assoc, neg_groups, t.repeat(K), device)
    return np.stack([(-r[0]).cpu().numpy() for r in rows])  # [groups, K]


def _replay(*a, desc="replay", **kw):
    phase = {"Calibration pass B (val replay, test gate)": "val",
             "Inferenza (replay test)": "test"}.get(desc)
    if phase is None:
        return _orig_replay(*a, desc=desc, **kw)
    S.update(phase=phase, i=0, offset=S["train_end"] if phase == "val" else S["val_end"],
             rec={k: [] for k in ("per", "raw", "idx", "z", "x_idx", "x_abl", "x_neg")})
    t0 = time.time()
    scores, labels = _orig_replay(*a, desc=desc, **kw)
    R, d = S["rec"], S["data"]
    idx = np.asarray(R["idx"])
    raw = np.asarray(R["raw"])
    shift = np.log(scores) - np.log1p(-scores) - raw  # precursor prior actually applied
    np.savez_compressed(
        f"{OUT}/diag_{phase}.npz",
        idx=idx, raw=raw, score=scores, shift=shift, per=np.stack(R["per"]), z=np.stack(R["z"]),
        x_idx=np.asarray(R["x_idx"]), x_abl=np.asarray(R["x_abl"]), x_neg=np.stack(R["x_neg"]),
        types=d.types.numpy()[idx], y=d.y.numpy()[idx], scenario=d.scenario.numpy()[idx],
        msg=d.msg.numpy()[idx], t=d.t.numpy()[idx], user=d.user.numpy()[idx],
        device=d.device_nodes.numpy()[idx], dst=d.dst.numpy()[idx],
        config=d.config_nodes.numpy()[idx], source=d.source_nodes.numpy()[idx],
    )
    print(f"[diag] {phase}: saved {len(idx)} events, {len(R['x_idx'])} extra-scored "
          f"({time.time() - t0:.0f}s)", flush=True)
    S["phase"] = None
    return scores, labels


T.chain_logits = _chain
T._replay = _replay
T._synthetic_stream_data = _synth

if __name__ == "__main__":
    import argparse, dataclasses
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    OUT = a.out
    cfg = TGNConfig()
    over = {k: v for k, v in (("num_events", a.events), ("epochs", a.epochs)) if v is not None}
    cfg = dataclasses.replace(cfg, **over)
    assert cfg.eval_batch_size == 1
    m = T.train_tgn(cfg, save=False)
    np.save(f"{OUT}/metrics.npy", m, allow_pickle=True)
    print("[diag] done", flush=True)
