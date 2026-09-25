"""Fusion probe on set-membership flags (no model): MAX (= OR) vs SUM of per-pair novelty.

Pairs mirror the TGN edge set (src|cfg, cfg|dev, cfg|usr, dev|usr, usr|res) plus the
src|usr relation the TGN has no edge / counter for. Commit gate mirrors the TGN replay:
ground-truth benign through train_end, then `not signal_dirty`. Dev seeds only.
"""
import sys, time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg

PAIRS = ("src|cfg", "cfg|dev", "cfg|usr", "dev|usr", "usr|res", "src|usr")
TGN = PAIRS[:5]


def flags(s, train_end, gate="dirty", q=3):
    y, msg = s.y.numpy(), s.msg.numpy()
    cf, sr, dv, us, ds = (x.numpy() for x in (s.config, s.source, s.device, s.user, s.dst))
    n = len(y)
    F = np.zeros((n, len(PAIRS)), dtype=bool)
    seen = [set() for _ in PAIRS]
    cand = [dict() for _ in PAIRS]  # quarantine: sightings of a not-yet-committed pair
    for i in range(n):
        keys = ((sr[i], cf[i]), (cf[i], dv[i]), (cf[i], us[i]), (dv[i], us[i]), (us[i], ds[i]), (sr[i], us[i]))
        for j, (k, st) in enumerate(zip(keys, seen)):
            F[i, j] = k not in st
        dirty = bool(msg[i, 0] == 0 or msg[i, 1:4].any())
        if i < train_end:
            if y[i] == 0:
                for k, st in zip(keys, seen):
                    st.add(k)
            continue
        if dirty:
            continue
        if gate == "dirty":          # TGN replay gate: commit every signal-clean event
            for k, st in zip(keys, seen):
                st.add(k)
        elif gate == "self":         # rules proto-self: commit only if no pair is new
            if not F[i].any():
                for k, st in zip(keys, seen):
                    st.add(k)
        elif gate == "quarantine":   # a new pair is committed after q clean sightings
            for j, (k, st) in enumerate(zip(keys, seen)):
                if k in st:
                    continue
                c = cand[j].get(k, 0) + 1
                if c >= q:
                    st.add(k); cand[j].pop(k, None)
                else:
                    cand[j][k] = c
    return F


def main(seeds):
    cfg = TGNConfig()
    rows = []
    for seed in seeds:
        t0 = time.time()
        s = generate_streaming_data(**{**stream_kwargs_from_cfg(cfg), "seed": seed})
        n = len(s.y)
        train_end = int(n * cfg.train_frac)
        val_end = train_end + int(n * cfg.val_frac)
        types = s.types.numpy()
        test = np.arange(n) >= val_end
        tgn_idx = [PAIRS.index(p) for p in TGN]
        scores = {}
        for gate, q in (("dirty", 0), ("quarantine", 3), ("quarantine", 10), ("self", 0)):
            F = flags(s, train_end, gate, q).astype(float)
            g = gate if gate != "quarantine" else f"quar{q}"
            scores[f"{g:6s} MAX(tgn)"] = F[:, tgn_idx].max(1)
            scores[f"{g:6s} SUM(tgn)"] = F[:, tgn_idx].sum(1)
            scores[f"{g:6s} SUM(tgn+src|usr)"] = F.sum(1)
            ben = test & (types == 0)
            print(f"  seed {seed} gate {g}: benign test novelty per pair", {p: round(float(F[ben, j].mean()), 3) for j, p in enumerate(PAIRS)}, flush=True)
        for tid, name in ((3, "lateral"), (4, "theft")):
            sel = test & ((types == 0) | (types == tid))
            lab = (types[sel] == tid).astype(int)
            for k, v in scores.items():
                rows.append((seed, name, k, roc_auc_score(lab, v[sel]), average_precision_score(lab, v[sel]), int(lab.sum())))
        print(f"seed {seed}: {time.time()-t0:.1f}s", flush=True)
    import collections
    agg = collections.defaultdict(list)
    for seed, name, k, auc, ap, npos in rows:
        agg[(name, k)].append((auc, ap, npos))
    print(f"\n{'class':8s} {'score':26s} {'AUC mean±sd':>14s} {'AP mean':>8s}  n/seed")
    for (name, k), v in agg.items():
        a = np.array(v)
        print(f"{name:8s} {k:26s} {a[:,0].mean():.3f}±{a[:,0].std():.3f}  {a[:,1].mean():8.4f}  {a[:,2].astype(int).tolist()}")


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]] or [2000])
