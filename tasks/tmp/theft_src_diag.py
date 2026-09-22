"""Theft src|usr_new AUC (leakage-audit protocol) across seeds, with/without fix H."""
import dataclasses, numpy as np
from sklearn.metrics import roc_auc_score
from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg
from graphagate.data.lookup_rules import lookup_flags
import sys
N = int(sys.argv[1])
for p in (0.0, 0.7):
    out = []
    for seed in list(range(40, 50)) + [2000, 2001, 2002, 2003]:
        cfg = TGNConfig(num_events=N, seed=seed, p_harvest_cached=p)
        s = generate_streaming_data(**stream_kwargs_from_cfg(cfg))
        ty = s.types.numpy(); ts = int(len(ty) * (cfg.train_frac + cfg.val_frac))
        f = lookup_flags(s, "proto-self", ts)
        te = np.arange(len(ty)) >= ts
        sel = te & ((ty == 0) | (ty == 4))
        out.append(roc_auc_score((ty[sel] == 4), f["src|usr_new"][sel]))
    print(f"p={p}: " + " ".join(f"{a:.3f}" for a in out), f"| mean {np.mean(out):.3f} max {np.max(out):.3f}", flush=True)
