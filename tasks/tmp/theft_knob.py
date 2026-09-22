"""Theft src|usr_new firing and AUC (audit protocol) vs p_theft_known_source."""
import sys, numpy as np
from sklearn.metrics import roc_auc_score
from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg
from graphagate.data.lookup_rules import lookup_flags
for p in map(float, sys.argv[1:]):
    auc, rt, rb = [], [], []
    for seed in (42, 44, 48, 7, 123, 2000, 2001):
        cfg = TGNConfig(seed=seed, p_theft_known_source=p)
        s = generate_streaming_data(**stream_kwargs_from_cfg(cfg))
        ty = s.types.numpy(); ts = int(len(ty) * (cfg.train_frac + cfg.val_frac))
        f = lookup_flags(s, "proto-self", ts)["src|usr_new"]
        te = np.arange(len(ty)) >= ts
        sel = te & ((ty == 0) | (ty == 4))
        auc.append(roc_auc_score(ty[sel] == 4, f[sel])); rt.append(f[te & (ty == 4)].mean()); rb.append(f[te & (ty == 0)].mean())
    print(f"p={p}: auc " + " ".join(f"{a:.3f}" for a in auc) + f" | max {max(auc):.3f} | theft rate {np.mean(rt):.2f} benign {np.mean(rb):.3f}", flush=True)
