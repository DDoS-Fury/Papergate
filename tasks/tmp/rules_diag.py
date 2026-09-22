"""Seed 2000, old (50 users) vs new (100 users) config: how often each lookup rule fires
on test benign vs lateral vs theft, and how many benign events carry a sensor/deny signal."""
import dataclasses, sys, numpy as np
from sklearn.metrics import roc_auc_score
from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg
from graphagate.data.lookup_rules import lookup_flags, FLAGS

OLD = dict(num_users=50, num_new_users=12, num_devices=80, num_sources=150)
for name, ov in (("old u50", OLD), ("new u100", {})):
    cfg = dataclasses.replace(TGNConfig(), seed=2000, **ov)
    s = generate_streaming_data(**stream_kwargs_from_cfg(cfg))
    N = len(s.types); va = int(N * (cfg.train_frac + cfg.val_frac))
    ty = s.types.numpy()[va:]
    f = lookup_flags(s, gate="proto-self", test_start=va)
    groups = {"benign": ty == 0, "lateral": ty == 3, "theft": ty == 4}
    print(f"\n== {name}: test n={len(ty)} benign={groups['benign'].sum()} lat={groups['lateral'].sum()} theft={groups['theft'].sum()}")
    print(f"{'flag':14s}" + "".join(f"{g:>9s}" for g in groups))
    for k in FLAGS:
        x = f[k][va:]
        print(f"{k:14s}" + "".join(f"{x[m].mean():9.3f}" for m in groups.values()))
    st = f["stateful"][va:]
    for g in ("lateral", "theft"):
        m = groups["benign"] | groups[g]
        print(f"stateful AUC {g}: {roc_auc_score(groups[g][m], st[m]):.3f}")
    print("stateful score distribution (benign / lateral / theft):")
    for v in range(4):
        print(f"  ={v}" if v < 3 else "  >=3", "".join(f"{((st[m] == v) if v < 3 else (st[m] >= 3)).mean():9.3f}" for m in groups.values()))
