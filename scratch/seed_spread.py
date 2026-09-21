"""Spread of the audit statistics over seeds outside the test's SEEDS (threshold check)."""
import sys
import numpy as np
sys.path.insert(0, "tests")
import test_leakage_audit as t
from graphagate.config import TGNConfig
from graphagate.data.lookup_rules import lookup_flags

RULES = ("cfg_new", "src_new", "dev_new", "cfg|dev_new", "cfg|usr_new", "dev|usr_new", "src|usr_new", "role_changed")
worst_lookup, worst_static = [], []
for seed in (1, 2, 3, 4, 5, 6, 8, 9, 42, 7, 123):
    s = t._stream(seed); ty = s.types.numpy(); ts = int(len(ty) * 0.8)
    f = lookup_flags(s, "proto-self", ts); te = np.arange(len(ty)) >= ts
    lk = max((t._auc((ty[te & ((ty == 0) | (ty == k))] == k).astype(int), f[r][te & ((ty == 0) | (ty == k))].astype(float)), k, r)
             for k in (3, 4) for r in RULES)
    cols = t._columns(s)
    st = max((t._auc((ty[(ty == 0) | (ty == k)] == k).astype(int), v[(ty == 0) | (ty == k)]), k, c)
             for k in (3, 4) for c, v in cols.items())
    worst_lookup.append(lk[0]); worst_static.append(st[0])
    print(f"seed={seed:3d} worst lookup {lk[0]:.3f} ({lk[1]},{lk[2]})  worst static {st[0]:.3f} ({st[1]},{st[2]})", flush=True)
print(f"lookup: mean {np.mean(worst_lookup):.3f} max {np.max(worst_lookup):.3f} | static: mean {np.mean(worst_static):.3f} max {np.max(worst_static):.3f}")
