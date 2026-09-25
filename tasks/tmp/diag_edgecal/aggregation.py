"""Struct-head collapse check + label-free per-edge calibration of the max aggregation.
Calibration uses ONLY clean benign events of the validation slice (no attack label)."""
import sys
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
D = sys.argv[1]
va, te = (dict(np.load(f"{D}/diag_{p}.npz")) for p in ("val", "test"))
dirty = lambda m: (m[:, 0] == 0) | (m[:, 1:4] > 0.5).any(1)
print("struct part per edge (test): std / min / max")
for g in range(5):
    s = te["per"][:, g, 2]
    print(f"  edge {g}: std={s.std():.4f} min={s.min():.3f} max={s.max():.3f}")
vb = (va["types"] == 0) & ~dirty(va["msg"])
cb = (te["types"] == 0) & ~dirty(te["msg"])

def surprisal(ref, x):  # -log empirical upper-tail p-value against benign val
    ref = np.sort(ref)
    p = (len(ref) - np.searchsorted(ref, x, side="left") + 1) / (len(ref) + 1)
    return -np.log(p)

S = np.stack([surprisal(va["per"][vb, g, 0], te["per"][:, g, 0]) for g in range(5)], 1)
cands = {
    "attuale: max logit (+precursor)": te["score"],
    "max logit, senza precursor": te["raw"],
    "max surprisal per-edge": S.max(1),
    "somma surprisal (Fisher)": S.sum(1),
    "somma surprisal + precursor": S.sum(1) + te["shift"],
    "solo access edge": te["per"][:, 0, 0],
}
for c, name in ((3, "lateral"), (4, "theft")):
    m = te["types"] == c
    sel = cb | m
    y = m[sel].astype(int)
    print(f"\n{name} (n={m.sum()}) vs benigno pulito test (n={cb.sum()}), caso AP={y.mean():.4f}")
    for k, s in cands.items():
        s = s[sel]
        thr = np.quantile(s[y == 0], 0.99)
        thr5 = np.quantile(s[y == 0], 0.95)
        print(f"  {k:32s} AUC={roc_auc_score(y, s):.3f} AP={average_precision_score(y, s):.4f} "
              f"rec@FPR1%={(s[y==1] > thr).mean():.3f} rec@FPR5%={(s[y==1] > thr5).mean():.3f}")
