"""Per-type effect of per-edge benign calibration variants (fit on val benign only)."""
import sys
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
D = sys.argv[1]
va, te = (dict(np.load(f"{D}/diag_{p}.npz")) for p in ("val", "test"))
dirty = lambda m: (m[:, 0] == 0) | (m[:, 1:4] > 0.5).any(1)
refset = sys.argv[2] if len(sys.argv) > 2 else "all"
vb = va["types"] == 0
if refset == "clean":
    vb &= ~dirty(va["msg"])
L = te["per"][:, :, 0]
R = va["per"][vb, :, 0]

def emp(ref, x):
    ref = np.sort(ref)
    p = (len(ref) - np.searchsorted(ref, x, side="left") + 1) / (len(ref) + 1)
    return -np.log(p)

def emp_tail(ref, x, q=0.99):
    """Empirical surprisal below the q-quantile, exponential tail fitted above it."""
    ref = np.sort(ref)
    u = np.quantile(ref, q)
    exc = ref[ref > u] - u
    scale = exc.mean()
    s = emp(ref, x)
    hi = x > u
    s[hi] = -np.log(1 - q) + (x[hi] - u) / scale
    return s

def rz(ref, x):
    med = np.median(ref)
    mad = np.median(np.abs(ref - med)) * 1.4826
    return (x - med) / mad

sh = np.nan_to_num(te["shift"], posinf=0.0)  # score==1.0 (raw>33): shift not recoverable, irrelevant there
V = {
    "attuale (max logit + prec)": te["raw"] + sh,
    "max surprisal emp + prec": np.stack([emp(R[:, g], L[:, g]) for g in range(5)], 1).max(1) + sh,
    "max surprisal emp+coda + prec": np.stack([emp_tail(R[:, g], L[:, g]) for g in range(5)], 1).max(1) + sh,
    "max robust-z + prec": np.stack([rz(R[:, g], L[:, g]) for g in range(5)], 1).max(1) + sh,
}
if __name__ != "__main__":
    pass
names = {1: "policy", 2: "contextual", 3: "lateral", 4: "theft", 5: "exfil", 6: "benign-denied"}
ben = te["types"] == 0
y = te["y"]
print(f"reference = val benign ({refset}), n={vb.sum()}")
for k, s in V.items():
    thr = np.quantile(s[ben], 0.99)
    print(f"\n{k}: aggregate AUC={roc_auc_score(y, s):.3f} AP={average_precision_score(y, s):.3f} "
          f"rec@FPR1%={(s[y==1] > thr).mean():.3f} | ties sopra max benigno-val: "
          f"{(s >= s.max()).sum()}")
    for c, nm in names.items():
        m = te["types"] == c
        sel = ben | m
        yt = m[sel].astype(int)
        print(f"   {nm:13s} n={m.sum():4d} AUC={roc_auc_score(yt, s[sel]):.3f} "
              f"AP={average_precision_score(yt, s[sel]):.4f} rec@FPR1%={(s[m] > thr).mean():.3f}")
