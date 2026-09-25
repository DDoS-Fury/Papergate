"""Compare two fusion_e2e.py dumps: pass iff every array agrees to |Δ| <= 1e-7 + 1e-5·max|ref|."""
import sys, numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
assert (a["sd_keys"] == b["sd_keys"]).all(), "state_dict keys/shapes differ"
if "sd_maxdiff" in b: print("state_dict loaded strictly; max |Δ| seeded init:", float(b["sd_maxdiff"]))
ok, ng = True, 0
for k in a.files:
    if k == "sd_keys": continue
    if k not in b.files: print("MISSING", k); ok = False; continue
    d = float(np.max(np.abs(a[k] - b[k]))) if a[k].size else 0.0
    scale = float(np.max(np.abs(a[k]))) if a[k].size else 0.0
    good = d <= 1e-7 + 1e-5 * scale
    ok &= good
    ng += k.startswith("grad:")
    if not k.startswith("grad:") or not good:
        print(f"{k:28s} max|Δ|={d:.3e} (scale {scale:.2e}) {'ok' if good else 'FAIL'}")
print("grads compared:", ng)
print("OVERALL", "PASS" if ok else "FAIL")
