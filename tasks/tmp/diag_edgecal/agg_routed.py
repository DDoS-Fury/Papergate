"""Stream-routed view: clean and dirty events ranked/thresholded separately (as the pipeline routes)."""
import sys
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
sys.argv = [sys.argv[0], sys.argv[1], "clean"]
exec(open(sys.argv[0].replace("agg_routed", "agg_variants")).read().split("names = {1:")[0])
names = {1: "policy", 2: "contextual", 3: "lateral", 4: "theft", 5: "exfil", 6: "benign-denied"}
dt = dirty(te["msg"])
ben = te["types"] == 0
for k in ("attuale (max logit + prec)", "max surprisal emp+coda + prec"):
    s = V[k]
    print(f"\n{k}")
    for sub, nm_s in ((~dt, "CLEAN"), (dt, "DIRTY")):
        b = ben & sub
        thr = np.quantile(s[b], 0.99)
        cells = []
        for c, nm in names.items():
            m = (te["types"] == c) & sub
            if m.sum() < 5:
                continue
            sel = b | m
            yt = m[sel].astype(int)
            cells.append(f"{nm}(n={m.sum()}) AP={average_precision_score(yt, s[sel]):.3f} "
                         f"rec@1%={(s[m] > thr).mean():.3f}")
        print(f"  {nm_s} (benign n={b.sum()}): " + " | ".join(cells))
