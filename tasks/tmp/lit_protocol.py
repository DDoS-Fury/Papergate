"""Our numbers under the protocol the lateral-movement literature uses, so the comparison
is on the same metrics even though the datasets differ.

Euler (NDSS'22) and the ANSSI GFM study report AUC, average precision (AP) and Rec@B —
the recall obtained when an analyst investigates the B highest-scoring items in each time
window. AP and Rec@B are the honest metrics at extreme imbalance; AUC is not (base rate
fallacy). Time windows are 30 minutes, as in Euler (delta = 1800 s).

Run:
  docker compose run --rm --no-deps -T -v "$PWD/tasks:/app/tasks" \
      --entrypoint python train-tgn /app/tasks/tmp/lit_protocol.py <dump.npz>
"""
import dataclasses, sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg

d = np.load(sys.argv[1])
cfg = dataclasses.replace(TGNConfig(), seed=2000)
s = generate_streaming_data(**stream_kwargs_from_cfg(cfg))
va = int(d["test_start"])
sco, lab = np.asarray(d["test_scores"], dtype=np.float64), np.asarray(d["test_labels"])
ty, tt = s.types.numpy()[va:], s.t.numpy()[va:].astype(np.float64)

for name, pos in (("lateral (tipo 3)", ty == 3), ("tutte le anomalie", lab == 1)):
    keep = pos | (lab == 0)
    y, v = pos[keep].astype(int), sco[keep]
    auc, ap = roc_auc_score(y, v), average_precision_score(y, v)
    prev = y.mean()
    print(f"\n{name}: positivi={y.sum()} negativi={(y == 0).sum()} prevalenza={prev:.5f}")
    print(f"  AUC={auc:.4f}  AP={ap:.4f}  (AP di un classificatore casuale = {prev:.5f}, "
          f"lift={ap / prev:.1f}x)")
    win = ((tt[keep] - tt[keep].min()) // 1800).astype(np.int64)   # 30 min, come Euler
    for B in (3, 5, 10):
        caught = 0
        for w in np.unique(win):
            m = win == w
            vv, yy = v[m], y[m]
            if yy.sum() == 0:
                continue
            top = np.argsort(vv)[::-1][:B]
            caught += int(yy[top].sum())
        print(f"  Rec@{B:<2d} = {caught / y.sum():.4f}  ({caught}/{y.sum()})")
    print(f"  finestre da 30 min con >=1 positivo: "
          f"{len({int(w) for w in win[y == 1]})} su {len(np.unique(win))}")
