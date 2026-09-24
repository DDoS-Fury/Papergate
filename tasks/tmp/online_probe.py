"""The real-time framing: per-request decision, causal features only.

The device-week numbers in tasks/todo.md use CALENDAR buckets, so each bucket contains
events later than the decision point — a look-ahead, not a deployable capability. The
deployable analogue of the same idea is a TRAILING window: at request i the accumulator may
read only that actor's own earlier events. This measures that, on the same scores, and adds
the two metrics a streaming gate actually lives or dies by:

  - time to detection: events and hours from the start of a campaign to its first alert;
  - campaigns stopped before they reach the exfiltration phase (type 5 on the same device),
    which is the operational question a ZTA gate answers.

Every threshold is fitted at 1% per-event benign FPR, so every row is comparable to the
per-event baseline and to the others.

Run:
  docker compose run --rm --no-deps -T -v "$PWD/tasks:/app/tasks" \
      --entrypoint python train-tgn /app/tasks/tmp/online_probe.py <dump.npz>
"""
import dataclasses, sys

import numpy as np
from sklearn.metrics import roc_auc_score

from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg

DAY = 86400.0
d = np.load(sys.argv[1])
cfg = dataclasses.replace(TGNConfig(), seed=2000)
s = generate_streaming_data(**stream_kwargs_from_cfg(cfg))
va = int(d["test_start"])
sco, lab = np.asarray(d["test_scores"], dtype=np.float64), np.asarray(d["test_labels"])
ty, dev, tt = s.types.numpy()[va:], s.device.numpy()[va:], s.t.numpy()[va:].astype(np.float64)
p = np.clip(sco, 1e-300, 1 - 1e-16)
lg = np.log(p) - np.log1p(-p)            # per-event anomaly logit, already causal
order = np.argsort(tt, kind="stable")
assert (np.diff(tt[order]) >= 0).all()

def trailing_topk(k: int, window_days: float) -> np.ndarray:
    """Mean of the k highest logits among that device's events in (t-W, t], current included.

    Strictly causal: the loop walks the stream in time order and only ever reads entries it
    has already appended for that device.
    """
    out = np.empty(len(lg))
    hist: dict[int, list[tuple[float, float]]] = {}
    w = window_days * DAY
    for i in order:
        a = int(dev[i]); t_i = tt[i]
        h = hist.setdefault(a, [])
        h.append((t_i, lg[i]))
        lo = t_i - w
        while h and h[0][0] < lo:
            h.pop(0)
        vals = sorted((v for _, v in h), reverse=True)[:k]
        out[i] = float(np.mean(vals))
    return out

ben, lat = lab == 0, ty == 3
variants = {"per-event (nessun accumulo)": lg}
for k, w in ((3, 7.0), (3, 14.0), (5, 7.0), (8, 14.0), (3, 3.0)):
    variants[f"trailing top-{k}, finestra {w:.0f}g"] = trailing_topk(k, w)

# campaigns: lateral events of one device split on a 7-day gap (see lateral_chain_diag.py)
idx_lat = np.nonzero(lat)[0]
camps = []
for m in np.unique(dev[idx_lat]):
    idx = idx_lat[dev[idx_lat] == m]
    idx = idx[np.argsort(tt[idx])]
    cur = [idx[0]]
    for i in idx[1:]:
        if tt[i] - tt[cur[-1]] < 7 * DAY:
            cur.append(i)
        else:
            camps.append(cur); cur = [i]
    camps.append(cur)
# first exfiltration event on each device, if any
first_exfil = {int(m): tt[np.nonzero((dev == m) & (ty == 5))[0]].min()
               for m in np.unique(dev[ty == 5])}
print(f"dump={sys.argv[1].split('/')[-1]}  campagne={len(camps)}  "
       f"device con exfil={len(first_exfil)}  eventi test={len(lg)}")
print(f"{'variante':30s} {'AUC':>6s} {'recall':>7s} {'prec':>6s} | {'camp.':>6s} "
      f"{'ev.aTTD':>8s} {'h aTTD':>8s} {'pre-exfil':>10s}")
print("-" * 100)
for name, v in variants.items():
    thr = float(np.quantile(v[ben], 0.99))
    flag = v >= thr
    tp, fp = int((flag & (lab == 1)).sum()), int((flag & ben).sum())
    auc = roc_auc_score(np.r_[np.zeros(ben.sum()), np.ones(lat.sum())], np.r_[v[ben], v[lat]])
    hit, ttd_ev, ttd_h, pre = 0, [], [], 0
    for c in camps:
        c = sorted(c, key=lambda i: tt[i])
        f = [j for j, i in enumerate(c) if flag[i]]
        if not f:
            continue
        hit += 1
        ttd_ev.append(f[0] + 1)
        ttd_h.append((tt[c[f[0]]] - tt[c[0]]) / 3600.0)
        fe = first_exfil.get(int(dev[c[0]]))
        if fe is None or tt[c[f[0]]] < fe:
            pre += 1
    print(f"{name:30s} {auc:6.4f} {flag[lat].mean():7.4f} "
          f"{tp / (tp + fp) if tp + fp else float('nan'):6.4f} | {hit:2d}/{len(camps):2d} "
          f"{np.median(ttd_ev):8.1f} {np.median(ttd_h):8.1f} {pre:4d}/{len(camps):2d}   ")
