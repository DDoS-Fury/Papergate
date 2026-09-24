"""De-risk point 5 offline: does ACCUMULATING the score per actor beat OR-ing it?

The OR (any event over threshold flags the bucket) is what a per-event detector gives you
for free, and it is the worst aggregator for the false-alarm rate: one false positive lights
the whole window. An accumulator averages the noise down instead. Both are computed here on
the SAME scores, at bucket level, so the difference is the information the accumulator adds
and nothing else. If the accumulator's bucket AUC is not above the max's, point 5 is not
worth building.

Buckets are (device, day) / (device, week) of the test slice; a bucket is positive if it
contains >= 1 ground-truth lateral event, negative if it contains no anomaly at all.

Run:
  docker compose run --rm --no-deps -T -v "$PWD/tasks:/app/tasks" \
      --entrypoint python train-tgn /app/tasks/tmp/aggregator_probe.py <dump.npz>
"""
import dataclasses, sys

import numpy as np
from sklearn.metrics import roc_auc_score

from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg

d = np.load(sys.argv[1])
cfg = dataclasses.replace(TGNConfig(), seed=2000)
s = generate_streaming_data(**stream_kwargs_from_cfg(cfg))
va = int(d["test_start"])
sco, lab = np.asarray(d["test_scores"], dtype=np.float64), d["test_labels"]
ty, dev, tt = s.types.numpy()[va:], s.device.numpy()[va:], s.t.numpy()[va:]
# Work on the logit: the accumulator has to sum evidence, and probabilities near 1 do not add.
p = np.clip(sco, 1e-300, 1 - 1e-16)
lg = np.log(p) - np.log1p(-p)

print(f"dump={sys.argv[1].split('/')[-1]}")
print(f"per-event lateral AUC (benign vs lateral) = "
      f"{roc_auc_score(np.r_[np.zeros((lab == 0).sum()), np.ones((ty == 3).sum())], np.r_[lg[lab == 0], lg[ty == 3]]):.4f}")

def topk_sum(v, k):
    v = np.sort(v)[::-1][:k]
    return float(v.sum())

for label, span in (("device-day", 86400), ("device-week", 7 * 86400)):
    b = dev.astype(np.int64) * 10_000 + (tt - tt.min()) // span
    keys, inv = np.unique(b, return_inverse=True)
    n = len(keys)
    has_lat = np.zeros(n, bool); has_any = np.zeros(n, bool)
    np.logical_or.at(has_lat, inv, ty == 3)
    np.logical_or.at(has_any, inv, lab == 1)
    keep = has_lat | ~has_any          # positives = lateral buckets, negatives = clean buckets
    y = has_lat[keep].astype(int)
    groups = [lg[inv == i] for i in np.nonzero(keep)[0]]
    sizes = np.array([len(g) for g in groups])
    aggs = {
        "max  (= OR, free)": np.array([g.max() for g in groups]),
        "mean": np.array([g.mean() for g in groups]),
        "sum": np.array([g.sum() for g in groups]),
        "top-3 sum": np.array([topk_sum(g, 3) for g in groups]),
        "top-3 mean": np.array([topk_sum(g, 3) / min(3, len(g)) for g in groups]),
    }
    print(f"\n{label}: buckets={keep.sum()} (pos={y.sum()} neg={(y == 0).sum()}) "
          f"events/bucket median={np.median(sizes):.0f} max={sizes.max()}")
    for name, v in aggs.items():
        auc = roc_auc_score(y, v)
        # recall at the 1%-FPR point of the NEGATIVE buckets, i.e. same budget discipline
        t = np.quantile(v[y == 0], 0.99)
        rec = float((v[y == 1] >= t).mean())
        t5 = np.quantile(v[y == 0], 0.95)
        rec5 = float((v[y == 1] >= t5).mean())
        line = f"  {name:18s} AUC={auc:.4f}"
        for tag, q in (("1%", 0.99), ("5%", 0.95)):
            th = np.quantile(v[y == 0], q)
            tp = int((v[y == 1] >= th).sum()); fp = int((v[y == 0] >= th).sum())
            line += (f" | @{tag}FPR recall={tp / y.sum():.3f} "
                     f"prec={tp / (tp + fp) if tp + fp else float('nan'):.3f} (TP={tp} FP={fp})")
        print(line)
