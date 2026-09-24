"""Point 6 done honestly: if the positives are counted per device-window, the false alarms
must be counted per device-window too. Otherwise 'campaign recall' buys recall by spreading
the same false positives over more buckets and never paying for them.

A bucket is (device, calendar day) or (device, calendar week) inside the test slice.
  recall = compromised buckets with >= 1 flagged event / compromised buckets
  FPR    = clean buckets with >= 1 flagged event / clean buckets
A compromised bucket is one containing >= 1 ground-truth lateral event; a clean bucket is
one containing no anomaly of any type.

Run:
  docker compose run --rm --no-deps -T -v "$PWD/tasks:/app/tasks" \
      --entrypoint python train-tgn /app/tasks/tmp/agg_metrics.py <dump.npz>
"""
import dataclasses, sys

import numpy as np

from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg

d = np.load(sys.argv[1])
cfg = dataclasses.replace(TGNConfig(), seed=2000)
s = generate_streaming_data(**stream_kwargs_from_cfg(cfg))
va = int(d["test_start"])
sco, lab = d["test_scores"], d["test_labels"]
thr = float(d["thr_global"])
ty, dev, tt = s.types.numpy()[va:], s.device.numpy()[va:], s.t.numpy()[va:]
flag = sco >= thr

print(f"dump={sys.argv[1].split('/')[-1]}  thr={thr:.6f}")
print(f"per-event: lateral recall={(sco[ty == 3] >= thr).mean():.4f} "
      f"benign FPR={(sco[lab == 0] >= thr).mean():.4f}")

for label, span in (("device-day", 86400), ("device-week", 7 * 86400)):
    b = dev.astype(np.int64) * 10_000 + (tt - tt.min()) // span
    keys, inv = np.unique(b, return_inverse=True)
    n = len(keys)
    has_lat = np.zeros(n, bool); has_any = np.zeros(n, bool); has_flag = np.zeros(n, bool)
    np.logical_or.at(has_lat, inv, ty == 3)
    np.logical_or.at(has_any, inv, lab == 1)
    np.logical_or.at(has_flag, inv, flag)
    clean = ~has_any
    rec = has_flag[has_lat].mean()
    fpr = has_flag[clean].mean()
    # Precision at this aggregation: compromised buckets among the alerting ones.
    alert = has_flag
    prec = has_lat[alert].mean() if alert.any() else float("nan")
    print(f"\n{label}: buckets={n} compromised={has_lat.sum()} clean={clean.sum()}")
    print(f"  recall={rec:.4f} ({has_flag[has_lat].sum()}/{has_lat.sum()})  "
          f"FPR={fpr:.4f} ({has_flag[clean].sum()}/{clean.sum()})  "
          f"precision={prec:.4f} ({has_lat[alert].sum()}/{alert.sum()})")
    # How much of the alert budget the aggregation actually costs, in buckets per day.
    days = (tt.max() - tt.min()) / 86400
    print(f"  alerting buckets/day={alert.sum() / days:.1f} over {days:.0f} days, "
          f"{len(np.unique(dev))} devices")
