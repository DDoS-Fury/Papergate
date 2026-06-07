"""Unit check for cost-sensitive / signal-routed thresholding (graphagate.calibration).

Runnable in the repo style (no pytest dependency):

    docker run --rm -v "$PWD/src:/app/src" -v "$PWD/tests:/app/tests" \
        --entrypoint python graphagate /app/tests/test_calibration.py

It builds a score distribution that reproduces the operational problem — lateral-movement
scores ranked above benign but sitting *below* the benign-FPR threshold — and asserts that:
  1. the cost-sensitive threshold recovers materially more lateral recall than the FPR one;
  2. signal routing never raises the benign false-positive rate above a stated bound;
  3. routed_predict applies the per-stream thresholds element-wise as documented.
"""

import sys

import numpy as np

from graphagate.calibration import (
    cost_sensitive_threshold,
    operating_point,
    recall_fpr_curve,
    routed_predict,
)


def _make_clean_stream(seed=0):
    """Signal-clean validation slice: benign vs lateral, laterals only mildly elevated."""
    rng = np.random.default_rng(seed)
    benign = np.clip(rng.normal(0.10, 0.05, size=5000), 0, 1)
    lateral = np.clip(rng.normal(0.18, 0.05, size=200), 0, 1)
    scores = np.concatenate([benign, lateral])
    labels = np.concatenate([np.zeros(benign.size), np.ones(lateral.size)]).astype(int)
    return scores, labels, benign


def main() -> int:
    failures = []
    scores, labels, benign = _make_clean_stream()

    # 1. Cost-sensitive recovers more lateral recall than the benign-FPR threshold.
    thr_fpr = float(np.quantile(benign, 0.99))  # the legacy threshold@FPR=1%
    thr_cost = cost_sensitive_threshold(labels=labels, scores=scores, cost_ratio=20.0)
    op_fpr = operating_point(scores, labels, labels * 3, thr_fpr)   # types: 0 / 3 (lateral)
    op_cost = operating_point(scores, labels, labels * 3, thr_cost)
    print(f"FPR  threshold={thr_fpr:.4f} | lateral_recall={op_fpr['recall']:.3f} | fpr={op_fpr['fpr']:.4f}")
    print(f"cost threshold={thr_cost:.4f} | lateral_recall={op_cost['recall']:.3f} | fpr={op_cost['fpr']:.4f}")
    if not (thr_cost < thr_fpr):
        failures.append("cost-sensitive threshold should be below the FPR threshold")
    if not (op_cost["recall"] > op_fpr["recall"] + 0.10):
        failures.append("cost-sensitive threshold should lift lateral recall by >0.10")

    # 2. FPR cap guardrail: capping benign FPR keeps it within bound (recall traded away).
    thr_capped = cost_sensitive_threshold(scores=scores, labels=labels, cost_ratio=20.0,
                                          target_fpr_cap=0.02)
    op_capped = operating_point(scores, labels, labels * 3, thr_capped)
    print(f"capped(2%) threshold={thr_capped:.4f} | fpr={op_capped['fpr']:.4f}")
    if not (op_capped["fpr"] <= 0.02 + 1e-9):
        failures.append("target_fpr_cap must bound the benign FPR")

    # 3. routed_predict applies clean/dirty thresholds element-wise.
    s = np.array([0.30, 0.30, 0.05, 0.05])
    dirty = np.array([False, True, False, True])
    preds = routed_predict(s, dirty, threshold_clean=0.20, threshold_dirty=0.50)
    # clean 0.30 >= 0.20 -> 1 ; dirty 0.30 < 0.50 -> 0 ; clean 0.05 < 0.20 -> 0 ; dirty 0.05 < 0.50 -> 0
    if preds.tolist() != [1, 0, 0, 0]:
        failures.append(f"routed_predict mismatch: {preds.tolist()} != [1,0,0,0]")

    # 4. recall_fpr_curve is monotone non-increasing in both columns as threshold rises.
    curve = recall_fpr_curve(scores, labels, n_points=6)
    recs = [r for _, r, _ in curve]
    fprs = [f for _, _, f in curve]
    if any(recs[i] < recs[i + 1] - 1e-9 for i in range(len(recs) - 1)):
        failures.append("recall_fpr_curve recall should be non-increasing in threshold")
    if any(fprs[i] < fprs[i + 1] - 1e-9 for i in range(len(fprs) - 1)):
        failures.append("recall_fpr_curve fpr should be non-increasing in threshold")

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nOK: cost-sensitive + signal-routing thresholding behaves as specified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
