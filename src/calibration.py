"""Decision-threshold calibration for the streaming TGN.

The model *ranks* anomalies well (lateral-movement AUC ~0.76) but a single threshold
picked only to hold a benign false-positive rate (``target_fpr``) sits above where the
signal-clean lateral scores cluster, so operational recall collapses to a few percent.

This module turns that ranking into recall with a **cost-sensitive** threshold: pick the
operating point that minimises ``cost_ratio * FN + FP``. With ``cost_ratio = C_fn/C_fp``
large (a missed lateral movement costs far more than a false alarm the orchestrator can
re-challenge) the threshold drops to where the laterals are, trading a controlled amount
of precision for recall. The functions are pure (numpy only) so both the training pipeline
and the external-dataset evaluation (``tests/eval_lanl.py``) reuse them.

The companion serving lever is *signal routing*: at inference time the true class is
unknown, but the edge signal (broken JA3 / Snort / sensors) is observable, so we apply the
recall-oriented threshold only to the **signal-clean** stream — where benign and lateral
are otherwise indistinguishable — and keep the conservative ``target_fpr`` threshold for
signal-dirty events (already caught by the cheap rule baseline). See ``routed_predict``.
"""

from __future__ import annotations

import numpy as np


def cost_sensitive_threshold(
    scores,
    labels,
    *,
    cost_ratio: float,
    target_fpr_cap: float | None = None,
) -> float:
    """Threshold minimising ``cost_ratio * FN + FP`` over the score distribution.

    ``scores`` are anomaly scores, ``labels`` the 0/1 ground truth on the **same**
    population the threshold will be applied to (e.g. the signal-clean validation slice
    for the clean-stream threshold). ``cost_ratio = C_fn / C_fp`` weights a missed
    detection against a false alarm; larger → lower threshold → higher recall.

    ``target_fpr_cap`` optionally restricts the search to thresholds whose benign
    false-positive rate does not exceed the cap (a guardrail so the recall push cannot
    blow up precision); if no candidate satisfies it, the global cost minimiser is used.

    Decision rule is ``score >= threshold``. Ties in cost are broken toward the **higher**
    threshold (fewer false positives at equal cost). Returns a finite float; a value above
    every observed score means "flag nothing".
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = np.sort(scores[labels == 1])
    neg = np.sort(scores[labels == 0])
    n_pos, n_neg = pos.size, neg.size

    # Candidate thresholds: every observed score, plus a sentinel just above the max so
    # "flag nothing" is reachable. ``score >= thr`` ⇒ at thr = a score, that event fires.
    sentinel = (scores.max() if scores.size else 1.0) + 1e-9
    cand = np.concatenate([np.unique(scores), [sentinel]])

    # FP = benign with score >= thr ; FN = malicious with score < thr  (vectorised).
    fp = n_neg - np.searchsorted(neg, cand, side="left")
    fn = np.searchsorted(pos, cand, side="left")
    cost = cost_ratio * fn + fp

    eligible = np.ones_like(cand, dtype=bool)
    if target_fpr_cap is not None and n_neg > 0:
        capped = (fp / n_neg) <= target_fpr_cap
        if capped.any():
            eligible = capped

    cost_masked = np.where(eligible, cost, np.inf)
    min_cost = cost_masked.min()
    # Highest threshold among the cost minimisers (fewest false positives at equal cost).
    best = cand[(cost_masked == min_cost)].max()
    return float(best)


def operating_point(scores, labels, types, threshold) -> dict:
    """Metrics of ``score >= threshold`` against ``labels`` (with per-type recall).

    ``types`` is the anomaly-type array (0 benign, 1 policy, 2 contextual, 3 lateral);
    per-type recall is reported for every non-benign type present.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels, dtype=np.int64)
    types = np.asarray(types, dtype=np.int64)
    preds = (scores >= threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    n_neg = int((labels == 0).sum())

    per_type = {}
    for ty in sorted(t for t in np.unique(types) if t != 0):
        sel = types == ty
        per_type[int(ty)] = float(preds[sel].mean()) if sel.any() else float("nan")

    return {
        "threshold": float(threshold),
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "fpr": fp / n_neg if n_neg else 0.0,
        "per_type_recall": per_type,
    }


def recall_fpr_curve(scores, labels, *, n_points: int = 11):
    """A small ``[(threshold, recall, fpr), ...]`` trade-off curve for reporting.

    Thresholds are sampled at evenly spaced score quantiles so an operator/OPA can read
    off the recall available at each benign false-positive rate. ``recall`` here is over
    all positives in ``labels`` (use the lateral subset upstream for a lateral-only curve).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = np.sort(scores[labels == 1])
    neg = np.sort(scores[labels == 0])
    n_pos, n_neg = pos.size, neg.size
    qs = np.linspace(0.0, 1.0, n_points)
    thrs = np.quantile(scores, qs) if scores.size else qs
    out = []
    for thr in thrs:
        rec = 1.0 - (np.searchsorted(pos, thr, side="left") / n_pos) if n_pos else float("nan")
        fpr = (n_neg - np.searchsorted(neg, thr, side="left")) / n_neg if n_neg else float("nan")
        out.append((float(thr), float(rec), float(fpr)))
    return out


def routed_predict(scores, dirty_mask, threshold_clean, threshold_dirty):
    """Signal-routed decision: clean events use ``threshold_clean``, dirty use ``dirty``.

    ``dirty_mask`` marks events whose edge signal already fires (broken JA3 / Snort /
    sensor). Those keep the conservative threshold; the recall-oriented clean threshold is
    applied only where benign and lateral are otherwise indistinguishable. Returns a 0/1
    prediction array. This mirrors what ``serve_tgn.score_event`` does online per event.
    """
    scores = np.asarray(scores)
    dirty_mask = np.asarray(dirty_mask, dtype=bool)
    eff = np.where(dirty_mask, threshold_dirty, threshold_clean)
    return (scores >= eff).astype(int)
