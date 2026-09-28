"""Evaluate the TGN on DARPA OpTC with the LMDEval protocol (Larroche 2026, arXiv 2607.29390).

Input: the flow list built by ``scripts/optc_extract.py build``. The run reuses
``train_tgn(dataset=...)`` (one-class training on the benign slice, benign calibration,
streaming test replay with commit-everything, since OpTC has no policy engine) and then
computes the protocol's per-event metrics from the raw test scores:

  * LM only   : positives = "Lateral movement" ids, every other test event is a negative
                (as LMDEval's ``--lm-only``: "Other" red-team events count as negatives);
  * all       : positives = every red-team id.

Also printed: LM vs benign only (Other left out), prevalence, TPR at 1% / 0.1% FPR on the test
ROC, alerts per day at the validation 1%-FPR threshold. Raw scores go to ``--scores-out``
before any metric is computed.

Usage (docker-compose profile ``eval-optc``, or):
    python tests/eval_optc.py --flows /data/optc/optc_flows.csv.gz \
        --val-start 2019-09-22T20:00:00-04:00 --test-start 2019-09-23T00:00:00-04:00
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.optc import T_LATERAL, T_OTHER, load_optc_stream, to_ts  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve  # noqa: E402

from graphagate.config import TGNConfig  # noqa: E402
from graphagate.train_tgn import train_tgn  # noqa: E402


def tpr_at_fpr(y, s, fpr_max: float) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= fpr_max].max())


def report(name: str, y: np.ndarray, s: np.ndarray) -> dict:
    if y.sum() == 0 or y.sum() == len(y):
        print(f"  {name:<22} n_pos={int(y.sum())}: metriche non definite")
        return {}
    r = {"n_pos": int(y.sum()), "n": int(len(y)), "prevalence": float(y.mean()),
         "auc": float(roc_auc_score(y, s)), "ap": float(average_precision_score(y, s)),
         "tpr@1%": tpr_at_fpr(y, s, 0.01), "tpr@0.1%": tpr_at_fpr(y, s, 0.001)}
    print(f"  {name:<22} n_pos={r['n_pos']:>5} n={r['n']:>9} prev={r['prevalence']:.2e} "
          f"AUC={r['auc']:.4f} AP={r['ap']:.4f} (x{r['ap'] / r['prevalence']:.1f} sul caso) "
          f"TPR@1%={r['tpr@1%']:.3f} TPR@0.1%={r['tpr@0.1%']:.3f}")
    return r


def main() -> int:
    p = argparse.ArgumentParser(description="TGN on OpTC, LMDEval protocol.")
    p.add_argument("--flows", default="/data/optc/optc_flows.csv.gz")
    p.add_argument("--val-start", required=True, help="ISO time: end of training, start of calibration")
    p.add_argument("--test-start", required=True, help="ISO time: start of the test period")
    p.add_argument("--t-min", default=None, help="ISO time: drop earlier flows")
    p.add_argument("--t-max", default=None, help="ISO time: drop later flows")
    p.add_argument("--nodes", choices=("lmdeval", "enriched"), default="lmdeval")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=TGNConfig().seed)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--scores-out", default=None, help=".npz with the raw test scores and labels")
    args = p.parse_args()

    print("--- LOADING OpTC FLOWS ---")
    data, train_frac, val_frac, df = load_optc_stream(
        args.flows, val_start=to_ts(args.val_start), test_start=to_ts(args.test_start), nodes=args.nodes,
        t_min=to_ts(args.t_min) if args.t_min else None, t_max=to_ts(args.t_max) if args.t_max else None)

    cfg = dataclasses.replace(TGNConfig(), epochs=args.epochs, seed=args.seed, train_frac=train_frac,
                              val_frac=val_frac, eval_batch_size=args.eval_batch_size)
    print("\n--- TRAIN + EVALUATE (OpTC, LMDEval) ---")
    m = train_tgn(cfg, dataset=data, save=False, return_scores=True)

    s = np.asarray(m["test_scores"], dtype=np.float64)
    test = df.iloc[m["test_start"]:]
    types = data.types[m["test_start"]:].numpy()
    assert len(s) == len(test) == len(types)
    if args.scores_out:
        np.savez_compressed(args.scores_out, scores=s, types=types, ts=test["timestamp_abs"].to_numpy(),
                            threshold_dirty=m["threshold_dirty"])
        print(f"punteggi grezzi -> {args.scores_out}")

    days = (test["timestamp_abs"].iloc[-1] - test["timestamp_abs"].iloc[0]) / 86400
    print(f"\n--- OpTC / LMDEval SUMMARY (nodes={args.nodes}, seed={args.seed}, "
          f"eval_batch_size={args.eval_batch_size}, test={len(s)} eventi, {days:.2f} giorni) ---")
    lm, oth = types == T_LATERAL, types == T_OTHER
    report("LM only (Other=neg)", lm.astype(int), s)
    report("all malicious", (lm | oth).astype(int), s)
    keep = ~oth
    report("LM vs benign", lm[keep].astype(int), s[keep])
    thr = m["threshold_dirty"]
    alerts = int((s[~(lm | oth)] >= thr).sum())
    print(f"  soglia val FPR 1% = {thr:.6f}: allarmi benigni={alerts} ({alerts / max(days, 1e-9):.0f}/giorno), "
          f"LM rilevati={int((s[lm] >= thr).sum())}/{int(lm.sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
