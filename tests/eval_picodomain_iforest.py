"""Isolation Forest anomaly-detection baseline on PicoDomain Zeek telemetry.

Runs a non-relational, one-class Isolation Forest on the mapped PicoDomain stream
(using the exact same 70/10/20 chronological split and 45-dim per-event tabular features
as in the paper's baselines).

Quantifies how much detection is possible WITHOUT graph structure / temporal memory on
real enterprise network telemetry (Kerberos + SMB + DCE-RPC + TLS JA3).

Usage:
    python tests/eval_picodomain_iforest.py [--log-dir /data/logs] [--red-log /data/Red Log.xlsx]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

# Ensure the sibling `datasets` package and root `src` are importable.
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CUR_DIR)
sys.path.insert(0, CUR_DIR)
sys.path.insert(0, os.path.join(PARENT_DIR, "src"))

from datasets.picodomain import (  # noqa: E402
    T_BENIGN,
    T_CONTEXTUAL,
    T_LATERAL,
    T_THEFT,
    load_picodomain_stream,
)
from graphagate.eval_common import causal_hist_features  # noqa: E402


def _binary_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float, float]:
    """Precision and recall of ``scores >= threshold`` against binary ``labels``."""
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def _build_features(msg, src, dst, node_features, y) -> np.ndarray:
    """Build the 45-dim tabular feature matrix for each event."""
    msg_np = msg.numpy()
    nf_np = node_features.numpy()
    src_feat = nf_np[src.numpy()]
    dst_feat = nf_np[dst.numpy()]
    hist = causal_hist_features(src.numpy(), dst.numpy(), y.numpy())
    return np.concatenate([msg_np, src_feat, dst_feat, hist], axis=1)


def main() -> int:
    default_log_dir = "/data/logs" if os.path.exists("/data/logs") else os.path.join(PARENT_DIR, "data", "logs")
    default_red_log = "/data/Red Log.xlsx" if os.path.exists("/data/Red Log.xlsx") else os.path.join(PARENT_DIR, "data", "Red Log.xlsx")

    parser = argparse.ArgumentParser(description="Evaluate Isolation Forest on PicoDomain Zeek logs.")
    parser.add_argument("--log-dir", default=os.environ.get("PICO_LOGS", default_log_dir), help=f"Path to extracted Zeek logs (default: {default_log_dir})")
    parser.add_argument("--red-log", default=os.environ.get("PICO_REDLOG", default_red_log), help=f"Path to Red Log.xlsx (default: {default_red_log})")
    parser.add_argument("--max-events", type=int, default=200_000)
    parser.add_argument("--bind-ttl", type=float, default=36_000.0)
    parser.add_argument("--label-window", type=float, default=90.0)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("================================================================")
    print("      PicoDomain Baseline: ISOLATION FOREST                     ")
    print("================================================================")
    print(f"Log dir:        {args.log_dir}")
    print(f"Red log:        {args.red_log}")
    print(f"Train/Val split: {args.train_frac:.1%} / {args.val_frac:.1%}")
    print(f"Random seed:    {args.seed}\n")

    print("--- STEP 1: LOADING PICODOMAIN STREAM ---")
    data = load_picodomain_stream(
        args.log_dir,
        args.red_log,
        max_events=args.max_events,
        bind_ttl=args.bind_ttl,
        label_window=args.label_window,
    )

    print("\n--- STEP 2: EXTRACTING 45-DIMENSIONAL FEATURES ---")
    actor_node = data.device_nodes if data.device_nodes is not None else data.user
    X = _build_features(data.msg, actor_node, data.dst, data.node_features, data.y)
    y = data.y.numpy()
    types = data.types.numpy()

    n = len(y)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    train_end = n_train
    val_end = n_train + n_val

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    test_types = types[val_end:]

    print(f"Features matrix shape: {X.shape}")
    print(f"Split: train={len(X_train)} (anomalies: {int(y_train.sum())}), "
          f"val={len(X_val)} (anomalies: {int(y_val.sum())}), "
          f"test={len(X_test)} (anomalies: {int(y_test.sum())})")

    print("\n--- STEP 3: TRAINING ONE-CLASS ISOLATION FOREST (Benign Only) ---")
    benign_train_mask = y_train == 0
    X_train_benign = X_train[benign_train_mask]
    print(f"Fitting on {len(X_train_benign)} benign events...")

    iso = IsolationForest(
        n_estimators=150,
        contamination="auto",
        random_state=args.seed,
        n_jobs=-1,
    )
    iso.fit(X_train_benign)

    print("\n--- STEP 4: THRESHOLD CALIBRATION (Benign Validation Slice) ---")
    val_benign_mask = y_val == 0
    val_scores = -iso.score_samples(X_val[val_benign_mask])
    threshold = float(np.percentile(val_scores, 100.0 * (1.0 - args.target_fpr)))
    print(f"Calibrated threshold at FPR={args.target_fpr:.2f}: {threshold:.4f}")

    print("\n--- STEP 5: EVALUATION ON TEST STREAM ---")
    test_scores = -iso.score_samples(X_test)
    agg_auc = roc_auc_score(y_test, test_scores)
    agg_ap = average_precision_score(y_test, test_scores)
    prec, rec = _binary_metrics(test_scores, y_test, threshold)
    benign_test_mask = y_test == 0
    benign_fpr = float((test_scores[benign_test_mask] >= threshold).mean()) if benign_test_mask.any() else 0.0

    print("================================================================")
    print("             Isolation Forest RESULTS on PicoDomain             ")
    print("================================================================")
    print(f"  Aggregate ROC-AUC:      {agg_auc:.4f}")
    print(f"  Aggregate PR-AUC (AP):  {agg_ap:.4f}")
    print(f"  Precision @ threshold:  {prec:.4f}")
    print(f"  Recall @ threshold:     {rec:.4f}")
    print(f"  Benign FPR:             {benign_fpr:.4f}")
    print("----------------------------------------------------------------")
    print("  Per-Tactic Breakdown:")

    type_mapping = [
        ("theft (Credential Theft)", T_THEFT),
        ("contextual (C2/Recon)", T_CONTEXTUAL),
        ("lateral (Lateral Movement)", T_LATERAL),
    ]

    for name, code in type_mapping:
        mask = (test_types == code) | (test_types == T_BENIGN)
        n_pos = int((test_types == code).sum())
        if n_pos == 0 or not mask.any():
            continue
        y_sub = (test_types[mask] == code).astype(int)
        scores_sub = test_scores[mask]
        auc_sub = roc_auc_score(y_sub, scores_sub) if len(np.unique(y_sub)) > 1 else 0.0
        ap_sub = average_precision_score(y_sub, scores_sub) if len(np.unique(y_sub)) > 1 else 0.0
        _, rec_sub = _binary_metrics(scores_sub, y_sub, threshold)
        print(f"    - {name:<28s} AUC={auc_sub:.4f}  AP={ap_sub:.4f}  Recall={rec_sub:.4f}  (n={n_pos})")
    print("================================================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
