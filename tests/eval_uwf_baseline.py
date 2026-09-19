"""Tabular Anomaly-Detection Baselines (Isolation Forest & XGBoost) on UWF-ZeekData24.

Compares classic non-relational baselines (Isolation Forest, XGBoost) against the streaming TGN
on the UWF-ZeekData24 dataset.

Protocol (mirrors ``graphagate.train_tgn`` & ``tests/eval_uwf_zeekdata.py``):
  1. Same UWF-ZeekData24 stream and chronological split (70% train / 10% val / 20% test).
  2. Per-event static feature vector (45 dims):
     - 10-dim edge message (service method, volumes, timing recency)
     - 16-dim source node features
     - 16-dim destination node features
     - 3-dim causal interaction history counters (per-pair / per-src counts)
  3. Models:
     - Isolation Forest: One-class unsupervised fit on benign training traffic only (``y_train == 0``).
       Decision threshold calibrated on benign validation at target FPR (default: 99th percentile).
     - XGBoost: Supervised gradient boosting classifier trained to distinguish benign from attacks.

Usage:
    python tests/eval_uwf_baseline.py --model isolation_forest [--data-dir data/uwf_zeekdata24]
    python tests/eval_uwf_baseline.py --model xgboost [--data-dir data/uwf_zeekdata24]
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

from datasets.uwf_zeekdata import (  # noqa: E402
    T_BENIGN,
    T_CONTEXTUAL,
    T_EXFIL,
    T_LATERAL,
    T_THEFT,
    load_uwf_stream,
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


def run_isolation_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_types: np.ndarray,
    *,
    n_estimators: int = 150,
    target_fpr: float = 0.01,
    seed: int = 42,
) -> None:
    print("\n--- TRAINING: ONE-CLASS ISOLATION FOREST (Benign Only) ---")
    # Fit strictly on benign training traffic (one-class anomaly detection)
    benign_train_mask = y_train == 0
    X_train_benign = X_train[benign_train_mask]
    print(f"Fitting on {len(X_train_benign)} benign events (out of {len(X_train)} train events)...")

    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )
    iso.fit(X_train_benign)

    print("\n--- THRESHOLD CALIBRATION (Validation Set) ---")
    # Anomaly score: higher = more anomalous
    val_benign_mask = y_val == 0
    val_scores = -iso.score_samples(X_val[val_benign_mask])
    threshold = float(np.percentile(val_scores, 100.0 * (1.0 - target_fpr)))
    print(f"Calibrated threshold at FPR={target_fpr:.2f}: {threshold:.4f}")

    print("\n--- EVALUATION: TEST STREAM ---")
    test_scores = -iso.score_samples(X_test)
    _report_results("Isolation Forest", test_scores, y_test, test_types, threshold)


def run_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_types: np.ndarray,
    *,
    seed: int = 42,
) -> None:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("Error: xgboost is not installed. Install it via pip or run inside Docker.", file=sys.stderr)
        return

    print("\n--- TRAINING: SUPERVISED XGBOOST ---")
    # Note: If train slice is 100% benign, a purely supervised classifier needs positive examples.
    # We take a small stratified validation slice of attacks to provide supervision,
    # or train on available labels.
    if y_train.sum() == 0:
        print("[!] Note: Train slice is 100% benign (one-class regime).")
        print("    For supervised XGBoost, mixing 10% of attack events into training to provide class labels...")
        # Borrow a small fraction of attack events from test for training supervision
        attack_indices = np.where(y_test == 1)[0]
        np.random.seed(seed)
        n_borrow = min(len(attack_indices) // 4, 1000)
        borrowed = np.random.choice(attack_indices, size=n_borrow, replace=False)

        X_train_xgb = np.vstack([X_train, X_test[borrowed]])
        y_train_xgb = np.concatenate([y_train, y_test[borrowed]])

        # Remove borrowed from test to avoid train-test contamination
        remaining_test_mask = np.ones(len(y_test), dtype=bool)
        remaining_test_mask[borrowed] = False
        X_test_eval = X_test[remaining_test_mask]
        y_test_eval = y_test[remaining_test_mask]
        test_types_eval = test_types[remaining_test_mask]
    else:
        X_train_xgb, y_train_xgb = X_train, y_train
        X_test_eval, y_test_eval = X_test, y_test
        test_types_eval = test_types

    print(f"XGBoost train events: {len(X_train_xgb)} ({int(y_train_xgb.sum())} positive, {int((y_train_xgb == 0).sum())} benign)")

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train_xgb, y_train_xgb)

    # Score = probability of anomaly P(y=1)
    val_probs = clf.predict_proba(X_val)[:, 1]
    threshold = float(np.percentile(val_probs, 99.0))  # 99th percentile of validation
    print(f"Calibrated threshold (99th percentile): {threshold:.4f}")

    print("\n--- EVALUATION: TEST STREAM ---")
    test_scores = clf.predict_proba(X_test_eval)[:, 1]
    _report_results("XGBoost", test_scores, y_test_eval, test_types_eval, threshold)


def _report_results(
    model_name: str,
    scores: np.ndarray,
    y_test: np.ndarray,
    test_types: np.ndarray,
    threshold: float,
) -> None:
    agg_auc = roc_auc_score(y_test, scores)
    agg_ap = average_precision_score(y_test, scores)
    prec, rec = _binary_metrics(scores, y_test, threshold)
    benign_mask = y_test == 0
    benign_fpr = float((scores[benign_mask] >= threshold).mean()) if benign_mask.any() else 0.0

    print("================================================================")
    print(f"             {model_name} RESULTS on UWF-ZeekData24              ")
    print("================================================================")
    print(f"  Aggregate ROC-AUC:      {agg_auc:.4f}")
    print(f"  Aggregate PR-AUC (AP):  {agg_ap:.4f}")
    print(f"  Precision @ threshold:  {prec:.4f}")
    print(f"  Recall @ threshold:     {rec:.4f}")
    print(f"  Benign FPR:             {benign_fpr:.4f}")
    print("----------------------------------------------------------------")
    print("  Per-Tactic Breakdown:")

    type_mapping = [
        ("theft (Credential Access)", T_THEFT),
        ("contextual (Reconnaissance)", T_CONTEXTUAL),
        ("lateral (Intrusion/Lateral)", T_LATERAL),
        ("exfil (Exfiltration)", T_EXFIL),
    ]

    for name, code in type_mapping:
        mask = (test_types == code) | (test_types == T_BENIGN)
        n_pos = int((test_types == code).sum())
        if n_pos == 0 or not mask.any():
            continue
        y_sub = (test_types[mask] == code).astype(int)
        scores_sub = scores[mask]
        auc_sub = roc_auc_score(y_sub, scores_sub) if len(np.unique(y_sub)) > 1 else 0.0
        ap_sub = average_precision_score(y_sub, scores_sub) if len(np.unique(y_sub)) > 1 else 0.0
        _, rec_sub = _binary_metrics(scores_sub, y_sub, threshold)
        print(f"    - {name:<28s} AUC={auc_sub:.4f}  AP={ap_sub:.4f}  Recall={rec_sub:.4f}  (n={n_pos})")
    print("================================================================\n")


def main() -> int:
    default_data_dir = os.path.join(PARENT_DIR, "data", "uwf_zeekdata24")
    parser = argparse.ArgumentParser(description="Run Isolation Forest / XGBoost on UWF-ZeekData24.")
    parser.add_argument(
        "--model",
        choices=["isolation_forest", "xgboost"],
        default="isolation_forest",
        help="Baseline model to run (default: isolation_forest)",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("UWF_DATA_DIR", default_data_dir),
        help=f"Path to UWF-ZeekData24 directory (default: {default_data_dir})",
    )
    parser.add_argument("--max-benign", type=int, default=30_000, help="Max benign events to load (default: 30000)")
    parser.add_argument("--max-attack", type=int, default=3_000, help="Max attack events per category (default: 3000)")
    parser.add_argument("--train-frac", type=float, default=0.7, help="Train split fraction (default: 0.7)")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Validation split fraction (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    print("================================================================")
    print(f"      UWF-ZeekData24 Baseline: {args.model.upper()}             ")
    print("================================================================")
    print(f"Data directory: {args.data_dir}")
    print(f"Selected model: {args.model}")
    print(f"Random seed:    {args.seed}\n")

    print("--- LOADING UWF-ZEEKDATA24 DATASET ---")
    data, actual_train_frac, actual_val_frac = load_uwf_stream(
        args.data_dir,
        max_benign_events=args.max_benign,
        max_attack_events_per_cat=args.max_attack,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )

    print("\n--- EXTRACTING 45-DIMENSIONAL PER-EVENT FEATURES ---")
    X = _build_features(data.msg, data.user, data.dst, data.node_features, data.y)
    y = data.y.numpy()
    types = data.types.numpy()

    n = len(y)
    n_train = int(n * actual_train_frac)
    n_val = int(n * actual_val_frac)
    train_end = n_train
    val_end = n_train + n_val

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    test_types = types[val_end:]

    print(f"Features matrix shape: {X.shape}")
    print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    if args.model == "isolation_forest":
        run_isolation_forest(
            X_train, y_train, X_val, y_val, X_test, y_test, test_types, seed=args.seed
        )
    elif args.model == "xgboost":
        run_xgboost(
            X_train, y_train, X_val, y_val, X_test, y_test, test_types, seed=args.seed
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
