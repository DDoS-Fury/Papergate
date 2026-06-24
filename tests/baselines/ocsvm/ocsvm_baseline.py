"""One-Class SVM anomaly-detection baseline for Graphagate.

Why this baseline exists:
  A classic, non-relational unsupervised detector that — like the Isolation Forest
  baseline — sees each access event as an i.i.d. static feature vector with no graph
  structure or temporal history. One-Class SVM (RBF kernel) is the other canonical
  "novelty detection" reference reviewers expect alongside Isolation Forest, so the
  gap between it and the TGN again quantifies how much detection comes from
  structure / interaction history rather than the raw features of a single event.

Protocol (identical to ``graphagate.train_tgn`` so the numbers sit side by side):
  same synthetic stream + seed, same chronological 70/10/20 split, fit on benign
  train events only, threshold calibrated on the benign validation slice at
  ``target_fpr``, report aggregate ROC-AUC / AP + per-type breakdown on test.

Scalability note:
  RBF One-Class SVM is O(n^2) in the number of training samples, which is
  impractical at ~35k benign train events. We therefore fit on a uniform random
  subsample (``OCSVM_FIT_SAMPLES``) of the benign training events — standard practice
  for kernel OC-SVM — while still scoring the *full* val/test slices.

Anomaly score orientation:
  We use ``-score_samples(X)`` so that higher = more anomalous, matching the TGN's
  ``1 - P(benign)`` (see ``graphagate.serve_tgn.infer_score``).

Run inside the project's Docker image (torch is required only to import the
canonical data generator; it is not installed on the host). See README.md.
"""

import random

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import average_precision_score, roc_auc_score

from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data
from graphagate.eval_common import causal_hist_features, causal_precursor_factor

# Subsample size for the (O(n^2)) kernel fit.
OCSVM_FIT_SAMPLES = 5000


def _binary_metrics(scores, labels, threshold):
    """Precision / recall of ``score >= threshold`` (verbatim from train_tgn)."""
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def _build_features(msg, src, dst, node_features, y):
    """Per-event features: edge msg (6) ⊕ src attrs (16) ⊕ dst attrs (16) ⊕ hist (3) = 41.

    Identical to the Isolation Forest baseline (incl. the causal interaction-history
    counts) so the two non-relational detectors see exactly the same information, and the
    comparison to the TGN is fair (history is given, not the temporal graph).
    """
    msg_np = msg.numpy()
    nf_np = node_features.numpy()
    src_feat = nf_np[src.numpy()]
    dst_feat = nf_np[dst.numpy()]
    hist = causal_hist_features(src.numpy(), dst.numpy(), y.numpy())  # [N, 3]
    return np.concatenate([msg_np, src_feat, dst_feat, hist], axis=1)  # [N, 41]


def ocsvm_baseline(cfg: TGNConfig = TGNConfig()):
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    print("Generating synthetic streaming data (same params as TGN)...")
    stream = generate_streaming_data(
        num_users=cfg.num_users,
        num_devices=cfg.num_devices,
        num_sources=cfg.num_sources,
        num_resources=cfg.num_resources,
        num_events=cfg.num_events,
        num_wipe_slots=cfg.num_wipe_slots,
        num_theft_slots=cfg.num_theft_slots,
        benign_explore_prob=cfg.benign_explore_prob,
        p_roam=cfg.p_roam,
        p_shared_device=cfg.p_shared_device,
        p_cookie_wipe=cfg.p_cookie_wipe,
        p_cred_theft=cfg.p_cred_theft,
        seed=cfg.seed,
    )
    # Tabular actor = the DEVICE node (hardware id), the v2 analogue of the old
    # IP-keyed src; the access target stays the resource.
    src, dst, t, msg, y, types, node_features = (
        stream.device, stream.dst, stream.t, stream.msg, stream.y, stream.types,
        stream.node_features,
    )

    X = _build_features(msg, src, dst, node_features, y)
    y_np = y.numpy()
    types_np = types.numpy()

    # Kill-chain precursor prior (same multiplicative factor + params as the TGN).
    precursor_fac = causal_precursor_factor(
        src.numpy(), t.numpy(), msg.numpy(), cfg.precursor_half_life, cfg.precursor_max_boost
    )

    n = len(src)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)
    train_end, val_end = n_train, n_train + n_val

    X_train, y_train = X[:train_end], y_np[:train_end]
    X_val, y_val = X[train_end:val_end], y_np[train_end:val_end]
    X_test, y_test = X[val_end:], y_np[val_end:]
    test_types = types_np[val_end:]

    # --- FIT (unsupervised, benign train events only; subsampled for the kernel) --
    print("--- INIZIO ADDESTRAMENTO UNSUPERVISED (One-Class SVM) ---")
    X_train_benign = X_train[y_train == 0]
    if X_train_benign.shape[0] > OCSVM_FIT_SAMPLES:
        sel = np.random.choice(X_train_benign.shape[0], OCSVM_FIT_SAMPLES, replace=False)
        X_fit = X_train_benign[sel]
    else:
        X_fit = X_train_benign
    print(f"Train benigni: fit su {X_fit.shape[0]} / {X_train_benign.shape[0]} eventi benigni")

    # Standardise (RBF SVM is scale-sensitive); fit the scaler on the same subsample.
    scaler = StandardScaler().fit(X_fit)
    model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    model.fit(scaler.transform(X_fit))

    def anomaly_score(features):
        return -model.score_samples(scaler.transform(features))

    # --- THRESHOLD CALIBRATION (held-out benign slice) -----------------------
    print("\n--- CALIBRAZIONE SOGLIA (su flusso di validazione benigno) ---")
    val_scores = anomaly_score(X_val) * precursor_fac[train_end:val_end]
    benign_val_scores = val_scores[y_val == 0]
    if benign_val_scores.size == 0:
        raise RuntimeError("No benign events in the validation slice for calibration.")
    threshold = float(np.quantile(benign_val_scores, 1.0 - cfg.target_fpr))
    print(
        f"Benign val score: mean={benign_val_scores.mean():.4f} "
        f"p95={np.quantile(benign_val_scores, 0.95):.4f} | "
        f"threshold@FPR={cfg.target_fpr}: {threshold:.4f}"
    )

    # --- TEST EVALUATION -----------------------------------------------------
    print("\n--- INIZIO FASE DI INFERENZA / ANOMALY DETECTION ---")
    test_scores = anomaly_score(X_test) * precursor_fac[val_end:]

    auc = roc_auc_score(y_test, test_scores)
    ap = average_precision_score(y_test, test_scores)
    precision, recall = _binary_metrics(test_scores, y_test, threshold)
    print(f"Test Stream | AUC: {auc:.4f} | AP: {ap:.4f}")
    print(f"At threshold {threshold:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    # --- PER-ANOMALY-TYPE BREAKDOWN ------------------------------------------
    print("\n--- METRICHE PER TIPO DI ANOMALIA ---")
    per_type = {}
    benign = test_types == 0
    for type_id, name in ((1, "policy"), (2, "contextual"), (3, "lateral")):
        sel = benign | (test_types == type_id)
        s_sel, l_sel = test_scores[sel], (test_types[sel] == type_id).astype(int)
        if l_sel.sum() == 0:
            continue
        t_auc = roc_auc_score(l_sel, s_sel)
        t_ap = average_precision_score(l_sel, s_sel)
        _, t_recall = _binary_metrics(s_sel, l_sel, threshold)
        per_type[name] = {"auc": float(t_auc), "ap": float(t_ap),
                          "recall": float(t_recall), "n": int(l_sel.sum())}
        print(f"  {name:10s} | n={int(l_sel.sum()):4d} | AUC: {t_auc:.4f} | "
              f"AP: {t_ap:.4f} | Recall@thr: {t_recall:.4f}")

    # Machine-readable summary (consumed by tests/regen_report_tables.py). Recall here is
    # at the global @target_fpr threshold — the apples-to-apples Panel A (tab:baselines) metric.
    return {
        "agg_auc": float(auc), "agg_ap": float(ap),
        "agg_precision": float(precision), "agg_recall": float(recall),
        "per_type": per_type,
    }


def main():
    ocsvm_baseline()


if __name__ == "__main__":
    main()
