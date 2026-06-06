"""Isolation Forest anomaly-detection baseline for Graphagate.

Why this baseline exists:
  The streaming TGN scores each access event using its *relational + temporal*
  context (recurrent memory + temporal neighbourhood). This baseline deliberately
  throws all of that away: it treats every event as an i.i.d. static feature
  vector and runs a classic, non-relational anomaly detector (Isolation Forest)
  over it. The gap between this and the TGN therefore quantifies how much of the
  detection — and in particular the *policy* and *lateral-movement* recall — comes
  from graph structure / interaction history rather than from the raw features of
  a single event.

Protocol (identical to ``graphagate.train_tgn`` so the numbers sit side by side):
  1. Same synthetic ZTA stream (same ``TGNConfig`` hyper-parameters + seed).
  2. Same chronological train(70%) / val(10%) / test(20%) split — the stream is
     already time-ordered, so we never shuffle.
  3. Fit unsupervised on the *benign* training events only (``y_train == 0``).
  4. Calibrate the decision threshold on the *benign validation* slice at the
     configured ``target_fpr`` (99th percentile of benign scores).
  5. Report on the *test* slice: aggregate ROC-AUC / AP, precision/recall at the
     calibrated threshold, and a per-anomaly-type breakdown (policy / contextual /
     lateral) computed benign-vs-that-type.

Anomaly score orientation:
  We use ``-score_samples(X)`` so that *higher = more anomalous*, matching the
  TGN's ``1 - P(benign)`` (see ``graphagate.serve_tgn.infer_score``). Sklearn's
  ``score_samples`` returns the (signed) average path length where *higher = more
  normal*, so the negation gives the right orientation for calibration and all
  metrics below.

Run it inside the project's Docker image (torch is required only to import the
canonical data generator; it is not installed on the host). See README.md.
"""

import random

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data


def _binary_metrics(scores, labels, threshold):
    """Precision / recall of ``score >= threshold`` against ``labels``.

    Replicated verbatim from ``graphagate.train_tgn._binary_metrics`` so the
    precision/recall reported here are computed identically to the TGN's.
    """
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def _build_features(msg, src, dst, node_features):
    """Per-event static feature matrix for the Isolation Forest.

    Each event is described *without any history*: the 6-dim edge feature
    ``msg[i]`` concatenated with the 16-dim static attributes of its two
    endpoints (source IP and destination resource), giving a 38-dim vector.
    This is the most a non-relational detector can see about a single event.

    Tensors are torch tensors → converted to numpy here (sklearn input).
    """
    msg_np = msg.numpy()                       # [N, 6]
    nf_np = node_features.numpy()              # [total_nodes, 16]
    src_feat = nf_np[src.numpy()]              # [N, 16]  source (IP) attributes
    dst_feat = nf_np[dst.numpy()]              # [N, 16]  destination (resource) attributes
    return np.concatenate([msg_np, src_feat, dst_feat], axis=1)  # [N, 38]


def isolation_forest_baseline(cfg: TGNConfig = TGNConfig()):
    # Seed everything up front for reproducibility (the generator re-seeds numpy
    # and stdlib ``random`` internally too, but we seed here so the IsolationForest
    # and any incidental randomness are deterministic as well).
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    print("Generating synthetic streaming data (same params as TGN)...")
    src, dst, t, msg, y, types, node_features, resource_uris, _ = generate_streaming_data(
        num_users=cfg.num_users,
        num_ips=cfg.num_ips,
        num_resources=cfg.num_resources,
        num_events=cfg.num_events,
        seed=cfg.seed,
    )

    # Static per-event features (no graph structure, no temporal context).
    X = _build_features(msg, src, dst, node_features)
    y_np = y.numpy()
    types_np = types.numpy()

    # Chronological split (the stream is already time-ordered → no shuffling).
    n = len(src)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)
    train_end, val_end = n_train, n_train + n_val

    X_train, y_train = X[:train_end], y_np[:train_end]
    X_val, y_val = X[train_end:val_end], y_np[train_end:val_end]
    X_test, y_test = X[val_end:], y_np[val_end:]
    test_types = types_np[val_end:]

    # --- FIT (unsupervised, benign train events only) ------------------------
    print("--- INIZIO ADDESTRAMENTO UNSUPERVISED (Isolation Forest) ---")
    X_train_benign = X_train[y_train == 0]
    print(f"Train benigni: {X_train_benign.shape[0]} / {X_train.shape[0]} eventi")
    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=cfg.seed,
    )
    model.fit(X_train_benign)

    # Anomaly score: higher = more anomalous (matches the TGN's 1 - P(benign)).
    def anomaly_score(features):
        return -model.score_samples(features)

    # --- THRESHOLD CALIBRATION (held-out benign slice) -----------------------
    print("\n--- CALIBRAZIONE SOGLIA (su flusso di validazione benigno) ---")
    val_scores = anomaly_score(X_val)
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
    test_scores = anomaly_score(X_test)

    auc = roc_auc_score(y_test, test_scores)
    ap = average_precision_score(y_test, test_scores)
    precision, recall = _binary_metrics(test_scores, y_test, threshold)
    print(f"Test Stream | AUC: {auc:.4f} | AP: {ap:.4f}")
    print(f"At threshold {threshold:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    # --- PER-ANOMALY-TYPE BREAKDOWN ------------------------------------------
    # AUC/AP per type computed benign (type 0) vs that type only, so the aggregate
    # cannot mask a class the detector handles poorly. As for the TGN: contextual
    # anomalies are near-trivial (separable on edge features), while policy and
    # lateral anomalies share benign edge features and are the genuinely hard cases
    # for a non-relational detector — that is exactly the contrast this baseline
    # is meant to expose.
    print("\n--- METRICHE PER TIPO DI ANOMALIA ---")
    benign = test_types == 0
    for type_id, name in ((1, "policy    "), (2, "contextual"), (3, "lateral   ")):
        sel = benign | (test_types == type_id)
        s_sel, l_sel = test_scores[sel], (test_types[sel] == type_id).astype(int)
        if l_sel.sum() == 0:
            continue
        t_auc = roc_auc_score(l_sel, s_sel)
        t_ap = average_precision_score(l_sel, s_sel)
        _, t_recall = _binary_metrics(s_sel, l_sel, threshold)
        print(f"  {name} | n={int(l_sel.sum()):4d} | AUC: {t_auc:.4f} | "
              f"AP: {t_ap:.4f} | Recall@thr: {t_recall:.4f}")


def main():
    isolation_forest_baseline()


if __name__ == "__main__":
    main()
