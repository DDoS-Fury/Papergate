"""XGBoost supervised baseline for Graphagate.

A classic, non-relational supervised classifier (XGBoost) that uses the same
per-event static features and interaction history as the Isolation Forest.
It is trained on the entire training set (both benign and anomalous events)
to serve as a strong supervised baseline.
"""

import random
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import average_precision_score, roc_auc_score

from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data
from graphagate.eval_common import causal_hist_features, causal_precursor_factor

def _binary_metrics(scores, labels, threshold):
    """Precision / recall of ``score >= threshold`` against ``labels``."""
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall

def _build_features(msg, src, dst, node_features, y):
    """Per-event feature matrix for XGBoost."""
    msg_np = msg.numpy()                       # [N, 6]
    nf_np = node_features.numpy()              # [total_nodes, 16]
    src_feat = nf_np[src.numpy()]              # [N, 16]
    dst_feat = nf_np[dst.numpy()]              # [N, 16]
    hist = causal_hist_features(src.numpy(), dst.numpy(), y.numpy())  # [N, 3]
    return np.concatenate([msg_np, src_feat, dst_feat, hist], axis=1)  # [N, 41]

def xgboost_baseline(cfg: TGNConfig = TGNConfig()):
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
    src, dst, t, msg, y, types, node_features = (
        stream.device, stream.dst, stream.t, stream.msg, stream.y, stream.types,
        stream.node_features,
    )

    X = _build_features(msg, src, dst, node_features, y)
    y_np = y.numpy()
    types_np = types.numpy()

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

    # --- FIT (supervised, all train events) ------------------------
    print("--- INIZIO ADDESTRAMENTO E TUNING SUPERVISIONATO (XGBoost) ---")
    print(f"Train eventi totali: {X_train.shape[0]} (di cui {y_train.sum()} anomalie)")
    
    base_model = XGBClassifier(
        tree_method="hist",
        random_state=cfg.seed,
        n_jobs=18,
        eval_metric="auc"
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0]
    }

    # Impostiamo n_jobs=1 per la ricerca in sé per non sovraccaricare la CPU,
    # dal momento che XGBoost userà internamente 18 thread per il training.
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_grid,
        n_iter=10,
        scoring="roc_auc",
        cv=3,
        random_state=cfg.seed,
        n_jobs=1,
        verbose=1
    )
    
    print("Avvio RandomizedSearchCV su 3 fold...")
    search.fit(X_train, y_train)
    
    print(f"\nMigliori parametri trovati: {search.best_params_}")
    print(f"Miglior CV ROC AUC: {search.best_score_:.4f}")
    
    model = search.best_estimator_

    # Anomaly score: higher = more anomalous (matches the TGN's 1 - P(benign)).
    def anomaly_score(features):
        return model.predict_proba(features)[:, 1]

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
    xgboost_baseline()

if __name__ == "__main__":
    main()
