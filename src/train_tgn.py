"""Train and evaluate the streaming Temporal Graph Network (v2).

Pipeline:
  1. Generate a chronologically ordered synthetic ZTA access stream.
  2. Split it by time into train (70%) / val (10%) / test (20%).
  3. Train unsupervised on benign traffic with structural + contextual negatives;
     memory is updated with benign events only.
  4. Calibrate the anomaly threshold on a held-out benign slice at ``target_fpr``.
  5. Evaluate on the test stream **event-by-event**, through the exact serving code
     path (``graphagate.serve_tgn``), so the reported AUC/AP reflects deployment.
  6. Persist the deployable artifact (weights + memory + registry + threshold).
"""

import numpy as np
import torch
from torch.optim import Adam
from sklearn.metrics import average_precision_score, roc_auc_score

from graphagate.config import TGNConfig, TGN_CHECKPOINT_PATH, TGN_STATS_PATH
from graphagate.data.stream_synthetic import generate_streaming_data
from graphagate.model.registry import NodeRegistry
from graphagate.model.tgn import ZTATemporalGraphNetwork
from graphagate.serve_tgn import infer_score, save_model, update_memory


def _replay(model, src, dst, t, msg, y, device, *, threshold=None, gate_by_label=False):
    """Per-event streaming replay matching the serving path.

    Memory update gating:
      - ``gate_by_label=True``  -> update on ground-truth benign (calibration);
      - ``gate_by_label=False`` -> update on predicted benign (``score < threshold``),
        i.e. the realistic, label-free serving behaviour.

    Returns ``(scores, labels)`` as numpy arrays.
    """
    model.eval()
    scores = np.empty(src.shape[0], dtype=np.float64)
    labels = np.empty(src.shape[0], dtype=np.int64)
    src_l, dst_l, t_l, y_l = src.tolist(), dst.tolist(), t.tolist(), y.tolist()

    for i in range(len(src_l)):
        s, d, tv, lab = src_l[i], dst_l[i], t_l[i], y_l[i]
        msg_vec = msg[i]
        score = infer_score(model, s, d, tv, msg_vec, device)
        scores[i] = score
        labels[i] = lab

        do_update = (lab == 0) if gate_by_label else (score < threshold)
        if do_update:
            update_memory(model, s, d, tv, msg_vec, device)

    return scores, labels


def train_tgn(cfg: TGNConfig = TGNConfig()):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("Generating streaming data...")
    src, dst, t, msg, y, _node_features = generate_streaming_data(
        num_users=cfg.num_users,
        num_ips=cfg.num_ips,
        num_resources=cfg.num_resources,
        num_events=cfg.num_events,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Entity registry: known synthetic ids get an identity mapping; spare capacity
    # is reserved for entities first seen at inference time.
    registry = NodeRegistry(capacity=cfg.capacity)
    registry.preregister(range(cfg.total_nodes))

    model = ZTATemporalGraphNetwork(
        num_nodes=cfg.capacity,
        node_feat_dim=cfg.node_feat_dim,
        msg_dim=cfg.msg_dim,
        memory_dim=cfg.memory_dim,
        time_dim=cfg.time_dim,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Chronological split (the stream is already time-ordered).
    n = len(src)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)
    train_end, val_end = n_train, n_train + n_val
    bs = cfg.batch_size

    print("--- INIZIO ADDESTRAMENTO UNSUPERVISED ---")
    for epoch in range(1, cfg.epochs + 1):
        model.memory.reset_state()  # restart the recurrent memory each epoch
        model.train()

        total_loss = 0.0
        num_train_batches = train_end // bs

        for i in range(num_train_batches):
            optimizer.zero_grad()
            start_idx, end_idx = i * bs, i * bs + bs

            b_src = src[start_idx:end_idx].to(device)
            b_dst = dst[start_idx:end_idx].to(device)
            b_t = t[start_idx:end_idx].to(device)
            b_msg = msg[start_idx:end_idx].to(device)
            b_y = y[start_idx:end_idx].to(device)

            # The baseline is learned from benign traffic only.
            benign_mask = b_y == 0
            if not benign_mask.any():
                continue

            p_src = b_src[benign_mask]
            p_dst = b_dst[benign_mask]
            p_t = b_t[benign_mask]
            p_msg = b_msg[benign_mask]

            # --- POSITIVE EDGES (healthy behaviour) ---
            n_id, inv = torch.unique(torch.cat([p_src, p_dst]), return_inverse=True)
            pos_edge_index = torch.stack([inv[: len(p_src)], inv[len(p_src):]], dim=0)
            pos_out = model(n_id, pos_edge_index, p_t, p_msg).squeeze(-1)

            # --- STRUCTURAL NEGATIVES (lateral movement / enumeration) ---
            neg_dst = torch.randint(cfg.num_users, cfg.total_nodes, (len(p_src),), device=device)
            n_id_neg, inv_neg = torch.unique(torch.cat([p_src, neg_dst]), return_inverse=True)
            neg_edge_index = torch.stack([inv_neg[: len(p_src)], inv_neg[len(p_src):]], dim=0)
            neg_out_struct = model(n_id_neg, neg_edge_index, p_t, p_msg).squeeze(-1)

            # --- CONTEXTUAL NEGATIVES (out-of-distribution feature perturbation) ---
            neg_msg = p_msg.clone()
            noise_mask = torch.rand_like(neg_msg) < 0.20
            neg_msg[noise_mask] = 1.0 - neg_msg[noise_mask]
            neg_out_ctx = model(n_id, pos_edge_index, p_t, neg_msg).squeeze(-1)

            # --- UNSUPERVISED LOSS: positive -> 1, negatives -> 0 ---
            loss = (
                criterion(pos_out, torch.ones_like(pos_out))
                + criterion(neg_out_struct, torch.zeros_like(neg_out_struct))
                + criterion(neg_out_ctx, torch.zeros_like(neg_out_ctx))
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Update memory with the benign traffic only.
            model.memory.update_state(p_src, p_dst, p_t, p_msg)
            model.memory.detach()

        print(f"Epoch {epoch:02d} | Train Loss: {total_loss / max(num_train_batches, 1):.4f}")

    # --- THRESHOLD CALIBRATION (held-out benign slice) -----------------------
    print("\n--- CALIBRAZIONE SOGLIA (su flusso di validazione benigno) ---")
    val_scores, val_labels = _replay(
        model, src[train_end:val_end], dst[train_end:val_end], t[train_end:val_end],
        msg[train_end:val_end], y[train_end:val_end], device, gate_by_label=True,
    )
    benign_val_scores = val_scores[val_labels == 0]
    if benign_val_scores.size == 0:
        raise RuntimeError("No benign events in the validation slice for calibration.")
    threshold = float(np.quantile(benign_val_scores, 1.0 - cfg.target_fpr))
    print(
        f"Benign val score: mean={benign_val_scores.mean():.4f} "
        f"p95={np.quantile(benign_val_scores, 0.95):.4f} | "
        f"threshold@FPR={cfg.target_fpr}: {threshold:.4f}"
    )

    # --- STREAMING EVALUATION (event-by-event, predicted-benign gating) ------
    print("\n--- INIZIO FASE DI INFERENZA / ANOMALY DETECTION (per-evento) ---")
    test_scores, test_labels = _replay(
        model, src[val_end:], dst[val_end:], t[val_end:], msg[val_end:], y[val_end:],
        device, threshold=threshold, gate_by_label=False,
    )

    auc = roc_auc_score(test_labels, test_scores)
    ap = average_precision_score(test_labels, test_scores)
    preds = (test_scores >= threshold).astype(int)
    tp = int(((preds == 1) & (test_labels == 1)).sum())
    fp = int(((preds == 1) & (test_labels == 0)).sum())
    fn = int(((preds == 0) & (test_labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    print(f"Test Stream | AUC: {auc:.4f} | AP: {ap:.4f}")
    print(f"At threshold {threshold:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    # --- PERSIST DEPLOYABLE ARTIFACT -----------------------------------------
    hp = {
        "capacity": cfg.capacity,
        "node_feat_dim": cfg.node_feat_dim,
        "msg_dim": cfg.msg_dim,
        "memory_dim": cfg.memory_dim,
        "time_dim": cfg.time_dim,
        "target_fpr": cfg.target_fpr,
    }
    save_model(model, registry, threshold, hp, TGN_CHECKPOINT_PATH, TGN_STATS_PATH)
    print(f"\nSaved checkpoint -> {TGN_CHECKPOINT_PATH}")
    print(f"Saved stats      -> {TGN_STATS_PATH}")


if __name__ == "__main__":
    train_tgn()
