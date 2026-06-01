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

import random

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


def _binary_metrics(scores, labels, threshold):
    """Precision / recall (+ raw counts) of ``score >= threshold`` against ``labels``."""
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def _rule_baseline(test_msg):
    """Trivial detector: flag if any Zero-Trust edge signal fires.

    Columns are ``[ja3, snort, s1, s2, s3, method]``: an event is suspicious when the
    TLS trust is broken (``ja3==0``), Snort alerts (``snort==1``) or any sensor fires.
    This catches *contextual* anomalies but is blind to *policy* violations (which
    share benign edge features) — it quantifies how much the TGN adds beyond rules.
    """
    return (
        (test_msg[:, 0] == 0.0)
        | (test_msg[:, 1] == 1.0)
        | (test_msg[:, 2] == 1.0)
        | (test_msg[:, 3] == 1.0)
        | (test_msg[:, 4] == 1.0)
    ).astype(int)


def train_tgn(cfg: TGNConfig = TGNConfig()):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    print("Generating streaming data...")
    src, dst, t, msg, y, types, node_features, resource_uris = generate_streaming_data(
        num_users=cfg.num_users,
        num_ips=cfg.num_ips,
        num_resources=cfg.num_resources,
        num_events=cfg.num_events,
        seed=cfg.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Entity registry: users and IPs are still mapped by their int ids, but resources
    # are registered using their actual string URIs so the Orchestrator can send strings natively.
    registry = NodeRegistry(capacity=cfg.capacity)
    registry.preregister(range(cfg.num_users + cfg.num_ips))
    res_start = cfg.num_users + cfg.num_ips
    for i, uri in enumerate(resource_uris):
        slot = res_start + i
        registry._key_to_idx[uri] = slot
        registry._idx_to_key[slot] = uri
        registry._next_idx = max(registry._next_idx, slot + 1)

    model = ZTATemporalGraphNetwork(
        num_nodes=cfg.capacity,
        node_feat_dim=cfg.node_feat_dim,
        msg_dim=cfg.msg_dim,
        memory_dim=cfg.memory_dim,
        time_dim=cfg.time_dim,
        num_hops=cfg.num_hops,
    ).to(device)

    # Load the static node attributes (role / clearance / tier) into the model's
    # buffer for the preregistered training entities. Slots reserved for entities
    # first seen at serving time stay zero until those entities supply their features.
    with torch.no_grad():
        model.node_feat[: cfg.total_nodes] = node_features.to(device)

    # Bounded temporal neighbour loader (built on the model device after .to).
    model.init_neighbor_loader(cfg.neighbor_size, device)

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
        model.neighbor_loader.reset_state()  # ...and the temporal neighbourhood
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

            # Structural negatives: src paired with an unrelated node anywhere in the
            # graph (coarse "wrong target type" signal).
            neg_dst = torch.randint(cfg.num_users, cfg.total_nodes, (len(p_src),), device=device)
            # HARD structural negatives: src paired with a *non-habitual* resource — one
            # it is NOT currently a neighbour of. This is exactly the lateral-movement
            # pattern (authorised-but-unusual access) and forces the model to learn each
            # entity's habitual resource set, the fine-grained signal a random-resource
            # negative would wash out (it often collides with a habitual resource).
            res_lo = cfg.total_nodes - cfg.num_resources
            num_res = cfg.num_resources
            nl = model.neighbor_loader
            nbr = nl.neighbors[p_src]                 # [B, K] global neighbour ids
            res_local = nbr - res_lo                  # -> [0, num_res); non-resources out of range
            is_res = (nl.e_id[p_src] >= 0) & (res_local >= 0) & (res_local < num_res)
            occ = torch.zeros(len(p_src), num_res, dtype=torch.bool, device=device)
            rows = torch.arange(len(p_src), device=device).unsqueeze(1).expand_as(res_local)
            occ[rows[is_res], res_local[is_res]] = True  # habitual resources so far
            # Random pick among the non-habitual resources (K < num_res => always some).
            pick = torch.rand(len(p_src), num_res, device=device)
            pick[occ] = -1.0
            hard_dst = pick.argmax(dim=1) + res_lo

            # Expand every involved node to its stored temporal neighbourhood and embed
            # once; the heads below differ only in which endpoints / message they score,
            # sharing the same history-conditioned embeddings.
            nodes = torch.cat([p_src, p_dst, neg_dst, hard_dst]).unique()
            n_id, edge_index, hist_t, hist_msg = model.neighbor_loader(nodes)
            z = model.embed(n_id, edge_index, hist_t, hist_msg)
            assoc = model.neighbor_loader._assoc
            nf = model.node_feat[n_id]
            s_loc, d_loc = assoc[p_src], assoc[p_dst]
            nd_loc, hd_loc = assoc[neg_dst], assoc[hard_dst]

            # --- POSITIVE EDGES (healthy behaviour) ---
            pos_out = model.score(z, nf, s_loc, d_loc, p_msg)

            # --- STRUCTURAL NEGATIVES (src paired with an unrelated target) ---
            neg_out_struct = model.score(z, nf, s_loc, nd_loc, p_msg)

            # --- HARD STRUCTURAL NEGATIVES (src paired with a non-habitual resource) ---
            neg_out_hard = model.score(z, nf, s_loc, hd_loc, p_msg)

            # --- CONTEXTUAL NEGATIVES (out-of-distribution feature perturbation) ---
            neg_msg = p_msg.clone()
            noise_mask = torch.rand_like(neg_msg) < 0.20
            neg_msg[noise_mask] = 1.0 - neg_msg[noise_mask]
            neg_out_ctx = model.score(z, nf, s_loc, d_loc, neg_msg)

            # --- UNSUPERVISED LOSS: positive -> 1, negatives -> 0 ---
            loss = (
                criterion(pos_out, torch.ones_like(pos_out))
                + criterion(neg_out_struct, torch.zeros_like(neg_out_struct))
                + 5.0 * criterion(neg_out_hard, torch.zeros_like(neg_out_hard))
                + criterion(neg_out_ctx, torch.zeros_like(neg_out_ctx))
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Predict-then-update: commit benign traffic to memory and neighbour store.
            model.memory.update_state(p_src, p_dst, p_t, p_msg)
            model.memory.detach()
            model.neighbor_loader.insert(p_src, p_dst, p_t, p_msg)

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

    test_types = types[val_end:].numpy()
    test_msg = msg[val_end:].numpy()

    auc = roc_auc_score(test_labels, test_scores)
    ap = average_precision_score(test_labels, test_scores)
    precision, recall = _binary_metrics(test_scores, test_labels, threshold)
    print(f"Test Stream | AUC: {auc:.4f} | AP: {ap:.4f}")
    print(f"At threshold {threshold:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    # --- PER-ANOMALY-TYPE BREAKDOWN ------------------------------------------
    # AUC/AP per type are computed against benign only (type 0) vs that type, so an
    # aggregate metric cannot hide a class the model handles poorly. Contextual
    # anomalies are near-trivial (separable on edge features); policy anomalies share
    # benign edge features and are the genuinely hard case.
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

    # --- RULE-BASED BASELINE (value-add reference) ---------------------------
    base_pred = _rule_baseline(test_msg)
    b_tp = int(((base_pred == 1) & (test_labels == 1)).sum())
    b_fp = int(((base_pred == 1) & (test_labels == 0)).sum())
    b_fn = int(((base_pred == 0) & (test_labels == 1)).sum())
    b_precision = b_tp / (b_tp + b_fp) if (b_tp + b_fp) else 0.0
    b_recall = b_tp / (b_tp + b_fn) if (b_tp + b_fn) else 0.0
    base_policy_recall = (
        ((base_pred == 1) & (test_types == 1)).sum() / max((test_types == 1).sum(), 1)
    )
    base_lateral_recall = (
        ((base_pred == 1) & (test_types == 3)).sum() / max((test_types == 3).sum(), 1)
    )
    print("\n--- BASELINE A REGOLE (signal-only) ---")
    print(f"  Precision: {b_precision:.4f} | Recall: {b_recall:.4f} | "
          f"Recall policy: {base_policy_recall:.4f} | Recall lateral: {base_lateral_recall:.4f}")

    # --- PERSIST DEPLOYABLE ARTIFACT -----------------------------------------
    hp = {
        "capacity": cfg.capacity,
        "node_feat_dim": cfg.node_feat_dim,
        "msg_dim": cfg.msg_dim,
        "memory_dim": cfg.memory_dim,
        "time_dim": cfg.time_dim,
        "num_hops": cfg.num_hops,
        "neighbor_size": cfg.neighbor_size,
        "target_fpr": cfg.target_fpr,
    }
    save_model(model, registry, threshold, hp, TGN_CHECKPOINT_PATH, TGN_STATS_PATH)
    print(f"\nSaved checkpoint -> {TGN_CHECKPOINT_PATH}")
    print(f"Saved stats      -> {TGN_STATS_PATH}")


if __name__ == "__main__":
    train_tgn()
