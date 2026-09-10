"""2-node Vanilla TGN baseline (User -> Resource) — standard Dynamic Graph.

This baseline isolates the effect of the 5-NODE DECOMPOSITION compared to a standard
streaming Temporal Graph Network (TGN) on the 2-node graph (User -> Resource).

Characteristics:
  * Same temporal architecture (GRU memory, time encoding, temporal neighbor loader,
    MLP + structural cosine similarity head).
  * 2-node graph: every request is modelled exclusively as the direct User -> Resource
    edge, without the intermediate binding edges (Source -> Config -> Device -> User).
  * Same curriculum: InfoNCE ranking over K=5 random negative resources + positive
    anchor BCE + contextual BCE on Gaussian noise.
  * Strictly chronological streaming replay, calibrated at target_fpr (1%) on the
    benign validation slice.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import average_precision_score, roc_auc_score

from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg
from graphagate.eval_common import causal_precursor_factor
from graphagate.model.registry import NodeRegistry
from graphagate.model.tgn import ZTATemporalGraphNetwork, stable_hash
from graphagate.serve_tgn import precursor_boost, record_alert


def _binary_metrics(scores, labels, threshold):
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def tgn_2node_baseline(cfg: Optional[TGNConfig] = None) -> dict:
    """Train and evaluate the 2-node TGN baseline (User -> Resource)."""
    if cfg is None:
        cfg = TGNConfig()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    print("Generating synthetic streaming data (TGN params)...")
    stream = generate_streaming_data(**stream_kwargs_from_cfg(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2-entity node space (User and Resource)
    # user_nodes: [0, user_num), resource_nodes: [res_lo, res_lo + res_num)
    user = stream.user
    dst = stream.dst
    t = stream.t
    msg = stream.msg
    y = stream.y
    types = stream.types
    node_features = stream.node_features
    keys = stream.keys
    total_nodes = stream.num_nodes

    # Initialise the registry and the TGN model (standard capacity)
    capacity = total_nodes + cfg.capacity_headroom
    registry = NodeRegistry(capacity=capacity)
    registry.preregister(keys)

    model = ZTATemporalGraphNetwork(
        num_nodes=capacity,
        node_feat_dim=cfg.node_feat_dim,
        msg_dim=cfg.msg_dim,
        memory_dim=cfg.memory_dim,
        time_dim=cfg.time_dim,
        num_hops=cfg.num_hops,
        hash_buckets=cfg.hash_buckets,
        hash_dim=cfg.hash_dim,
        hist_feat_dim=cfg.hist_feat_dim,
        gnn_heads=cfg.gnn_heads,
        link_pred_hidden_layers=cfg.link_pred_hidden_layers,
    ).to(device)

    with torch.no_grad():
        model.node_feat[:total_nodes] = node_features.to(device)
        hashes = [stable_hash(registry._idx_to_key[i], cfg.hash_buckets) for i in range(total_nodes)]
        model.node_hash[:total_nodes] = torch.tensor(hashes, dtype=torch.long, device=device)

    model.init_neighbor_loader(cfg.neighbor_size, device)
    model.use_struct_head = True
    model.use_hash_identity = True
    model.use_hist_feats = True
    model.use_precursor = True
    model.precursor_half_life = cfg.precursor_half_life
    model.precursor_max_boost = cfg.precursor_max_boost

    # Chronological split: 70% train / 10% val / 20% test
    n = len(user)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)
    train_end, val_end = n_train, n_train + n_val

    # Negative sampling range over the resources
    res_lo = stream.res_lo
    res_num = stream.res_num
    K = cfg.infonce_k
    batch_size = cfg.batch_size
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)

    print("--- TRAINING START: 2-NODE VANILLA TGN (User -> Resource) ---")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        model.memory.reset_state()
        model.neighbor_loader.reset_state()
        model.last_contact.clear()
        model.pair_count.clear()
        model.src_count.clear()
        model.recent_alert.clear()

        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, train_end, batch_size):
            end = min(start + batch_size, train_end)
            bu, bd, bt, bmsg, by = user[start:end], dst[start:end], t[start:end], msg[start:end], y[start:end]

            # Benign traffic only for the one-class training
            benign_mask = by == 0
            if not benign_mask.any():
                continue

            pu = bu[benign_mask].to(device)
            pd = bd[benign_mask].to(device)
            pt = bt[benign_mask].to(device)
            pmsg = bmsg[benign_mask].to(device).float()
            P = int(pu.shape[0])

            # Random structural negatives over the resources
            neg_res = torch.randint(res_lo, res_lo + res_num, (P * K,), device=device)
            pu_rep = pu.repeat_interleave(K)
            pt_rep = pt.repeat_interleave(K)
            pmsg_rep = pmsg.repeat_interleave(K, dim=0)

            # Sample temporal neighbours for the involved nodes (user and dst only)
            query_nodes = torch.cat([pu, pd, neg_res]).unique()
            n_id, edge_index, hist_t, hist_msg = model.neighbor_loader(query_nodes)
            z = model.embed(n_id, edge_index, hist_t, hist_msg)
            assoc = model.neighbor_loader._assoc
            nf = model.node_feat[n_id]
            h_idx = model.node_hash[n_id]

            # Causal interaction-history features
            u_list, d_list, t_list = pu.tolist(), pd.tolist(), pt.tolist()
            hist_pos = model.compute_hist_feats(u_list, d_list, device)
            hist_neg = model.compute_hist_feats(pu_rep.tolist(), neg_res.tolist(), device)

            # Recency
            d_pair_pos = model.pair_delta_t(u_list, d_list, t_list, device)
            d_src_pos = model.src_delta_t(pu, pt, device)
            d_pair_neg = model.pair_delta_t(pu_rep.tolist(), neg_res.tolist(), pt_rep.tolist(), device)
            d_src_neg = model.src_delta_t(pu_rep, pt_rep, device)

            pos_logits = model.score(z, nf, h_idx, assoc[pu], assoc[pd], pmsg, d_pair_pos, d_src_pos, hist_pos)
            neg_logits = model.score(z, nf, h_idx, assoc[pu_rep], assoc[neg_res], pmsg_rep, d_pair_neg, d_src_neg, hist_neg).view(P, K)

            # Contextual negatives (Gaussian noise on the message)
            neg_msg_ctx = pmsg + torch.randn_like(pmsg) * 0.5
            ctx_logits = model.score(z, nf, h_idx, assoc[pu], assoc[pd], neg_msg_ctx, d_pair_pos, d_src_pos, hist_pos)

            # Loss: InfoNCE + positive BCE + contextual BCE
            target = torch.zeros(P, dtype=torch.long, device=device)
            loss = (
                F.cross_entropy(torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1), target)
                + F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
                + F.binary_cross_entropy_with_logits(ctx_logits, torch.zeros_like(ctx_logits))
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Predict-then-update: advance memory only for benign events
            model.memory.update_state(pu, pd, pt, pmsg)
            model.memory.detach()
            model.neighbor_loader.insert(pu, pd, pt, pmsg)

            for j in range(P):
                u_j, d_j, t_j = u_list[j], d_list[j], t_list[j]
                model.last_contact[(u_j, d_j)] = t_j
                model.pair_count[(u_j, d_j)] = model.pair_count.get((u_j, d_j), 0) + 1
                model.src_count[u_j] = model.src_count.get(u_j, 0) + 1

            epoch_loss += loss.item()
            num_batches += 1

        print(f"Epoch {epoch:02d} | Loss: {epoch_loss / max(1, num_batches):.4f}")

    # Sequential validation replay and threshold calibration
    model.eval()
    val_scores = []
    val_labels = y[train_end:val_end].numpy()

    for i in range(train_end, val_end):
        ui, di, ti, msgi, yi = int(user[i]), int(dst[i]), int(t[i]), msg[i], int(y[i])
        b_u = torch.tensor([ui], dtype=torch.long, device=device)
        b_d = torch.tensor([di], dtype=torch.long, device=device)
        b_t = torch.tensor([ti], dtype=torch.long, device=device)
        b_msg = msgi.unsqueeze(0).to(device).float()

        with torch.no_grad():
            q_nodes = torch.unique(torch.cat([b_u, b_d]))
            n_id, edge_index, hist_t, hist_msg = model.neighbor_loader(q_nodes)
            z = model.embed(n_id, edge_index, hist_t, hist_msg)
            assoc = model.neighbor_loader._assoc
            nf = model.node_feat[n_id]
            h_idx = model.node_hash[n_id]

            d_pair = model.pair_delta_t([ui], [di], [ti], device)
            d_src = model.src_delta_t(b_u, b_t, device)
            hist = model.compute_hist_feats([ui], [di], device)

            logit = model.score(z, nf, h_idx, assoc[b_u], assoc[b_d], b_msg, d_pair, d_src, hist)
            score = 1.0 - torch.sigmoid(logit).item()
            val_scores.append(score)

            # Commit gate on benign events during calibration
            if yi == 0:
                model.memory.update_state(b_u, b_d, b_t, b_msg)
                model.memory.detach()
                model.neighbor_loader.insert(b_u, b_d, b_t, b_msg)
                model.last_contact[(ui, di)] = ti
                model.pair_count[(ui, di)] = model.pair_count.get((ui, di), 0) + 1
                model.src_count[ui] = model.src_count.get(ui, 0) + 1

    val_scores_np = np.array(val_scores)
    benign_val = val_scores_np[val_labels == 0]
    threshold = float(np.quantile(benign_val, 1.0 - cfg.target_fpr))
    print(f"\nCalibrated Threshold @ FPR {cfg.target_fpr}: {threshold:.4f}")

    # Sequential replay on the test set
    test_scores = []
    test_labels = y[val_end:].numpy()
    test_types = types[val_end:].numpy()

    for i in range(val_end, n):
        ui, di, ti, msgi, yi = int(user[i]), int(dst[i]), int(t[i]), msg[i], int(y[i])
        b_u = torch.tensor([ui], dtype=torch.long, device=device)
        b_d = torch.tensor([di], dtype=torch.long, device=device)
        b_t = torch.tensor([ti], dtype=torch.long, device=device)
        b_msg = msgi.unsqueeze(0).to(device).float()

        with torch.no_grad():
            q_nodes = torch.unique(torch.cat([b_u, b_d]))
            n_id, edge_index, hist_t, hist_msg = model.neighbor_loader(q_nodes)
            z = model.embed(n_id, edge_index, hist_t, hist_msg)
            assoc = model.neighbor_loader._assoc
            nf = model.node_feat[n_id]
            h_idx = model.node_hash[n_id]

            d_pair = model.pair_delta_t([ui], [di], [ti], device)
            d_src = model.src_delta_t(b_u, b_t, device)
            hist = model.compute_hist_feats([ui], [di], device)

            logit = model.score(z, nf, h_idx, assoc[b_u], assoc[b_d], b_msg, d_pair, d_src, hist)
            raw_score = 1.0 - torch.sigmoid(logit).item()

            # Precursor boost on the user node
            score = min(1.0, raw_score * precursor_boost(model, ui, ti))
            test_scores.append(score)

            if msgi[1] > 0.5 or score >= threshold:
                record_alert(model, ui, ti)

            # Anti-poisoning gate: commit only if not anomalous
            if score < threshold:
                model.memory.update_state(b_u, b_d, b_t, b_msg)
                model.memory.detach()
                model.neighbor_loader.insert(b_u, b_d, b_t, b_msg)
                model.last_contact[(ui, di)] = ti
                model.pair_count[(ui, di)] = model.pair_count.get((ui, di), 0) + 1
                model.src_count[ui] = model.src_count.get(ui, 0) + 1

    test_scores_np = np.array(test_scores)
    auc = roc_auc_score(test_labels, test_scores_np)
    ap = average_precision_score(test_labels, test_scores_np)
    precision, recall = _binary_metrics(test_scores_np, test_labels, threshold)

    print(f"\nTest 2-Node TGN | AUC: {auc:.4f} | AP: {ap:.4f}")
    print(f"At threshold {threshold:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    # Per-type breakdown
    per_type = {}
    benign_mask = test_types == 0
    for type_id, name in ((1, "policy"), (2, "contextual"), (3, "lateral"),
                          (4, "cred-theft"), (5, "exfil"), (6, "benign-denied")):
        sel = benign_mask | (test_types == type_id)
        s_sel, l_sel = test_scores_np[sel], (test_types[sel] == type_id).astype(int)
        if l_sel.sum() == 0:
            continue
        t_auc = roc_auc_score(l_sel, s_sel)
        t_ap = average_precision_score(l_sel, s_sel)
        _, t_recall = _binary_metrics(s_sel, l_sel, threshold)
        per_type[name] = {
            "auc": float(t_auc),
            "ap": float(t_ap),
            "recall": float(t_recall),
            "n": int(l_sel.sum()),
        }
        print(f"  {name:11s} | n={int(l_sel.sum()):4d} | AUC: {t_auc:.4f} | AP: {t_ap:.4f} | Recall@thr: {t_recall:.4f}")

    return {
        "agg_auc": float(auc),
        "agg_ap": float(ap),
        "agg_precision": float(precision),
        "agg_recall": float(recall),
        "per_type": per_type,
    }


if __name__ == "__main__":
    tgn_2node_baseline()
