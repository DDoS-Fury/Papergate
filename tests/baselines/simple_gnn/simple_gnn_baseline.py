"""Baseline *non-temporal* GNN — TGN ablation for ZTA anomaly detection.

Why this file exists
--------------------
An FAIR ablation of the TGN that isolates the TEMPORAL component alone. The
baseline keeps the exact same **de-circularised** negative-sampling curriculum as
the TGN — structural negative with a random destination + Gaussian-noise
contextual negative, equal weights — and the same link-prediction idea; it
removes only the TEMPORAL machinery (per-node recurrent memory + temporal
neighbourhood), replaced by a GNN over a **static** graph aggregated from the
benign train interactions. The delta versus the TGN therefore measures how much
the temporal dynamics are worth, at equal curriculum.

NB (de-circularisation): the negatives do not use habituality/authorization
(`adj`/`auth_mask`) nor a hard-negative x10: a construction derived from habituality
would mirror the test's definition of lateral movement (authorized but non-habitual
access) and inflate recall circularly. The negative destination is drawn uniformly
over the resources, as in the TGN.

Protocol (identical to the TGN for 1:1 comparability, see tests/baselines/README.md):
same data/seed, same chronological 70/10/20 split, training on benign only,
threshold calibrated on the benign validation slice at ``target_fpr``, aggregate
metrics + per-type breakdown on the test segment.
"""

import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.nn import SAGEConv
from sklearn.metrics import average_precision_score, roc_auc_score

from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg
from graphagate.eval_common import causal_hist_features, causal_precursor_factor


class StaticGNN(nn.Module):
    """2-layer GraphSAGE encoder on the static graph + link predictor MLP.

    Deliberately *simple* compared to the TGN: no recurrent memory, no temporal
    neighbourhood, no cosine structural head, no hashed-identity embeddings.
    The node embeddings ``z`` are a function of the static features propagated
    on the graph aggregated from the benign train events.
    """

    def __init__(self, node_feat_dim, msg_dim, hidden=64, hist_dim=3):
        super().__init__()
        self.conv1 = SAGEConv(node_feat_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.dropout = nn.Dropout(0.1)

        # Link predictor: MLP over [z_src ‖ z_dst ‖ msg ‖ hist] -> 1 logit.
        # SIMPLE analogue of the TGN's LinkPredictor, without the concatenated
        # static features and without the cosine head: the goal is to keep the
        # model minimal to measure the contribution of the temporal part — but
        # it receives the SAME history features (causal counts) as the TGN, so
        # the comparison is fair.
        self.lin1 = nn.Linear(hidden * 2 + msg_dim + hist_dim, hidden)
        self.lin2 = nn.Linear(hidden, 1)

    def encode(self, x, edge_index):
        """Node embeddings via message passing on the static graph."""
        h = self.conv1(x, edge_index).relu()
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        return h

    def link_pred(self, z_src, z_dst, msg, hist):
        """Benignity logit for the src->dst edge carrying ``msg`` (+ history features)."""
        h = torch.cat([z_src, z_dst, msg, hist], dim=-1)
        h = self.lin1(h).relu()
        return self.lin2(h).squeeze(-1)


def _binary_metrics(scores, labels, threshold):
    """Precision / recall of ``score >= threshold`` against ``labels``.

    Exact replica of ``train_tgn._binary_metrics`` for comparability.
    """
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def _score_events(model, z, src, dst, msg, hist, device, precursor_fac=None):
    """Anomaly score for every event (s, d, msg, hist).

    Orientation "higher = more anomalous", consistent with ``serve_tgn.infer_score``
    (= ``1 - P(benign)``): we use the FIXED post-training embeddings ``z``
    (transductive on the train graph) and compute ``1 - sigmoid(link_pred)``, then
    apply the same multiplicative kill-chain precursor prior as the TGN.
    """
    model.eval()
    with torch.no_grad():
        z_src = z[src.to(device)]
        z_dst = z[dst.to(device)]
        logits = model.link_pred(z_src, z_dst, msg.to(device), hist.to(device))
        scores = 1.0 - torch.sigmoid(logits)
    out = scores.cpu().numpy().astype(np.float64)
    if precursor_fac is not None:
        out = out * precursor_fac
    return out


def run(cfg: TGNConfig = TGNConfig()):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    print("Generating synthetic streaming data (TGN params)...")
    stream = generate_streaming_data(**stream_kwargs_from_cfg(cfg))
    # Tabular actor = the DEVICE node (hardware id), the v2 analogue of the old
    # IP-keyed src; the access target stays the resource.
    src, dst, t, msg, y, types, node_features = (
        stream.device, stream.dst, stream.t, stream.msg, stream.y, stream.types,
        stream.node_features,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Causal interaction-history features (same tabular information as the TGN) +
    # kill-chain precursor prior: given to the baseline so the comparison isolates
    # the temporal dynamics alone.
    hist_all = torch.tensor(
        causal_hist_features(src.numpy(), dst.numpy(), y.numpy()), dtype=torch.float
    )
    precursor_fac = causal_precursor_factor(
        src.numpy(), t.numpy(), msg.numpy(), cfg.precursor_half_life, cfg.precursor_max_boost
    )

    # Chronological split identical to the TGN (the stream is already time-sorted).
    n = len(src)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)
    train_end, val_end = n_train, n_train + n_val

    # --- STATIC GRAPH from the benign train events only ----------------------
    # A single aggregated, immutable graph: the edges are the benign interactions
    # of the train segment (y==0), made UNDIRECTED by duplicating directions.
    # This aggregation "flattens" the chronology: it is exactly the information
    # the TGN keeps and this ablation drops.
    tr_src = src[:train_end]
    tr_dst = dst[:train_end]
    tr_msg = msg[:train_end]
    tr_y = y[:train_end]

    benign = tr_y == 0
    b_src = tr_src[benign]
    b_dst = tr_dst[benign]
    b_msg = tr_msg[benign]
    b_hist = hist_all[:train_end][benign]  # causal history features of the benign positives

    # Undirected edge_index [2, 2*E]: both directions.
    edge_index = torch.stack(
        [
            torch.cat([b_src, b_dst]),
            torch.cat([b_dst, b_src]),
        ],
        dim=0,
    ).to(device)

    x = node_features.to(device)  # [total_nodes, 16] static features

    model = StaticGNN(node_feat_dim=cfg.node_feat_dim, msg_dim=cfg.msg_dim, hidden=64).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Benign train edges on device for training (positives).
    p_src = b_src.to(device)
    p_dst = b_dst.to(device)
    p_msg = b_msg.to(device)
    p_hist = b_hist.to(device)
    num_pos = p_src.shape[0]
    bs = cfg.batch_size

    # Resource id range for sampling structural negatives: uses the STREAM range
    # (as the TGN), not cfg.total_nodes, which does not count guests and would
    # sample negatives onto source/config/user slots.
    num_res = stream.res_num
    res_lo = stream.res_lo

    print("--- UNSUPERVISED TRAINING START (non-temporal GNN) ---")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        # Shuffle the benign edges every epoch (mini-batch SGD).
        perm = torch.randperm(num_pos, device=device)
        total_loss = 0.0
        num_batches = max(num_pos // bs, 1)

        for i in range(num_batches):
            optimizer.zero_grad()
            idx = perm[i * bs : i * bs + bs]
            if idx.numel() == 0:
                continue

            bp_src = p_src[idx]
            bp_dst = p_dst[idx]
            bp_msg = p_msg[idx]
            bp_hist = p_hist[idx]

            # Recompute the embeddings on the fixed static graph: the graph does
            # not change, but z evolves as the GNN weights are updated.
            z = model.encode(x, edge_index)

            # POSITIVE: the real edge (src, dst, msg, hist) -> 1.
            pos_logit = model.link_pred(z[bp_src], z[bp_dst], bp_msg, bp_hist)

            # STRUCTURAL NEGATIVE: (src, random resource, msg) -> 0. Identical to
            # the de-circularised TGN: destination drawn UNIFORMLY over the
            # resources, WITHOUT using habituality/authorization (which would
            # mirror the test's definition of lateral). No hard-negative x10.
            neg_dst = torch.randint(0, num_res, (idx.numel(),), device=device) + res_lo
            collide = neg_dst == bp_dst
            if collide.any():
                neg_dst[collide] = (
                    torch.randint(0, num_res, (int(collide.sum()),), device=device) + res_lo
                )
            # For a random destination the (src, dst) pair has almost never been
            # seen -> the negative's history features are ~zero (as for the TGN).
            neg_hist = torch.zeros_like(bp_hist)
            neg_logit = model.link_pred(z[bp_src], z[neg_dst], bp_msg, neg_hist)

            # CONTEXTUAL NEGATIVE: additive Gaussian noise on the msg (a
            # DIFFERENT mechanism than the 0/1 randomisation of the test's
            # contextual anomalies).
            neg_msg = bp_msg + torch.randn_like(bp_msg) * 0.5
            ctx_logit = model.link_pred(z[bp_src], z[bp_dst], neg_msg, bp_hist)

            # Same loss as the de-circularised TGN: pos vs structural +
            # contextual, equal weights (no x10 on the hard-negative).
            loss = (
                criterion(pos_logit, torch.ones_like(pos_logit))
                + criterion(neg_logit, torch.zeros_like(neg_logit))
                + criterion(ctx_logit, torch.zeros_like(ctx_logit))
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch:02d} | Train Loss: {total_loss / num_batches:.4f}")

    # FIXED post-training embeddings (transductive on the train graph); used for
    # validation, threshold calibration and test — as per spec.
    model.eval()
    with torch.no_grad():
        z = model.encode(x, edge_index)

    # --- THRESHOLD CALIBRATION (on the benign validation stream) -------------
    print("\n--- THRESHOLD CALIBRATION (on the benign validation stream) ---")
    val_scores = _score_events(
        model, z, src[train_end:val_end], dst[train_end:val_end], msg[train_end:val_end],
        hist_all[train_end:val_end], device, precursor_fac[train_end:val_end],
    )
    val_labels = y[train_end:val_end].numpy()
    benign_val_scores = val_scores[val_labels == 0]
    if benign_val_scores.size == 0:
        raise RuntimeError("No benign events in the validation slice for calibration.")
    threshold = float(np.quantile(benign_val_scores, 1.0 - cfg.target_fpr))
    print(
        f"Benign val score: mean={benign_val_scores.mean():.4f} "
        f"p95={np.quantile(benign_val_scores, 0.95):.4f} | "
        f"threshold@FPR={cfg.target_fpr}: {threshold:.4f}"
    )

    # --- INFERENCE / ANOMALY DETECTION on the test ---------------------------
    print("\n--- INFERENCE / ANOMALY DETECTION PHASE START ---")
    test_scores = _score_events(
        model, z, src[val_end:], dst[val_end:], msg[val_end:],
        hist_all[val_end:], device, precursor_fac[val_end:],
    )
    test_labels = y[val_end:].numpy()
    test_types = types[val_end:].numpy()

    auc = roc_auc_score(test_labels, test_scores)
    ap = average_precision_score(test_labels, test_scores)
    precision, recall = _binary_metrics(test_scores, test_labels, threshold)
    print(f"Test Stream | AUC: {auc:.4f} | AP: {ap:.4f}")
    print(f"At threshold {threshold:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    # --- PER-ANOMALY-TYPE BREAKDOWN ------------------------------------------
    # Computed benign (type 0) vs that type, identical to the TGN: an aggregate
    # can hide a poorly handled class. Expected: policy/contextual good, lateral
    # weak (the temporal signal is missing).
    print("\n--- PER-TYPE METRICS ---")
    per_type = {}
    benign_mask = test_types == 0
    for type_id, name in ((1, "policy"), (2, "contextual"), (3, "lateral"),
                          (4, "cred-theft"), (5, "exfil"), (6, "benign-denied")):
        sel = benign_mask | (test_types == type_id)
        s_sel, l_sel = test_scores[sel], (test_types[sel] == type_id).astype(int)
        if l_sel.sum() == 0:
            continue
        t_auc = roc_auc_score(l_sel, s_sel)
        t_ap = average_precision_score(l_sel, s_sel)
        _, t_recall = _binary_metrics(s_sel, l_sel, threshold)
        per_type[name] = {"auc": float(t_auc), "ap": float(t_ap),
                          "recall": float(t_recall), "n": int(l_sel.sum())}
        print(
            f"  {name:10s} | n={int(l_sel.sum()):4d} | AUC: {t_auc:.4f} | "
            f"AP: {t_ap:.4f} | Recall@thr: {t_recall:.4f}"
        )

    # Machine-readable summary (consumed by tests/regen_report_tables.py). Recall here is
    # at the global @target_fpr threshold — the apples-to-apples Panel A (tab:baselines) metric.
    return {
        "agg_auc": float(auc), "agg_ap": float(ap),
        "agg_precision": float(precision), "agg_recall": float(recall),
        "per_type": per_type,
    }


def main():
    run()


if __name__ == "__main__":
    main()
