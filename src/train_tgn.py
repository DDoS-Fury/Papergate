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
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import average_precision_score, roc_auc_score

from graphagate.config import TGNConfig, TGN_CHECKPOINT_PATH, TGN_STATS_PATH
from graphagate.data.stream_synthetic import generate_streaming_data
from graphagate.model.registry import NodeRegistry
from graphagate.model.tgn import ZTATemporalGraphNetwork, stable_hash
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

        snort_alert = msg_vec[1] > 0.5
        is_anomaly = (score >= threshold) if not gate_by_label else (lab == 1)
        if is_anomaly or snort_alert:
            model.node_feat[s, 14] = max(0.0, model.node_feat[s, 14].item() - 0.5)
        else:
            model.node_feat[s, 14] = min(1.0, model.node_feat[s, 14].item() + 0.01)

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


def _sample_structural_negatives(num_events, num_res, res_lo, device, *, avoid=None):
    """Uniform random resource destinations for self-supervised negatives.

    Standard temporal link-prediction negative sampling: pair each src with a
    resource drawn *uniformly at random* over the whole resource space. The
    objective then learns ``P(dst | src, history)`` — each entity's habitual
    access distribution — so an unusual access (a lateral movement, or a policy
    violation) is scored as a low-likelihood event under the learned baseline.

    De-circularisation guard: this sampler takes only the resource id-range, **not**
    the data generator's ``auth_mask`` / habitual sets. It therefore cannot encode
    the evaluation's *specific* anomaly construction (authorised-but-non-habitual).
    The previous hard-negative did exactly that (and weighted it 10x), which made
    the reported recall a memorised curriculum rather than honest generalisation.
    NOTE: this is the standard self-supervised setup, **not** a zero-shot claim —
    random destinations are mostly non-habitual, so the objective does learn the
    generic "unusual access" notion; it simply no longer mirrors the test rule.
    """
    neg = torch.randint(0, num_res, (num_events,), device=device) + res_lo
    if avoid is not None:
        # Avoid the degenerate case where the random draw equals the true benign dst
        # (a false negative label). One re-roll suffices at num_res >> 1.
        collide = neg == avoid
        if collide.any():
            neg[collide] = (
                torch.randint(0, num_res, (int(collide.sum()),), device=device) + res_lo
            )
    return neg


def train_tgn(cfg: TGNConfig = TGNConfig(), *, use_struct_head=True,
              use_hash_identity=True, use_hist_feats=True, save=True):
    """Train + evaluate the streaming TGN.

    The keyword flags drive the ablation study (``tests/ablations``): they toggle the
    structural-compatibility head and the hashed-identity embedding. ``save=False``
    skips persisting the deployable artifact (ablation runs must not clobber the
    full-model checkpoint in ``public/``). Returns a metrics dict.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    print("Generating streaming data...")
    src, dst, t, msg, y, types, node_features, resource_uris, auth_mask = generate_streaming_data(
        num_users=cfg.num_users,
        num_ips=cfg.num_ips,
        num_resources=cfg.num_resources,
        num_events=cfg.num_events,
        benign_explore_prob=cfg.benign_explore_prob,
        seed=cfg.seed,
    )

    n = len(src)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)
    train_end, val_end = n_train, n_train + n_val
    bs = cfg.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # NOTE: ``auth_mask`` (the generator's per-IP authorised-resource matrix) is
    # intentionally NOT used during training — see _sample_structural_negatives.
    # Using it would re-introduce the circular "authorised-but-non-habitual" negative
    # that mirrors the lateral-movement test anomaly. It stays unused on purpose.

    print("--- INIZIO ADDESTRAMENTO UNSUPERVISED ---")

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
        hash_buckets=cfg.hash_buckets,
        hash_dim=cfg.hash_dim,
        hist_feat_dim=cfg.hist_feat_dim,
    ).to(device)

    # Load the static node attributes (role / clearance / tier) into the model's
    # buffer for the preregistered training entities. Slots reserved for entities
    # first seen at serving time stay zero until those entities supply their features.
    with torch.no_grad():
        model.node_feat[: cfg.total_nodes] = node_features.to(device)
        # Hashed Identity Trick (deterministic across processes/runs — see stable_hash).
        hashes = [stable_hash(registry._idx_to_key[i], cfg.hash_buckets) for i in range(cfg.total_nodes)]
        model.node_hash[: cfg.total_nodes] = torch.tensor(hashes, dtype=torch.long, device=device)

    # Bounded temporal neighbour loader (built on the model device after .to).
    model.init_neighbor_loader(cfg.neighbor_size, device)

    # Ablation switches (default ON = full model).
    model.use_struct_head = use_struct_head
    model.use_hash_identity = use_hash_identity
    model.use_hist_feats = use_hist_feats
    if not (use_struct_head and use_hash_identity and use_hist_feats):
        print(f"[ablation] use_struct_head={use_struct_head} use_hash_identity={use_hash_identity} "
              f"use_hist_feats={use_hist_feats}")

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)

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
        model.last_contact.clear()  # ...and the per-pair recency cache (Δt must reset too)
        model.pair_count.clear()  # ...and the interaction-history counters
        model.src_count.clear()
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

            # NOTE: the trust score (node_feat[:, 14]) is NO LONGER mutated from
            # ground-truth labels here. Doing so coupled the labels into a persistent
            # input feature and trained a self-fulfilling "low trust ⇒ anomaly" signal.
            # Trust is now a static, orchestrator-supplied attribute (optionally evolved
            # at serving time only); it is never used to build a training negative.

            # Only train on benign events.
            benign_mask = b_y == 0
            if not benign_mask.any():
                continue

            p_src = b_src[benign_mask]
            p_dst = b_dst[benign_mask]
            p_t = b_t[benign_mask]
            p_msg = b_msg[benign_mask]

            # STRUCTURAL NEGATIVES — for each positive, K uniform random resource
            # destinations (standard self-supervised temporal link prediction). They are
            # independent of the generator's auth_mask / habitual sets, so detection is
            # honest generalisation, not a memorised test rule. The objective below is a
            # *ranking* loss (InfoNCE): rank the true dst above the K alternatives given
            # the src's history — the likelihood signal that exposes lateral movement.
            P = len(p_src)
            K = cfg.infonce_k
            res_lo = cfg.total_nodes - cfg.num_resources
            num_res = cfg.num_resources
            neg_flat = _sample_structural_negatives(
                P * K, num_res, res_lo, device, avoid=p_dst.repeat_interleave(K)
            )  # (P*K,)
            src_rep = p_src.repeat_interleave(K)  # (P*K,)

            # Expand every involved node to its stored temporal neighbourhood and embed
            # once; the heads below differ only in which endpoints / message they score,
            # sharing the same history-conditioned embeddings.
            nodes = torch.cat([p_src, p_dst, neg_flat]).unique()
            n_id, edge_index, hist_t, hist_msg = model.neighbor_loader(nodes)
            z = model.embed(n_id, edge_index, hist_t, hist_msg)
            assoc = model.neighbor_loader._assoc
            nf = model.node_feat[n_id]
            h_idx = model.node_hash[n_id]
            s_loc, d_loc = assoc[p_src], assoc[p_dst]
            sK_loc, negK_loc = assoc[src_rep], assoc[neg_flat]

            # RECENCY (Δt since last src→dst contact) + interaction-history features.
            tv_l, s_l, dpos_l = p_t.tolist(), p_src.tolist(), p_dst.tolist()
            delta_t_pos = torch.tensor(
                [tv_l[i] - model.last_contact.get((s_l[i], dpos_l[i]), 0) for i in range(P)],
                dtype=torch.float, device=device,
            )
            srcrep_l, negflat_l = src_rep.tolist(), neg_flat.tolist()
            tvrep_l = p_t.repeat_interleave(K).tolist()
            delta_t_neg = torch.tensor(
                [tvrep_l[j] - model.last_contact.get((srcrep_l[j], negflat_l[j]), 0) for j in range(P * K)],
                dtype=torch.float, device=device,
            )
            delta_t_src = (p_t - model.memory.last_update[p_src]).to(torch.float)
            delta_t_src_rep = delta_t_src.repeat_interleave(K)

            hist_pos = model.compute_hist_feats(s_l, dpos_l, device)
            hist_neg = model.compute_hist_feats(srcrep_l, negflat_l, device)
            msg_rep = p_msg.repeat_interleave(K, dim=0)

            # --- POSITIVE EDGES (healthy behaviour) ---
            pos_out = model.score(z, nf, h_idx, s_loc, d_loc, p_msg, delta_t_pos, delta_t_src, hist_pos)

            # --- STRUCTURAL NEGATIVES (src paired with K random resources) ---
            negK_out = model.score(
                z, nf, h_idx, sK_loc, negK_loc, msg_rep, delta_t_neg, delta_t_src_rep, hist_neg
            ).view(P, K)

            # --- CONTEXTUAL NEGATIVES (off-manifold edge message via additive Gaussian
            # noise). A *different mechanism* from the eval's contextual anomalies
            # (discrete 0/1 signal randomisation), so the model is not handed the test
            # corruption. Contextual anomalies are near-trivial on edge features (the
            # rule baseline already catches them — see eval), so this term mainly keeps
            # the feature head from ignoring the message.
            neg_msg = p_msg + torch.randn_like(p_msg) * 0.5
            neg_out_ctx = model.score(z, nf, h_idx, s_loc, d_loc, neg_msg, delta_t_pos, delta_t_src, hist_pos)

            # --- UNSUPERVISED LOSS ---
            #   * InfoNCE ranking: among {true dst, K random dsts} the true dst must score
            #     most-benign (highest logit) given the src's history. This is the lateral
            #     objective and is AP-aligned (a relative/soft target, unlike the old hard
            #     0/1 negative that over-fit the circular non-habitual definition).
            #   * positive BCE anchor: keeps benign logits high so the FPR-calibrated
            #     threshold is meaningful (InfoNCE alone fixes only relative order).
            #   * contextual BCE: off-manifold message ⇒ anomalous.
            logits = torch.cat([pos_out.unsqueeze(1), negK_out], dim=1)  # (P, 1+K); col 0 = positive
            target = torch.zeros(P, dtype=torch.long, device=device)
            loss = (
                F.cross_entropy(logits, target)
                + F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
                + F.binary_cross_entropy_with_logits(neg_out_ctx, torch.zeros_like(neg_out_ctx))
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Predict-then-update: commit benign traffic to memory, neighbour store, the
            # recency cache and the interaction-history counters (benign-only — anomalies
            # never enter the baseline, matching the serving anti-poisoning gate).
            model.memory.update_state(p_src, p_dst, p_t, p_msg)
            model.memory.detach()
            model.neighbor_loader.insert(p_src, p_dst, p_t, p_msg)
            for i in range(P):
                s = s_l[i]
                d = dpos_l[i]
                model.last_contact[(s, d)] = tv_l[i]
                model.pair_count[(s, d)] = model.pair_count.get((s, d), 0) + 1
                model.src_count[s] = model.src_count.get(s, 0) + 1

        print(f"Epoch {epoch:02d} | Train Loss: {total_loss / max(num_train_batches, 1):.4f}")

    # --- THRESHOLD CALIBRATION (held-out benign slice) -----------------------
    print("\n--- CALIBRAZIONE SOGLIA (su flusso di validazione benigno) ---")
    # The eval replay evolves the runtime trust feature (node_feat[:, 14]); snapshot the
    # post-training node features so calibration does not bleed trust state into the test
    # replay and so re-runs stay idempotent.
    node_feat_post_train = model.node_feat.clone()
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
    # Memory + neighbour history legitimately continue from the (benign) calibration
    # slice, but reset the runtime trust feature to the post-training snapshot so the
    # test stream is not pre-conditioned by calibration.
    model.node_feat.copy_(node_feat_post_train)
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
    # Per-type AUC/AP are benign (type 0) vs that type, so an aggregate cannot mask a
    # weak class. NONE of the training negatives encode the generator's anomaly rules
    # (InfoNCE over random-destination negatives + Gaussian feature noise — see the loss),
    # AND the task is no longer degenerate: benign traffic now *also* makes
    # authorised-but-non-habitual accesses (benign_explore_prob), so "non-habitual" is not
    # a free label for lateral — the model must use the temporal pattern + history signals.
    # These numbers are therefore honest generalisation. The "vs-rule" column says whether
    # the cheap signal-only rule baseline (below) can also catch the class — i.e. how much
    # the TGN genuinely adds:
    #   policy  -> OPA-OWNED: blocked deterministically upstream; reported only as a sanity
    #     column, NOT a value-add of this model (kept for completeness).
    #   contextual -> rule-trivial: broken JA3 / Snort fires; the rule baseline matches it.
    #   lateral -> rule-blind and the GENUINELY HARD case (authorised, signal-clean,
    #     indistinguishable from benign exploration except by temporal/relational pattern).
    #     This is the model's real target.
    print("\n--- METRICHE PER TIPO DI ANOMALIA ---")
    vs_rule = {1: "OPA-owned  ", 2: "rule-trivial", 3: "rule-blind "}
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
        per_type[name] = {"auc": t_auc, "ap": t_ap, "recall": t_recall, "n": int(l_sel.sum())}
        print(f"  {name:10s} | {vs_rule[type_id]} | n={int(l_sel.sum()):4d} | AUC: {t_auc:.4f} | "
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
    base_ctx_recall = (
        ((base_pred == 1) & (test_types == 2)).sum() / max((test_types == 2).sum(), 1)
    )
    base_lateral_recall = (
        ((base_pred == 1) & (test_types == 3)).sum() / max((test_types == 3).sum(), 1)
    )
    print("\n--- BASELINE A REGOLE (signal-only) ---")
    print(f"  Precision: {b_precision:.4f} | Recall: {b_recall:.4f} | "
          f"Recall policy: {base_policy_recall:.4f} | Recall contextual: {base_ctx_recall:.4f} | "
          f"Recall lateral: {base_lateral_recall:.4f}")
    print("  NOTE: the rule baseline catches contextual anomalies almost entirely (edge "
          "signals only) but is blind to policy/lateral — that gap is the TGN's value-add.")

    # --- PERSIST DEPLOYABLE ARTIFACT -----------------------------------------
    # Ablation runs (save=False) must not overwrite the full-model artifact in public/.
    if save:
        hp = {
            "capacity": cfg.capacity,
            "node_feat_dim": cfg.node_feat_dim,
            "msg_dim": cfg.msg_dim,
            "memory_dim": cfg.memory_dim,
            "time_dim": cfg.time_dim,
            "num_hops": cfg.num_hops,
            "hash_buckets": cfg.hash_buckets,
            "hash_dim": cfg.hash_dim,
            "hist_feat_dim": cfg.hist_feat_dim,
            "neighbor_size": cfg.neighbor_size,
            "target_fpr": cfg.target_fpr,
        }
        save_model(model, registry, threshold, hp, TGN_CHECKPOINT_PATH, TGN_STATS_PATH)
        print(f"\nSaved checkpoint -> {TGN_CHECKPOINT_PATH}")
        print(f"Saved stats      -> {TGN_STATS_PATH}")

    return {
        "threshold": threshold,
        "agg_auc": auc,
        "agg_ap": ap,
        "agg_precision": precision,
        "agg_recall": recall,
        "per_type": per_type,
        "use_struct_head": use_struct_head,
        "use_hash_identity": use_hash_identity,
        "use_hist_feats": use_hist_feats,
    }


if __name__ == "__main__":
    train_tgn()
