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
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import average_precision_score, roc_auc_score

from graphagate.calibration import (
    cost_sensitive_threshold,
    operating_point,
    recall_fpr_curve,
    routed_predict,
)
from graphagate.config import TGNConfig, TGN_CHECKPOINT_PATH, TGN_STATS_PATH
from graphagate.data.stream_synthetic import generate_streaming_data
from graphagate.eval_common import causal_src_seen
from graphagate.model.registry import NodeRegistry
from graphagate.model.tgn import ZTATemporalGraphNetwork, stable_hash
from graphagate.serve_tgn import (
    infer_score,
    precursor_boost,
    record_alert,
    save_model,
    signal_dirty,
    update_memory,
)


def _replay(model, src, dst, t, msg, y, device, *, threshold=None,
            threshold_dirty=None, gate_by_label=False):
    """Per-event streaming replay matching the serving path.

    Memory update gating:
      - ``gate_by_label=True``  -> update on ground-truth benign (calibration);
      - ``gate_by_label=False`` -> update on predicted benign (``score < threshold``),
        i.e. the realistic, label-free serving behaviour.

    Decision threshold mirrors :func:`serve_tgn.score_event`: when ``threshold_dirty`` is
    given, the decision is *signal-routed* — events whose edge signal fires (broken JA3 /
    Snort / sensor) use ``threshold_dirty`` and the rest use the recall-oriented
    ``threshold`` (the signal-clean threshold). With ``threshold_dirty=None`` a single
    ``threshold`` is used everywhere (legacy behaviour). Returns ``(scores, labels)``.
    """
    model.eval()
    scores = np.empty(src.shape[0], dtype=np.float64)
    labels = np.empty(src.shape[0], dtype=np.int64)
    src_l, dst_l, t_l, y_l = src.tolist(), dst.tolist(), t.tolist(), y.tolist()

    for i in range(len(src_l)):
        s, d, tv, lab = src_l[i], dst_l[i], t_l[i], y_l[i]
        msg_vec = msg[i]
        # Same scoring path as serving: raw model score then the kill-chain precursor
        # prior (boosts an entity that recently alerted). Computed before record_alert.
        raw_score = infer_score(model, s, d, tv, msg_vec, device)
        score = min(1.0, raw_score * precursor_boost(model, s, tv))
        scores[i] = score
        labels[i] = lab

        eff_thr = threshold
        if threshold_dirty is not None and signal_dirty(msg_vec):
            eff_thr = threshold_dirty

        do_update = (lab == 0) if gate_by_label else (score < eff_thr)
        if do_update:
            update_memory(model, s, d, tv, msg_vec, device)

        snort_alert = msg_vec[1] > 0.5
        is_anomaly = (score >= eff_thr) if not gate_by_label else (lab == 1)
        if is_anomaly or snort_alert:
            record_alert(model, s, tv)  # arm the precursor (recon → lateral)
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


def _pr_from_preds(preds, labels):
    """Precision / recall of pre-computed 0/1 ``preds`` (e.g. signal-routed) vs ``labels``."""
    preds = np.asarray(preds).astype(int)
    labels = np.asarray(labels).astype(int)
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


@dataclass
class StreamData:
    """A ZTA access stream ready for :func:`train_tgn`, decoupled from the generator.

    This lets the training/calibration/evaluation pipeline run unchanged on an externally
    mapped dataset (e.g. LANL auth — see ``tests/eval_lanl.py``) for external validity.
    Node indices are ``0..num_nodes-1`` and ``keys[i]`` is the external registry key for
    slot ``i`` (used for the deterministic hashed-identity embedding). Structural negatives
    are drawn uniformly from the destination id-range ``[neg_lo, neg_lo + neg_num)`` — the
    resource range for the synthetic stream, the whole computer range for host-to-host auth.
    """

    src: torch.Tensor
    dst: torch.Tensor
    t: torch.Tensor
    msg: torch.Tensor
    y: torch.Tensor
    types: torch.Tensor
    node_features: torch.Tensor
    keys: list
    num_nodes: int
    neg_lo: int
    neg_num: int


def _synthetic_stream_data(cfg: TGNConfig) -> StreamData:
    """Build :class:`StreamData` from the synthetic generator (the default path).

    Reproduces the previous in-line setup exactly: users + IPs are keyed by their integer
    ids, resources by their URI strings (so the orchestrator can send strings natively), and
    structural negatives are sampled over the resource id-range.
    """
    src, dst, t, msg, y, types, node_features, resource_uris, _auth_mask = generate_streaming_data(
        num_users=cfg.num_users,
        num_ips=cfg.num_ips,
        num_resources=cfg.num_resources,
        num_events=cfg.num_events,
        benign_explore_prob=cfg.benign_explore_prob,
        seed=cfg.seed,
    )
    # auth_mask is intentionally dropped — see _sample_structural_negatives (de-circularisation).
    keys = list(range(cfg.num_users + cfg.num_ips)) + list(resource_uris)
    return StreamData(
        src=src, dst=dst, t=t, msg=msg, y=y, types=types, node_features=node_features,
        keys=keys, num_nodes=cfg.total_nodes,
        neg_lo=cfg.total_nodes - cfg.num_resources, neg_num=cfg.num_resources,
    )


def train_tgn(cfg: TGNConfig = TGNConfig(), *, dataset: "StreamData | None" = None,
              use_struct_head=True, use_hash_identity=True, use_hist_feats=True,
              use_precursor=True, save=True):
    """Train + evaluate the streaming TGN.

    The keyword flags drive the ablation study (``tests/ablations``): they toggle the
    structural-compatibility head and the hashed-identity embedding. ``save=False``
    skips persisting the deployable artifact (ablation runs must not clobber the
    full-model checkpoint in ``public/``). ``dataset`` injects an externally-mapped
    :class:`StreamData` (e.g. LANL auth) instead of the synthetic generator, reusing the
    whole pipeline for external validity; ``None`` is the default synthetic path. Returns
    a metrics dict.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    if dataset is None:
        print("Generating streaming data...")
        data = _synthetic_stream_data(cfg)
    else:
        print("Using injected dataset stream...")
        data = dataset
    src, dst, t, msg, y, types = data.src, data.dst, data.t, data.msg, data.y, data.types
    node_features = data.node_features
    total_nodes = data.num_nodes
    capacity = total_nodes + cfg.capacity_headroom
    neg_lo, neg_num = data.neg_lo, data.neg_num

    n = len(src)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)
    train_end, val_end = n_train, n_train + n_val
    bs = cfg.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # NOTE: the data generator's per-IP authorised-resource matrix is intentionally NOT
    # used during training — see _sample_structural_negatives. Using it would re-introduce
    # the circular "authorised-but-non-habitual" negative that mirrors the lateral-movement
    # test anomaly. It stays unused on purpose (dropped in _synthetic_stream_data).

    print("--- INIZIO ADDESTRAMENTO UNSUPERVISED ---")

    # Entity registry: ``data.keys[i]`` is the external key for slot ``i`` (int ids for
    # users/IPs, URI strings for resources in the synthetic stream; computer names for LANL).
    registry = NodeRegistry(capacity=capacity)
    registry.preregister(data.keys)

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
    ).to(device)

    # Load the static node attributes (role / clearance / tier) into the model's
    # buffer for the preregistered training entities. Slots reserved for entities
    # first seen at serving time stay zero until those entities supply their features.
    with torch.no_grad():
        model.node_feat[:total_nodes] = node_features.to(device)
        # Hashed Identity Trick (deterministic across processes/runs — see stable_hash).
        hashes = [stable_hash(registry._idx_to_key[i], cfg.hash_buckets) for i in range(total_nodes)]
        model.node_hash[:total_nodes] = torch.tensor(hashes, dtype=torch.long, device=device)

    # Bounded temporal neighbour loader (built on the model device after .to).
    model.init_neighbor_loader(cfg.neighbor_size, device)

    # Ablation switches (default ON = full model).
    model.use_struct_head = use_struct_head
    model.use_hash_identity = use_hash_identity
    model.use_hist_feats = use_hist_feats
    model.use_precursor = use_precursor
    # Kill-chain precursor knobs (serving-time prior; see serve_tgn.precursor_boost).
    model.precursor_half_life = cfg.precursor_half_life
    model.precursor_max_boost = cfg.precursor_max_boost
    if not (use_struct_head and use_hash_identity and use_hist_feats and use_precursor):
        print(f"[ablation] use_struct_head={use_struct_head} use_hash_identity={use_hash_identity} "
              f"use_hist_feats={use_hist_feats} use_precursor={use_precursor}")

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
            res_lo = neg_lo
            num_res = neg_num
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
    val_types = types[train_end:val_end].numpy()
    val_msg = msg[train_end:val_end].numpy()

    # Signal-DIRTY threshold (events whose edge signal already fires): keep the conservative
    # benign-FPR quantile — the cheap rule baseline already catches these, no need to drop it.
    threshold_dirty = float(np.quantile(benign_val_scores, 1.0 - cfg.target_fpr))

    # Signal-CLEAN threshold: among signal-clean events the realistic discrimination the model
    # owns is benign vs lateral (contextual is signal-dirty; policy is OPA-blocked upstream).
    # Calibrate the cost-sensitive threshold on exactly that population so the ~0.76 lateral
    # AUC becomes recall. ``threshold`` is the primary (clean-stream) decision threshold.
    val_clean = ~_rule_baseline(val_msg).astype(bool)
    cal_mask = val_clean & ((val_labels == 0) | (val_types == 3))
    cal_scores = val_scores[cal_mask]
    cal_labels = (val_types[cal_mask] == 3).astype(int)
    if cal_labels.sum() == 0:
        # No lateral examples to calibrate against (e.g. a window without red-team activity):
        # fall back to the conservative FPR threshold so behaviour degrades gracefully.
        threshold = threshold_dirty
    else:
        threshold = cost_sensitive_threshold(
            cal_scores, cal_labels, cost_ratio=cfg.cost_ratio,
            target_fpr_cap=cfg.clean_fpr_cap,
        )
    print(
        f"Benign val score: mean={benign_val_scores.mean():.4f} "
        f"p95={np.quantile(benign_val_scores, 0.95):.4f}"
    )
    print(
        f"threshold_dirty@FPR={cfg.target_fpr}: {threshold_dirty:.4f} | "
        f"threshold_clean@cost_ratio={cfg.cost_ratio}: {threshold:.4f} "
        f"(clean cal: n_benign={int((cal_labels == 0).sum())} n_lateral={int(cal_labels.sum())})"
    )
    # Lateral recall/FPR trade-off the clean threshold was picked from (operator/OPA reference).
    if cal_labels.sum() > 0:
        print("  clean-stream lateral recall vs benign FPR (reference curve):")
        for thr, rec, fpr in recall_fpr_curve(cal_scores, cal_labels, n_points=6):
            print(f"    thr={thr:.4f} | lateral_recall={rec:.3f} | benign_fpr={fpr:.4f}")

    # --- STREAMING EVALUATION (event-by-event, predicted-benign gating) ------
    print("\n--- INIZIO FASE DI INFERENZA / ANOMALY DETECTION (per-evento) ---")
    # Memory + neighbour history legitimately continue from the (benign) calibration
    # slice, but reset the runtime trust feature to the post-training snapshot so the
    # test stream is not pre-conditioned by calibration.
    model.node_feat.copy_(node_feat_post_train)
    model.recent_alert.clear()  # don't let calibration-slice alerts pre-condition the test stream
    test_scores, test_labels = _replay(
        model, src[val_end:], dst[val_end:], t[val_end:], msg[val_end:], y[val_end:],
        device, threshold=threshold, threshold_dirty=threshold_dirty, gate_by_label=False,
    )

    test_types = types[val_end:].numpy()
    test_msg = msg[val_end:].numpy()

    # Signal-routed decision (matches serving): clean events -> cost-sensitive threshold,
    # dirty events -> conservative FPR threshold.
    dirty_test = _rule_baseline(test_msg).astype(bool)
    test_preds = routed_predict(test_scores, dirty_test, threshold, threshold_dirty)

    auc = roc_auc_score(test_labels, test_scores)
    ap = average_precision_score(test_labels, test_scores)
    precision, recall = _pr_from_preds(test_preds, test_labels)
    print(f"Test Stream | AUC: {auc:.4f} | AP: {ap:.4f}")
    print(f"Routed decision | Precision: {precision:.4f} | Recall: {recall:.4f}")

    # --- HEADLINE: AUC -> operational recall ---------------------------------
    # Before = the previous single global threshold (the benign-FPR quantile applied to
    # everything). After = signal-routed cost-sensitive decision. This is the value-add.
    lat = test_types == 3
    benign_test = test_labels == 0
    old_preds = (test_scores >= threshold_dirty).astype(int)
    old_lat_recall = float(old_preds[lat].mean()) if lat.any() else float("nan")
    new_lat_recall = float(test_preds[lat].mean()) if lat.any() else float("nan")
    old_fpr = float(old_preds[benign_test].mean()) if benign_test.any() else float("nan")
    new_fpr = float(test_preds[benign_test].mean()) if benign_test.any() else float("nan")
    print("\n--- LATERAL RECALL: GLOBAL-FPR THRESHOLD  vs  COST-SENSITIVE ROUTING ---")
    print(f"  before (global @FPR={cfg.target_fpr}): lateral_recall={old_lat_recall:.4f} | benign_fpr={old_fpr:.4f}")
    print(f"  after  (routed cost-sensitive)       : lateral_recall={new_lat_recall:.4f} | benign_fpr={new_fpr:.4f}")

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
        # Recall at the routed operational decision (not a single global threshold).
        preds_sel = test_preds[sel]
        t_recall = float(preds_sel[l_sel == 1].mean()) if (l_sel == 1).any() else 0.0
        per_type[name] = {"auc": t_auc, "ap": t_ap, "recall": t_recall, "n": int(l_sel.sum())}
        print(f"  {name:10s} | {vs_rule[type_id]} | n={int(l_sel.sum()):4d} | AUC: {t_auc:.4f} | "
              f"AP: {t_ap:.4f} | Recall@thr: {t_recall:.4f}")

    # --- COLD-START CONDITIONING (lateral) -----------------------------------
    # Lateral movement can only be flagged for an entity the model has *some* history on;
    # many laterals hit IPs that are still cold (the stream admits IPs progressively, and
    # a freshly-compromised IP may have no benign history yet). Report lateral recall split
    # by whether the src had >=1 benign event before the event (warmed) vs not (cold), so
    # the honest "where detection is even possible" number is visible next to the overall one.
    src_seen_test = causal_src_seen(src.numpy(), y.numpy())[val_end:]
    lat_mask = test_types == 3
    lat_pred = test_preds  # routed operational decision
    warmed = lat_mask & src_seen_test
    cold = lat_mask & ~src_seen_test
    cold_start = {"n_warmed": int(warmed.sum()), "n_cold": int(cold.sum())}
    cold_start["recall_warmed"] = float(lat_pred[warmed].mean()) if warmed.any() else float("nan")
    cold_start["recall_cold"] = float(lat_pred[cold].mean()) if cold.any() else float("nan")
    if warmed.any() and lat_mask.any():
        from sklearn.metrics import roc_auc_score as _auc
        warmed_sel = (test_types == 0) | warmed
        cold_start["auc_warmed"] = float(_auc((test_types[warmed_sel] == 3).astype(int), test_scores[warmed_sel]))
    print("\n--- LATERAL: COLD-START CONDITIONING ---")
    print(f"  warmed src (has benign history): n={cold_start['n_warmed']:4d} | "
          f"recall@thr={cold_start['recall_warmed']:.4f} | AUC={cold_start.get('auc_warmed', float('nan')):.4f}")
    print(f"  cold   src (no history yet)     : n={cold_start['n_cold']:4d} | "
          f"recall@thr={cold_start['recall_cold']:.4f}  (detection not yet possible)")

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
            "capacity": capacity,
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
            "cost_ratio": cfg.cost_ratio,
            "clean_fpr_cap": cfg.clean_fpr_cap,
            "precursor_half_life": cfg.precursor_half_life,
            "precursor_max_boost": cfg.precursor_max_boost,
        }
        op_new = operating_point(test_scores, test_labels, test_types, threshold)
        save_model(
            model, registry, threshold, hp, TGN_CHECKPOINT_PATH, TGN_STATS_PATH,
            threshold_dirty=threshold_dirty,
            calibration={"mode": "cost", "cost_ratio": cfg.cost_ratio,
                         "clean_fpr_cap": cfg.clean_fpr_cap, "target_fpr": cfg.target_fpr},
            operating_point=op_new,
        )
        print(f"\nSaved checkpoint -> {TGN_CHECKPOINT_PATH}")
        print(f"Saved stats      -> {TGN_STATS_PATH}")

    return {
        "threshold": threshold,
        "threshold_dirty": threshold_dirty,
        "agg_auc": auc,
        "agg_ap": ap,
        "agg_precision": precision,
        "agg_recall": recall,
        "lateral_recall_before": old_lat_recall,
        "lateral_recall_after": new_lat_recall,
        "fpr_before": old_fpr,
        "fpr_after": new_fpr,
        "per_type": per_type,
        "cold_start": cold_start,
        "use_struct_head": use_struct_head,
        "use_hash_identity": use_hash_identity,
        "use_hist_feats": use_hist_feats,
        "use_precursor": use_precursor,
    }


if __name__ == "__main__":
    train_tgn()
