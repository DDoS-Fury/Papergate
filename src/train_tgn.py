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
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from graphagate.calibration import (
    cost_sensitive_threshold,
    operating_point,
    recall_fpr_curve,
    routed_predict,
)
from graphagate.config import TGNConfig, TGN_CHECKPOINT_PATH, TGN_STATS_PATH
from graphagate.data.stream_synthetic import (
    SCEN_ROAMING,
    SCEN_SHARED,
    SCEN_WIPED,
    generate_streaming_data,
)
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


def _pbar(iterable=None, *, total=None, desc=None):
    """A tqdm progress bar that renders cleanly in a TTY *and* in ``docker compose`` logs.

    On a real terminal it updates smoothly; when stderr is not a TTY (docker-compose / CI)
    it throttles to one ASCII line every ~10 s, so the logs get periodic, readable progress
    instead of a flood of carriage returns. It is never disabled — watching progress in the
    container logs is exactly the point.
    """
    is_tty = sys.stderr.isatty()
    return tqdm(
        iterable, total=total, desc=desc, disable=False,
        mininterval=(0.5 if is_tty else 10.0), dynamic_ncols=True, ascii=not is_tty,
    )


def _replay(model, source_nodes, device_nodes, user, dst, t, msg, y, device, *,
            config_nodes=None, threshold=None, threshold_dirty=None, gate_by_label=False,
            batch_size=1, desc="replay"):
    """Streaming replay matching the serving path (v4: up to 5 edges), optionally batched.

    ``source_nodes`` / ``config_nodes`` / ``device_nodes`` may be ``None`` (datasets
    without a network / config / hardware entity, e.g. LANL host-to-host auth): the
    corresponding binding edges are skipped, exactly like :func:`serve_tgn.score_event`
    with the matching key omitted. They are dataset-level (every event carries them, or
    none does), so each block is a set of equally-shaped edge groups. The causal chain is
    ``source → config → device → user → resource`` plus ``config → user``.

    Memory update gating (who is committed into TGN memory):
      - ``gate_by_label=True``  -> commit on ground-truth benign (calibration replay,
        which has no threshold yet so it cannot self-gate);
      - ``gate_by_label=False`` -> commit on the **OPA-ALLOW proxy**: every event the
        deterministic signal layer does not flag (``not signal_dirty``). This mirrors the
        deployed two-step flow — :func:`serve_tgn.score_event` with ``update=False`` returns
        the score, OPA folds it (soft-weighted) into a decision *dominated by the
        deterministic policy/sensor layer*, and :func:`serve_tgn.commit_event` runs only on
        ALLOW. Gating the commit on the model's *own* score (``score < threshold``) instead
        is the no-OPA, self-serving deployment: it starves misclassified-benign (FP) events
        out of memory, their state goes stale, their score stays high, and the benign FPR
        runs away (the val->test blow-up). Committing on the observable signal keeps benign
        memory fresh; the residual surface — signal-clean laterals OPA would also admit — is
        committed too, so the reported FPR is conservative rather than oracle-optimistic.

    Decision threshold mirrors :func:`serve_tgn.score_event`: when ``threshold_dirty`` is
    given, the decision is *signal-routed* — events whose edge signal fires (broken JA3 /
    Snort / sensor) use ``threshold_dirty`` and the rest use the recall-oriented
    ``threshold`` (the signal-clean threshold). With ``threshold_dirty=None`` a single
    ``threshold`` is used everywhere (legacy behaviour). Returns ``(scores, labels)``.

    ``batch_size`` controls offline GPU batching: ``1`` is bit-for-bit the old per-event
    loop; ``B>1`` scores a block of ``B`` events against the *start-of-batch* memory snapshot
    with one shared neighbour expansion + GNN forward (the exact pattern the training loop
    already uses) and commits the benign-gated memory updates in batch afterwards — the
    standard batched-TGN regime the model is trained under, which saturates the GPU on large
    streams (LANL). The cheap per-event feedback (kill-chain precursor, trust nudge,
    decision) stays strictly sequential, so the lateral-movement precursor keeps its exact
    within-batch ordering. This is the OFFLINE eval/calibration path only — the online
    server (:mod:`serve_tgn`) is untouched and remains strictly sequential.
    """
    model.eval()
    N = int(user.shape[0])
    scores = np.empty(N, dtype=np.float64)
    labels = np.asarray(y.tolist(), dtype=np.int64)
    u_l, d_l, t_l, y_l = user.tolist(), dst.tolist(), t.tolist(), y.tolist()
    dev_l = device_nodes.tolist() if device_nodes is not None else None
    src_l = source_nodes.tolist() if source_nodes is not None else None
    cfg_l = config_nodes.tolist() if config_nodes is not None else None
    has_bind = device_nodes is not None
    has_src = has_bind and source_nodes is not None
    has_config = config_nodes is not None
    msg_dim = msg.shape[1]
    TRUST = 14  # runtime trust slot in node_feat (the only column replay mutates)

    # CPU mirror of the trust feature: the sequential per-event feedback mutates it exactly
    # (at any batch size), written back to the GPU column once per block so the next block's
    # scoring sees it. Only Phase 1 reads node_feat[:, TRUST] (on GPU); only Phase 2 writes it.
    trust = model.node_feat[:, TRUST].detach().cpu().numpy().copy()

    def _bump(a, b, tv):
        """Advance the benign-gated recency / interaction-history counters (cf. update_memory)."""
        model.last_contact[(a, b)] = tv
        model.pair_count[(a, b)] = model.pair_count.get((a, b), 0) + 1
        model.src_count[a] = model.src_count.get(a, 0) + 1

    pbar = _pbar(total=N, desc=desc)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start

        with torch.no_grad():
            bu = user[start:end].to(device)
            bd = dst[start:end].to(device)
            bt = t[start:end].to(device)
            bmsg = msg[start:end].to(device).float()
            bdev = device_nodes[start:end].to(device) if has_bind else None
            bsrc = source_nodes[start:end].to(device) if has_src else None
            bcfg = config_nodes[start:end].to(device) if has_config else None
            zeros_msg = torch.zeros(B, msg_dim, device=device)
            us, ds, ts = u_l[start:end], d_l[start:end], t_l[start:end]
            devs = dev_l[start:end] if has_bind else None
            srcs = src_l[start:end] if has_src else None
            cfgs = cfg_l[start:end] if has_config else None

            # --- Phase 1: one shared neighbour expansion + GNN forward for the whole block.
            node_parts = [bu, bd]
            if has_bind:
                node_parts.append(bdev)
            if has_src:
                node_parts.append(bsrc)
            if has_config:
                node_parts.append(bcfg)
            nodes = torch.cat(node_parts).unique()
            n_id, edge_index, hist_t, hist_msg = model.neighbor_loader(nodes)
            z = model.embed(n_id, edge_index, hist_t, hist_msg)
            assoc = model.neighbor_loader._assoc
            nf = model.node_feat[n_id]
            h_idx = model.node_hash[n_id]

            def _grp_anom(s_g, d_g, s_list, d_list, msg_g, aux_list):
                """Anomaly score (1 - P(benign)) per edge of one group — a vectorised infer_score."""
                d_pair = torch.tensor(
                    [ts[k] - model.last_contact.get((s_list[k], d_list[k]), 0) for k in range(B)],
                    dtype=torch.float, device=device,
                )
                d_src = (bt - model.memory.last_update[s_g]).to(torch.float)
                hist = model.compute_hist_feats(s_list, d_list, device, aux_src_ids=aux_list)
                logit = model.score(
                    z, nf, h_idx, assoc[s_g], assoc[d_g], msg_g, d_pair, d_src, hist
                )
                return 1.0 - torch.sigmoid(logit)

            raw = _grp_anom(bu, bd, us, ds, bmsg, devs)  # access edge (aux src = device)
            if has_bind:
                raw = torch.maximum(raw, _grp_anom(bdev, bu, devs, us, zeros_msg, None))  # device→user
            if has_config:
                raw = torch.maximum(raw, _grp_anom(bcfg, bu, cfgs, us, zeros_msg, None))  # config→user
                if has_bind:
                    raw = torch.maximum(raw, _grp_anom(bcfg, bdev, cfgs, devs, zeros_msg, None))  # config→device
                if has_src:
                    raw = torch.maximum(raw, _grp_anom(bsrc, bcfg, srcs, cfgs, zeros_msg, None))  # source→config
            if has_src and not has_config:
                raw = torch.maximum(raw, _grp_anom(bsrc, bdev, srcs, devs, zeros_msg, None))  # legacy source→device
            raw_np = raw.detach().cpu().numpy()
            bmsg_rows = bmsg.tolist()

            # --- Phase 2: sequential per-event feedback (precursor / trust / decision).
            do_update = np.zeros(B, dtype=bool)
            for j in range(B):
                i = start + j
                lab = y_l[i]
                u = us[j]
                tv = ts[j]
                actor = devs[j] if has_bind else u
                score = min(1.0, float(raw_np[j]) * precursor_boost(model, actor, tv))
                scores[i] = score

                eff_thr = threshold
                msg_row = bmsg_rows[j]
                if threshold_dirty is not None and signal_dirty(msg_row):
                    eff_thr = threshold_dirty
                # Commit gate = OPA ALLOW proxy (signal-clean), NOT the model's own score:
                # see the docstring. eff_thr below still routes the trust/precursor alarm.
                do_update[j] = (lab == 0) if gate_by_label else (not signal_dirty(msg_row))

                snort_alert = msg_row[1] > 0.5
                is_anomaly = (lab == 1) if gate_by_label else (score >= eff_thr)
                if is_anomaly or snort_alert:
                    record_alert(model, actor, tv)  # arm the precursor (recon → lateral)
                    trust[actor] = max(0.0, trust[actor] - 0.5)
                    if actor != u:
                        trust[u] = max(0.0, trust[u] - 0.5)
                else:
                    trust[actor] = min(1.0, trust[actor] + 0.01)
                    if actor != u:
                        trust[u] = min(1.0, trust[u] + 0.01)

            # --- Phase 3: batched benign-gated memory / neighbour / counter commit (serving
            # edge order: source→device, device→user, user→resource — mirrors the train loop).
            sel = np.nonzero(do_update)[0]
            if sel.size:
                sel_t = torch.as_tensor(sel, dtype=torch.long, device=device)
                su, sd, st_, sm = bu[sel_t], bd[sel_t], bt[sel_t], bmsg[sel_t]
                zb = torch.zeros(int(sel.size), msg_dim, device=device)

                def _commit(s_t, d_t, m_t):
                    model.memory.update_state(s_t, d_t, st_, m_t)
                    model.memory.detach()
                    model.neighbor_loader.insert(s_t, d_t, st_, m_t)

                # Commit order: source→config, config→device, config→user, device→user, user→resource.
                if has_config and has_src:
                    _commit(bsrc[sel_t], bcfg[sel_t], zb)
                if has_config and has_bind:
                    _commit(bcfg[sel_t], bdev[sel_t], zb)
                if has_config:
                    _commit(bcfg[sel_t], su, zb)
                if has_src and not has_config:
                    _commit(bsrc[sel_t], bdev[sel_t], zb)
                if has_bind:
                    _commit(bdev[sel_t], su, zb)
                _commit(su, sd, sm)
                for j in sel.tolist():
                    u, d, tv = us[j], ds[j], ts[j]
                    if has_config and has_src:
                        _bump(srcs[j], cfgs[j], tv)
                    if has_config and has_bind:
                        _bump(cfgs[j], devs[j], tv)
                    if has_config:
                        _bump(cfgs[j], u, tv)
                    if has_src and not has_config:
                        _bump(srcs[j], devs[j], tv)
                    if has_bind:
                        _bump(devs[j], u, tv)
                    _bump(u, d, tv)
                    if has_bind:  # aux (device, resource) habituality counter (no temporal edge)
                        dvd = (devs[j], d)
                        model.pair_count[dvd] = model.pair_count.get(dvd, 0) + 1

            # Commit this block's trust feedback so the next block scores against it.
            model.node_feat[:, TRUST] = torch.as_tensor(
                trust, dtype=model.node_feat.dtype, device=device
            )
        pbar.update(B)
    pbar.close()
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

    Columns are ``[ja3, s1, s2, s3, method, role, clearance]``: an event is suspicious
    when the TLS trust is broken (``ja3==0``) or any Snort/sensor probe fires. Column 4
    is the HTTP method, NOT a sensor (an earlier revision included it, silently flagging
    every POST). This catches *contextual* anomalies but is blind to *policy* violations
    (which share benign edge features) — it quantifies how much the TGN adds beyond rules.
    """
    return (
        (test_msg[:, 0] == 0.0)
        | (test_msg[:, 1] == 1.0)
        | (test_msg[:, 2] == 1.0)
        | (test_msg[:, 3] == 1.0)
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
    for the access edge are drawn uniformly from ``[neg_lo, neg_lo + neg_num)`` — the
    resource range for the synthetic stream, the whole computer range for host-to-host auth.

    v4 schema: ``source_nodes`` (client IP), ``config_nodes`` (TLS/JA3 fingerprint) and
    ``device_nodes`` (hardware id) are optional. ``None`` skips the corresponding binding
    edge(s) and their training objective — a device-less dataset (LANL) degrades to the
    single ``user → dst`` edge. The ``usr_lo/usr_num`` / ``dev_lo/dev_num`` /
    ``cfg_lo/cfg_num`` ranges drive the binding-edge negative sampling (random users for
    device→user and config→user, random devices for config→device, random configs for
    source→config); they are only needed when the corresponding entity tensors are
    present. The causal chain is ``source → config → device → user → dst`` plus the
    ``config → user`` binding. ``scenario`` is the benign-context bitmask of the synthetic
    generator (``None`` skips scenario evals).
    """

    user: torch.Tensor
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
    device_nodes: torch.Tensor | None = None
    source_nodes: torch.Tensor | None = None
    config_nodes: torch.Tensor | None = None
    scenario: torch.Tensor | None = None
    usr_lo: int = 0
    usr_num: int = 0
    dev_lo: int = 0
    dev_num: int = 0
    cfg_lo: int = 0
    cfg_num: int = 0


def _synthetic_stream_data(cfg: TGNConfig) -> StreamData:
    """Build :class:`StreamData` from the synthetic v2 generator (the default path).

    Users are keyed by their integer ids, devices by ``tpm:<id>`` / ``ck:<uuid>``
    strings, sources by IP strings and resources by their URI strings (so the
    orchestrator can send all of them natively); structural negatives are sampled over
    the resource id-range.
    """
    s = generate_streaming_data(
        num_users=cfg.num_users,
        num_devices=cfg.num_devices,
        num_sources=cfg.num_sources,
        num_configs=cfg.num_configs,
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
        use_resource_risk=cfg.use_resource_risk,
        use_source_internal=cfg.use_source_internal,
        guest_device_fallback=cfg.guest_device_fallback,
    )
    return StreamData(
        user=s.user, dst=s.dst, t=s.t, msg=s.msg, y=s.y, types=s.types,
        node_features=s.node_features, keys=s.keys, num_nodes=s.num_nodes,
        neg_lo=s.res_lo, neg_num=s.res_num,
        device_nodes=s.device, source_nodes=s.source, config_nodes=s.config,
        scenario=s.scenario,
        usr_lo=s.user_lo, usr_num=s.user_num, dev_lo=s.dev_lo, dev_num=s.dev_num,
        cfg_lo=s.cfg_lo, cfg_num=s.cfg_num,
    )


def train_tgn(cfg: TGNConfig = TGNConfig(), *, dataset: "StreamData | None" = None,
              use_struct_head=True, use_hash_identity=True, use_hist_feats=True,
              use_precursor=True, use_config_node=True, save=True):
    """Train + evaluate the streaming TGN.

    The keyword flags drive the ablation study (``tests/ablations``): they toggle the
    structural-compatibility head and the hashed-identity embedding. ``use_config_node``
    (v4) drops the configuration node from training/eval — the stream is unchanged but
    the model falls back to the legacy ``source→device`` chain, i.e. a ≈v3 run on the
    same data; the Δ vs the full model isolates the config node's contribution.
    ``save=False`` skips persisting the deployable artifact (ablation runs must not
    clobber the full-model checkpoint in ``public/``). ``dataset`` injects an
    externally-mapped :class:`StreamData` (e.g. LANL auth) instead of the synthetic
    generator, reusing the whole pipeline for external validity; ``None`` is the default
    synthetic path. Returns a metrics dict.
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
    user_arr, dst, t, msg, y, types = data.user, data.dst, data.t, data.msg, data.y, data.types
    device_arr, source_arr, scenario = data.device_nodes, data.source_nodes, data.scenario
    config_arr = data.config_nodes
    # Config-node ablation (≈v3): drop the configuration node so the whole pipeline
    # (training loop, both _replay calls, history counters) falls back to the legacy
    # source→device chain on the SAME stream. Isolates the config node's contribution.
    if not use_config_node:
        config_arr = None
    node_features = data.node_features
    total_nodes = data.num_nodes
    capacity = total_nodes + cfg.capacity_headroom
    neg_lo, neg_num = data.neg_lo, data.neg_num

    n = len(dst)
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
        gnn_heads=cfg.gnn_heads,
        link_pred_hidden_layers=cfg.link_pred_hidden_layers,
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
    n = len(dst)
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

        epoch_bar = _pbar(range(num_train_batches), desc=f"Epoch {epoch:02d}/{cfg.epochs} [train]")
        for i in epoch_bar:
            optimizer.zero_grad()
            start_idx, end_idx = i * bs, i * bs + bs

            b_user = user_arr[start_idx:end_idx].to(device)
            b_dst = dst[start_idx:end_idx].to(device)
            b_t = t[start_idx:end_idx].to(device)
            b_msg = msg[start_idx:end_idx].to(device)
            b_y = y[start_idx:end_idx].to(device)
            b_device = device_arr[start_idx:end_idx].to(device) if device_arr is not None else None
            b_source = source_arr[start_idx:end_idx].to(device) if source_arr is not None else None
            b_config = config_arr[start_idx:end_idx].to(device) if config_arr is not None else None

            # NOTE: the trust score (node_feat[:, 14]) is NO LONGER mutated from
            # ground-truth labels here. Doing so coupled the labels into a persistent
            # input feature and trained a self-fulfilling "low trust ⇒ anomaly" signal.
            # Trust is now a static, orchestrator-supplied attribute (optionally evolved
            # at serving time only); it is never used to build a training negative.

            # Only train on benign events.
            benign_mask = b_y == 0
            if not benign_mask.any():
                continue

            p_user = b_user[benign_mask]
            p_dst = b_dst[benign_mask]
            p_t = b_t[benign_mask]
            p_msg = b_msg[benign_mask]
            p_device = b_device[benign_mask] if b_device is not None else None
            p_source = b_source[benign_mask] if b_source is not None else None
            p_config = b_config[benign_mask] if b_config is not None else None

            # STRUCTURAL NEGATIVES — for each positive, K uniform random alternatives
            # (standard self-supervised temporal link prediction), one set per edge of
            # the chain. They are independent of the generator's habitual sets, so
            # detection is honest generalisation, not a memorised test rule:
            #   * access  user→resource: K random RESOURCES — "which resources does this
            #     user habitually reach" (the lateral-movement objective);
            #   * binding device→user:   K random USERS — "which users does this machine
            #     habitually host" (the credential-theft / session-binding objective);
            #   * binding config→user:   K random USERS — "which clients does this user
            #     habitually use" (a stolen-credential client differs from the victim's);
            #   * binding config→device: K random DEVICES — "which machines run this
            #     config" (a new tool on a known device is lateral movement);
            #   * binding source→config: K random CONFIGS — "which clients live behind
            #     this IP" (the network-binding objective; replaces source→device, now
            #     routed through the config node; roaming positives teach IP tolerance).
            # The binding objectives are NOT optional: with the access edge anchored on
            # the (warm) user, a credential thief touching the victim's habitual
            # resources looks benign there — the anomaly lives only in the bindings.
            P = len(p_user)
            K = cfg.infonce_k
            neg_res = _sample_structural_negatives(
                P * K, neg_num, neg_lo, device, avoid=p_dst.repeat_interleave(K)
            )
            user_rep = p_user.repeat_interleave(K)
            has_bind = p_device is not None and data.dev_num > 0
            has_src = has_bind and p_source is not None
            has_config = p_config is not None and data.cfg_num > 0
            if has_bind:
                neg_usr = _sample_structural_negatives(
                    P * K, data.usr_num, data.usr_lo, device, avoid=user_rep
                )
                dev_rep = p_device.repeat_interleave(K)
            if has_src:
                src_rep = p_source.repeat_interleave(K)
            if has_config:
                cfg_rep = p_config.repeat_interleave(K)
                neg_cusr = _sample_structural_negatives(  # config→user negatives
                    P * K, data.usr_num, data.usr_lo, device, avoid=user_rep
                )
                if has_bind:
                    neg_cdev = _sample_structural_negatives(  # config→device negatives
                        P * K, data.dev_num, data.dev_lo, device, avoid=dev_rep
                    )
                if has_src:
                    neg_scfg = _sample_structural_negatives(  # source→config negatives
                        P * K, data.cfg_num, data.cfg_lo, device, avoid=cfg_rep
                    )
            if has_src and not has_config:
                # legacy source→device binding (no config node to route the chain through)
                neg_dev = _sample_structural_negatives(
                    P * K, data.dev_num, data.dev_lo, device, avoid=dev_rep
                )

            # Expand every involved node to its stored temporal neighbourhood and embed
            # once; the heads below differ only in which endpoints / message they score,
            # sharing the same history-conditioned embeddings.
            parts = [p_user, p_dst, neg_res]
            if has_bind:
                parts += [p_device, neg_usr]
            if has_config:
                parts += [p_config, neg_cusr]
                if has_bind:
                    parts += [neg_cdev]
                if has_src:
                    parts += [p_source, neg_scfg]
            if has_src and not has_config:
                parts += [p_source, neg_dev]
            nodes = torch.cat(parts).unique()
            n_id, edge_index, hist_t, hist_msg = model.neighbor_loader(nodes)
            z = model.embed(n_id, edge_index, hist_t, hist_msg)
            assoc = model.neighbor_loader._assoc
            nf = model.node_feat[n_id]
            h_idx = model.node_hash[n_id]

            tv_l, u_l, dpos_l = p_t.tolist(), p_user.tolist(), p_dst.tolist()
            dev_l = p_device.tolist() if has_bind else None
            src_l = p_source.tolist() if has_src else None
            cfg_l = p_config.tolist() if has_config else None
            tv_rep = p_t.repeat_interleave(K)

            def _edge_logits(src_nodes, dst_nodes, t_nodes, msgs, hist_feats):
                """Score one edge group: Δt(pair recency), Δt(src activity), then heads."""
                s_list, d_list, t_list = src_nodes.tolist(), dst_nodes.tolist(), t_nodes.tolist()
                d_pair = torch.tensor(
                    [t_list[j] - model.last_contact.get((s_list[j], d_list[j]), 0)
                     for j in range(len(s_list))],
                    dtype=torch.float, device=device,
                )
                d_src = (t_nodes - model.memory.last_update[src_nodes]).to(torch.float)
                return model.score(
                    z, nf, h_idx, assoc[src_nodes], assoc[dst_nodes], msgs, d_pair, d_src, hist_feats
                )

            zeros_msg = torch.zeros_like(p_msg)
            zeros_msg_rep = torch.zeros(P * K, p_msg.size(1), device=device)
            msg_rep = p_msg.repeat_interleave(K, dim=0)

            # --- ACCESS EDGE user→resource (full message + device-aux history) ---
            hist_acc_pos = model.compute_hist_feats(u_l, dpos_l, device, aux_src_ids=dev_l)
            hist_acc_neg = model.compute_hist_feats(
                user_rep.tolist(), neg_res.tolist(), device,
                aux_src_ids=dev_rep.tolist() if has_bind else None,
            )
            pos_access = _edge_logits(p_user, p_dst, p_t, p_msg, hist_acc_pos)
            neg_access = _edge_logits(user_rep, neg_res, tv_rep, msg_rep, hist_acc_neg).view(P, K)

            # --- CONTEXTUAL NEGATIVES (off-manifold edge message via additive Gaussian
            # noise). A *different mechanism* from the eval's contextual anomalies
            # (discrete 0/1 signal randomisation), so the model is not handed the test
            # corruption. Contextual anomalies are near-trivial on edge features (the
            # rule baseline already catches them — see eval), so this term mainly keeps
            # the feature head from ignoring the message.
            neg_msg = p_msg + torch.randn_like(p_msg) * 0.5
            neg_out_ctx = _edge_logits(p_user, p_dst, p_t, neg_msg, hist_acc_pos)

            # --- UNSUPERVISED LOSS ---
            #   * InfoNCE ranking per edge: among {true endpoint, K random alternatives}
            #     the true one must score most-benign given the src's history. AP-aligned
            #     (a relative/soft target, unlike a hard 0/1 negative).
            #   * positive BCE anchors: keep benign logits high so the FPR-calibrated
            #     threshold is meaningful (InfoNCE alone fixes only relative order).
            #   * contextual BCE: off-manifold message ⇒ anomalous.
            target = torch.zeros(P, dtype=torch.long, device=device)
            loss = (
                F.cross_entropy(torch.cat([pos_access.unsqueeze(1), neg_access], dim=1), target)
                + F.binary_cross_entropy_with_logits(pos_access, torch.ones_like(pos_access))
                + F.binary_cross_entropy_with_logits(neg_out_ctx, torch.zeros_like(neg_out_ctx))
            )

            def _binding_loss(p_src, p_dst_b, src_rep_b, neg_b, src_list, dst_list):
                """InfoNCE + positive-BCE for one zero-message binding edge group."""
                hist_pos = model.compute_hist_feats(src_list, dst_list, device)
                hist_neg = model.compute_hist_feats(src_rep_b.tolist(), neg_b.tolist(), device)
                pos = _edge_logits(p_src, p_dst_b, p_t, zeros_msg, hist_pos)
                neg = _edge_logits(src_rep_b, neg_b, tv_rep, zeros_msg_rep, hist_neg).view(P, K)
                return (
                    F.cross_entropy(torch.cat([pos.unsqueeze(1), neg], dim=1), target)
                    + F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
                )

            if has_bind:  # device → user
                loss = loss + _binding_loss(p_device, p_user, dev_rep, neg_usr, dev_l, u_l)
            if has_config:
                loss = loss + _binding_loss(p_config, p_user, cfg_rep, neg_cusr, cfg_l, u_l)  # config → user
                if has_bind:
                    loss = loss + _binding_loss(p_config, p_device, cfg_rep, neg_cdev, cfg_l, dev_l)  # config → device
                if has_src:
                    loss = loss + _binding_loss(p_source, p_config, src_rep, neg_scfg, src_l, cfg_l)  # source → config
            if has_src and not has_config:  # legacy source → device
                loss = loss + _binding_loss(p_source, p_device, src_rep, neg_dev, src_l, dev_l)

            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            total_loss += batch_loss
            # refresh=False: store the postfix but let the bar's own throttled refresh
            # (mininterval) draw it — otherwise every batch forces a line in non-TTY logs.
            epoch_bar.set_postfix(loss=f"{batch_loss:.4f}", refresh=False)

            # Predict-then-update: commit benign traffic to memory, neighbour store, the
            # recency cache and the interaction-history counters (benign-only — anomalies
            # never enter the baseline, matching the serving anti-poisoning gate). Edge
            # order mirrors serving: source→config, config→device, config→user,
            # device→user, user→resource.
            def _commit_edge(p_src, p_dst_e, e_msg):
                model.memory.update_state(p_src, p_dst_e, p_t, e_msg)
                model.memory.detach()
                model.neighbor_loader.insert(p_src, p_dst_e, p_t, e_msg)

            if has_config and has_src:
                _commit_edge(p_source, p_config, zeros_msg)
            if has_config and has_bind:
                _commit_edge(p_config, p_device, zeros_msg)
            if has_config:
                _commit_edge(p_config, p_user, zeros_msg)
            if has_src and not has_config:
                _commit_edge(p_source, p_device, zeros_msg)
            if has_bind:
                _commit_edge(p_device, p_user, zeros_msg)
            _commit_edge(p_user, p_dst, p_msg)
            for j in range(P):
                u, d, tv_j = u_l[j], dpos_l[j], tv_l[j]
                pairs = [(u, d)]
                if has_config and has_src:
                    pairs.append((src_l[j], cfg_l[j]))
                if has_config and has_bind:
                    pairs.append((cfg_l[j], dev_l[j]))
                if has_config:
                    pairs.append((cfg_l[j], u))
                if has_src and not has_config:
                    pairs.append((src_l[j], dev_l[j]))
                if has_bind:
                    pairs.append((dev_l[j], u))
                for a, b in pairs:
                    model.last_contact[(a, b)] = tv_j
                    model.pair_count[(a, b)] = model.pair_count.get((a, b), 0) + 1
                    model.src_count[a] = model.src_count.get(a, 0) + 1
                if has_bind:
                    # aux (device, resource) habituality counter — no temporal edge.
                    model.pair_count[(dev_l[j], d)] = model.pair_count.get((dev_l[j], d), 0) + 1

        print(f"Epoch {epoch:02d} | Train Loss: {total_loss / max(num_train_batches, 1):.4f}")

    # --- THRESHOLD CALIBRATION (held-out benign slice) -----------------------
    print("\n--- CALIBRAZIONE SOGLIA (su flusso di validazione benigno) ---")
    # The eval replay evolves the runtime trust feature (node_feat[:, 14]); snapshot the
    # post-training node features so calibration does not bleed trust state into the test
    # replay and so re-runs stay idempotent.
    node_feat_post_train = model.node_feat.clone()
    def _slice(arr, lo, hi):
        return arr[lo:hi] if arr is not None else None

    val_scores, val_labels = _replay(
        model, _slice(source_arr, train_end, val_end), _slice(device_arr, train_end, val_end),
        user_arr[train_end:val_end], dst[train_end:val_end], t[train_end:val_end],
        msg[train_end:val_end], y[train_end:val_end], device, gate_by_label=True,
        config_nodes=_slice(config_arr, train_end, val_end),
        batch_size=cfg.eval_batch_size, desc="Calibrazione (replay val)",
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
    if cfg.eval_batch_size > 1:
        print(
            f"\n[!] eval_batch_size={cfg.eval_batch_size} > 1: i punteggi sono "
            f"l'APPROSSIMAZIONE BATCHATA (veloce, ma NON identica al serving sequenziale: "
            f"staleness intra-batch). Per numeri finali / da pubblicare usa eval_batch_size=1.",
            file=sys.stderr,
        )
    _mode = f"batch={cfg.eval_batch_size}" if cfg.eval_batch_size > 1 else "per-evento"
    print(f"\n--- INIZIO FASE DI INFERENZA / ANOMALY DETECTION ({_mode}) ---")
    # Memory + neighbour history legitimately continue from the (benign) calibration
    # slice, but reset the runtime trust feature to the post-training snapshot so the
    # test stream is not pre-conditioned by calibration.
    model.node_feat.copy_(node_feat_post_train)
    model.recent_alert.clear()  # don't let calibration-slice alerts pre-condition the test stream
    test_scores, test_labels = _replay(
        model, _slice(source_arr, val_end, n), _slice(device_arr, val_end, n),
        user_arr[val_end:], dst[val_end:], t[val_end:], msg[val_end:], y[val_end:],
        device, config_nodes=_slice(config_arr, val_end, n),
        threshold=threshold, threshold_dirty=threshold_dirty, gate_by_label=False,
        batch_size=cfg.eval_batch_size, desc="Inferenza (replay test)",
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
    vs_rule = {1: "OPA-owned  ", 2: "rule-trivial", 3: "rule-blind ", 4: "rule-blind "}
    per_type = {}
    benign = test_types == 0
    for type_id, name in ((1, "policy"), (2, "contextual"), (3, "lateral"), (4, "cred-theft")):
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
    actor_arr = device_arr if device_arr is not None else user_arr
    src_seen_test = causal_src_seen(actor_arr.numpy(), y.numpy())[val_end:]
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

    # --- SCENARIO-LEVEL EVALUATION (the v2-schema goals) ----------------------
    # (a) roaming / wiped-cookie benign events must not become false positives;
    # (b) credential theft (new IP + new device on a known user) must be caught;
    # (c) lateral movement on shared machines must not be diluted by the user split.
    scenario_metrics = {}
    if scenario is not None:
        scen_test = scenario[val_end:].numpy()
        benign_t = test_labels == 0

        def _fpr(mask):
            return float(test_preds[mask].mean()) if mask.any() else float("nan")

        plain = benign_t & (scen_test == 0)
        roam = benign_t & ((scen_test & SCEN_ROAMING) != 0)
        wiped = benign_t & ((scen_test & SCEN_WIPED) != 0)
        scenario_metrics["fpr_plain"] = _fpr(plain)
        scenario_metrics["fpr_roaming"] = _fpr(roam)
        scenario_metrics["fpr_wiped"] = _fpr(wiped)
        scenario_metrics["n_roaming"] = int(roam.sum())
        scenario_metrics["n_wiped"] = int(wiped.sum())

        theft = test_types == 4
        if theft.any():
            scenario_metrics["theft_recall"] = float(test_preds[theft].mean())
            sel = benign | theft
            scenario_metrics["theft_auc"] = float(
                roc_auc_score((test_types[sel] == 4).astype(int), test_scores[sel])
            )
            scenario_metrics["n_theft"] = int(theft.sum())

        shared_lat = (test_types == 3) & ((scen_test & SCEN_SHARED) != 0)
        if shared_lat.any():
            scenario_metrics["shared_lateral_recall"] = float(test_preds[shared_lat].mean())
            scenario_metrics["n_shared_lateral"] = int(shared_lat.sum())

        print("\n--- SCENARI v2 (goal a/b/c) ---")
        print(f"  benign FPR  plain={scenario_metrics['fpr_plain']:.4f} | "
              f"roaming={scenario_metrics['fpr_roaming']:.4f} (n={scenario_metrics['n_roaming']}) | "
              f"wiped-cookie={scenario_metrics['fpr_wiped']:.4f} (n={scenario_metrics['n_wiped']})")
        if "theft_recall" in scenario_metrics:
            print(f"  credential theft: recall@thr={scenario_metrics['theft_recall']:.4f} | "
                  f"AUC={scenario_metrics['theft_auc']:.4f} (n={scenario_metrics['n_theft']})")
        if "shared_lateral_recall" in scenario_metrics:
            print(f"  lateral su device condivisi: recall@thr="
                  f"{scenario_metrics['shared_lateral_recall']:.4f} "
                  f"(n={scenario_metrics['n_shared_lateral']})")

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
            "schema_version": cfg.schema_version,
            "capacity": capacity,
            "node_feat_dim": cfg.node_feat_dim,
            "msg_dim": cfg.msg_dim,
            "memory_dim": cfg.memory_dim,
            "time_dim": cfg.time_dim,
            "num_hops": cfg.num_hops,
            "gnn_heads": cfg.gnn_heads,
            "link_pred_hidden_layers": cfg.link_pred_hidden_layers,
            "hash_buckets": cfg.hash_buckets,
            "hash_dim": cfg.hash_dim,
            "hist_feat_dim": cfg.hist_feat_dim,
            "neighbor_size": cfg.neighbor_size,
            "target_fpr": cfg.target_fpr,
            "cost_ratio": cfg.cost_ratio,
            "clean_fpr_cap": cfg.clean_fpr_cap,
            "precursor_half_life": cfg.precursor_half_life,
            "precursor_max_boost": cfg.precursor_max_boost,
            "use_resource_risk": cfg.use_resource_risk,
            "use_source_internal": cfg.use_source_internal,
            "guest_device_fallback": cfg.guest_device_fallback,
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
        "scenario": scenario_metrics,
        "use_struct_head": use_struct_head,
        "use_hash_identity": use_hash_identity,
        "use_hist_feats": use_hist_feats,
        "use_precursor": use_precursor,
        "use_config_node": use_config_node,
    }


if __name__ == "__main__":
    train_tgn()
