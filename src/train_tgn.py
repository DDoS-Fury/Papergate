"""Train and evaluate the streaming Temporal Graph Network.

Pipeline:
  1. Generate a chronologically ordered synthetic ZTA access stream.
  2. Split it by time into train (70%) / val (10%) / test (20%).
  3. Train ONE-CLASS on benign traffic with structural + contextual negatives;
     memory is updated with benign events only.
  4. Calibrate the anomaly threshold on a held-out benign slice at ``target_fpr``.
  5. Evaluate on the test stream **event-by-event**, reproducing the serving flow.
  6. Persist the deployable artifact (weights + memory + registry + threshold).

Supervision regime: the method is **one-class / semi-supervised**, not
unsupervised:

  * the training set is *selected* by ground-truth labels (``benign_mask = b_y == 0``),
    so the model never sees an attack but does rely on labels to know what to skip;
  * the primary operating threshold is fitted with ground-truth lateral-movement labels
    on the validation slice (``cost_sensitive_threshold``), which means a deployment
    needs labelled red-team data in its calibration window. The unsupervised
    benign-quantile threshold (``threshold_dirty``) is reported alongside it.

No attack label is ever used to compute a test-set score.

Memory commit gate (offline replays): an event is committed when the deterministic signal
layer does not flag it, a proxy for OPA ALLOW (``serve_tgn.commit_event`` runs only on
ALLOW). Gating on the model's own score would starve false-positive benign events of
updates and inflate the FPR; committing every signal-clean event, laterals included, keeps
the reported FPR conservative.
"""

import copy
import os
import random
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
from scipy.special import logit as _logit, expit as expit_np

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


from graphagate.calibration import (
    cost_sensitive_threshold,
    operating_point,
    recall_fpr_curve,
    routed_predict,
)
from graphagate.config import TGNConfig, TGN_CHECKPOINT_PATH, TGN_STATS_PATH
from graphagate.data.stream_synthetic import (
    SCEN_NEW_USER,
    SCEN_ROAMING,
    SCEN_SHARED,
    SCEN_WIPED,
    generate_streaming_data,
    stream_kwargs_from_cfg,
)
from graphagate.eval_common import binary_metrics, causal_src_seen, incident_report
from graphagate.model.registry import NodeRegistry
from graphagate.model.tgn import ZTATemporalGraphNetwork, stable_hash
from graphagate.serve_tgn import (
    EDGE_ACCESS,
    EDGE_DEV_USER,
    EDGE_CFG_USER,
    EDGE_CFG_DEV,
    EDGE_SRC_CFG,
    EDGE_SRC_DEV,
    anomaly_score,
    chain_edge_logits,
    combine_edge_logits,
    fit_edge_calibration,
    event_alarm,
    precursor_shift,
    record_alert,
    save_model,
    sensor_alarm,
    set_edge_calibration,
    signal_dirty,
)


def _pbar(iterable=None, *, total=None, desc=None):
    """tqdm bar; without a TTY (docker logs, CI) it prints one ASCII line every ~10 s."""
    is_tty = sys.stderr.isatty()
    return tqdm(
        iterable, total=total, desc=desc, disable=False,
        mininterval=(0.5 if is_tty else 10.0), dynamic_ncols=True, ascii=not is_tty,
    )


def fit_thresholds(scores, labels, v_types, v_msg, cfg):
    """``(threshold_clean, threshold_dirty, threshold_clean_unsup, benign_scores)`` from the
    scores of one validation replay (pure function of its inputs)."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    benign = scores[labels == 0]
    if benign.size == 0:
        raise RuntimeError("No benign events in the validation slice for calibration.")
    v_clean = ~_rule_baseline(v_msg).astype(bool)
    clean_benign = scores[v_clean & (labels == 0)]
    dirty_benign = scores[(~v_clean) & (labels == 0)]

    # Label-free alternative to t_clean (benign quantile over signal-clean events), stored
    # in the calibration metadata for deployments without red-team labels.
    t_unsup = float(np.quantile(clean_benign, 1.0 - cfg.target_fpr)) if clean_benign.size \
        else float(np.quantile(benign, 1.0 - cfg.target_fpr))

    # Clean threshold: cost-sensitive on lateral movement within the clean stream.
    mask = v_clean & ((labels == 0) | (v_types == 3))
    cal_labels_ = (v_types[mask] == 3).astype(int)
    if cal_labels_.sum() == 0:
        # No lateral example in the window: fall back to the FPR threshold.
        t_clean = t_unsup
    else:
        t_clean = cost_sensitive_threshold(
            scores[mask], cal_labels_, cost_ratio=cfg.cost_ratio,
            target_fpr_cap=cfg.clean_fpr_cap,
        )

    # Dirty threshold: cost-sensitive on contextual anomalies if present, else dirty benign quantile.
    mask_dirty = (~v_clean) & ((labels == 0) | (v_types == 2))
    cal_labels_dirty = (v_types[mask_dirty] == 2).astype(int)
    if cal_labels_dirty.sum() > 0:
        t_dirty = cost_sensitive_threshold(
            scores[mask_dirty], cal_labels_dirty, cost_ratio=cfg.cost_ratio,
            target_fpr_cap=cfg.dirty_fpr_cap,
        )
    elif dirty_benign.size > 0:
        t_dirty = float(np.quantile(dirty_benign, 1.0 - cfg.target_fpr))
    else:
        t_dirty = t_unsup

    return t_clean, t_dirty, t_unsup, benign


def _replay(model, source_nodes, device_nodes, user, dst, t, msg, y, device, *,
            config_nodes=None, threshold=None, threshold_dirty=None, threshold_arm=None,
            gate_by_label=False, batch_size=1, desc="replay", return_edge_logits=False):
    """Offline replay mirroring the serving path; returns ``(scores, labels)``.

    ``source_nodes`` / ``config_nodes`` / ``device_nodes`` may be ``None`` (dataset without
    that entity, e.g. LANL): their edges are skipped, as :func:`serve_tgn.score_event` does
    when the key is omitted.

    Memory commit gate:
      * ``gate_by_label=True``: ground-truth benign events (calibration replay);
      * ``gate_by_label=False``: signal-clean events, the OPA-ALLOW proxy of the module
        docstring (never the model's own score).

    Thresholds: with ``threshold_dirty`` the decision is signal-routed (signal-dirty events
    use ``threshold_dirty``, the rest ``threshold``); without it ``threshold`` applies to all.
    ``threshold_arm`` is the lower threshold on which the kill-chain precursor arms
    (:func:`serve_tgn.event_alarm`); ``None`` arms only on a flag or a sensor. Arming feeds
    the precursor prior only; node features are never mutated.

    ``batch_size=1`` is the exact per-event loop; ``B>1`` scores a block against the
    start-of-batch memory and commits afterwards (offline only: serving stays sequential).
    The precursor/decision feedback is sequential in both cases.

    ``return_edge_logits=True`` also returns ``{"edge_logits": {kind: [N]}, "shift": [N]}``,
    the inputs of the per-edge benign calibration.
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
    # Mirrors score_event: a missing device skips only the device's own edges
    # (config→device, device→user); the source→config and config→user edges stay.
    has_src = source_nodes is not None
    has_config = config_nodes is not None
    msg_dim = msg.shape[1]

    def _bump(a, b, tv):
        """Advance the benign-gated recency / interaction-history counters (cf. update_memory)."""
        model.last_contact[(a, b)] = tv
        model.pair_count[(a, b)] = model.pair_count.get((a, b), 0) + 1
        model.src_count[a] = model.src_count.get(a, 0) + 1

    edge_rec = {} if return_edge_logits else None
    shift_rec = np.empty(N, dtype=np.float64) if return_edge_logits else None

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

            # --- Phase 1: one neighbour expansion + GNN forward for the whole block, scored per
            # edge group; the event's anomaly logit is the max over its (benign-calibrated) edge
            # logits (serve_tgn.combine_edge_logits).
            groups = [(EDGE_ACCESS, bu, bd, bmsg, bdev)]  # access edge (aux src = device)
            if has_bind:
                groups.append((EDGE_DEV_USER, bdev, bu, zeros_msg, None))
            if has_config:
                groups.append((EDGE_CFG_USER, bcfg, bu, zeros_msg, None))
                if has_bind:
                    groups.append((EDGE_CFG_DEV, bcfg, bdev, zeros_msg, None))
                if has_src:
                    groups.append((EDGE_SRC_CFG, bsrc, bcfg, zeros_msg, None))
            if has_src and has_bind and not has_config:
                groups.append((EDGE_SRC_DEV, bsrc, bdev, zeros_msg, None))  # config-node ablation
            edge_logits = chain_edge_logits(model, groups, bt, device)
            if edge_rec is not None:
                for kind, v in edge_logits.items():
                    edge_rec.setdefault(kind, np.empty(N, dtype=np.float64))[start:end] = (
                        v.detach().cpu().numpy()
                    )
            raw = combine_edge_logits(model, edge_logits)
            raw_np = raw.detach().cpu().numpy()
            bmsg_rows = bmsg.tolist()

            # --- Phase 2: sequential per-event feedback (precursor / decision).
            do_update = np.zeros(B, dtype=bool)
            for j in range(B):
                i = start + j
                lab = y_l[i]
                u = us[j]
                tv = ts[j]
                actor = devs[j] if has_bind else u
                shift = precursor_shift(model, actor, tv)
                if shift_rec is not None:
                    shift_rec[i] = shift
                score = float(anomaly_score(raw_np[j] + shift))
                scores[i] = score

                eff_thr = threshold
                msg_row = bmsg_rows[j]
                if threshold_dirty is not None and signal_dirty(msg_row):
                    eff_thr = threshold_dirty
                # Commit gate: OPA-ALLOW proxy (signal-clean), not the model's own score.
                do_update[j] = (lab == 0) if gate_by_label else (not signal_dirty(msg_row))

                # Calibration pass: no threshold yet, the label or the sensor arms the precursor.
                if gate_by_label:
                    alarm = lab == 1 or sensor_alarm(msg_row)
                else:
                    # Also arm on threshold_arm: recon sits below the decision threshold.
                    alarm = event_alarm(score, flagged=eff_thr is not None and score >= eff_thr,
                                        threshold_arm=threshold_arm, features=msg_row)
                if alarm:
                    record_alert(model, actor, tv)  # arm the precursor (recon → lateral)

            # --- Phase 3: batched commit of the gated events (memory, neighbours, counters).
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
                if has_src and has_bind and not has_config:
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
                    if has_src and has_bind and not has_config:
                        _bump(srcs[j], devs[j], tv)
                    if has_bind:
                        _bump(devs[j], u, tv)
                    _bump(u, d, tv)
                    if has_bind:  # aux (device, resource) habituality counter (no temporal edge)
                        dvd = (devs[j], d)
                        model.pair_count[dvd] = model.pair_count.get(dvd, 0) + 1

        pbar.update(B)
    pbar.close()
    if return_edge_logits:
        return scores, labels, {"edge_logits": edge_rec, "shift": shift_rec}
    return scores, labels


def _rule_baseline(test_msg):
    """Signal-only detector: 1 if TLS trust is broken (``ja3 == 0``) or a sensor s1-s3 fires.

    Column 4 is the HTTP method, not a sensor. Catches contextual anomalies and misses the
    signal-clean ones (policy, lateral, theft): the floor the TGN must beat.
    """
    return (
        (test_msg[:, 0] == 0.0)
        | (test_msg[:, 1] == 1.0)
        | (test_msg[:, 2] == 1.0)
        | (test_msg[:, 3] == 1.0)
    ).astype(int)

def precursor_report(tag, scores, labels, types, msg, shift, thr_clean, thr_dirty, armed_nats=0.5):
    """Routed FPR / recall split by whether the kill-chain precursor shifted the event, and
    the same decision with the shift removed (counterfactual, same thresholds).

    "Armed" = shift >= ``armed_nats``: the decay never reaches 0, so ``shift > 0`` would count
    every entity that ever alerted (0.5 nats = 3 half-lives after a full 4-nat alert)."""
    shift = np.asarray(shift, dtype=np.float64)
    dirty = _rule_baseline(msg).astype(bool)
    pred = routed_predict(scores, dirty, thr_clean, thr_dirty).astype(bool)
    pred_ns = routed_predict(
        expit_np(_logit(np.asarray(scores, dtype=np.float64)) - shift),
        dirty, thr_clean, thr_dirty,
    ).astype(bool)
    armed = shift >= armed_nats
    benign = types == 0
    fp = pred & benign

    def _rate(p, m):
        return float(p[m].mean()) if m.any() else float("nan")

    print(f"\n--- PRECURSOR DIAGNOSTIC ({tag}, armed = shift >= {armed_nats} nats) ---")
    print(f"  benign FPR: with shift={_rate(pred, benign):.4f} | without={_rate(pred_ns, benign):.4f}")
    for name, m in (("armed", armed), ("unarmed", ~armed)):
        print(f"  benign {name:8s}: n={int((benign & m).sum()):6d} | FPR={_rate(pred, benign & m):.4f}")
    print(f"  share of benign FPs on armed events: {fp[armed].sum() / max(int(fp.sum()), 1):.3f}")
    for ty, name in ((3, "lateral"), (4, "cred-theft")):
        sel = types == ty
        if not sel.any():
            continue
        print(f"  {name:10s}: recall with shift={_rate(pred, sel):.4f} | without={_rate(pred_ns, sel):.4f} "
              f"| armed {int((sel & armed).sum())}/{int(sel.sum())}")


def combine_report(model, val_extra, val_types, val_msg, test_extra, test_types, target_fpr):
    """Offline comparison of the edge-combination rules on the recorded edge logits.

    Same arming as the run (shifts are reused, not replayed): a first-order comparison.
    Works on logits (comb + shift), not on expit scores, which saturate at 1.0.
    """
    prev = getattr(model, "edge_combine", "max")
    clean_benign_v = (val_types == 0) & ~_rule_baseline(val_msg).astype(bool)
    print("\n--- EDGE COMBINATION (offline, recorded logits) ---")
    for mode in ("max", "fisher"):
        model.edge_combine = mode

        def _lg(extra):
            comb = combine_edge_logits(model, {
                k: torch.as_tensor(v) for k, v in extra["edge_logits"].items()
            }).numpy()
            return comb + extra["shift"]

        lv, lt = _lg(val_extra), _lg(test_extra)
        thr = float(np.quantile(lv[clean_benign_v], 1.0 - target_fpr))
        benign = test_types == 0
        parts = []
        for ty, name in ((3, "lateral"), (4, "theft")):
            pos = test_types == ty
            if not pos.any():
                continue
            sel = benign | pos
            lab = pos[sel].astype(int)
            parts.append(f"{name} AUC={roc_auc_score(lab, lt[sel]):.4f} "
                         f"AP={average_precision_score(lab, lt[sel]):.4f} "
                         f"R={float((lt[pos] >= thr).mean()):.4f}")
        print(f"  {mode:6s} | " + " | ".join(parts)
              + f" | benign FPR={float((lt[benign] >= thr).mean()):.4f} (val clean @{target_fpr})")
    model.edge_combine = prev

def _sample_structural_negatives(num_events, num_res, res_lo, device, *, avoid=None, hard_pool=None, hard_ratio=0.4):
    """InfoNCE negatives: uniform over ``[res_lo, res_lo + num_res)``, a ``hard_ratio`` share
    drawn from ``hard_pool`` (the batch's endpoints: frequency-weighted, no labels).

    The hard draws fill the *prefix* of the flat ``P*K`` array (repeat_interleave layout), so
    the first ``hard_ratio`` of the positives get only in-batch negatives and the rest only
    uniform ones. Kept as is: the reference full run was trained this way, and random
    placement (with the other negative changes) did not reproduce it (tasks/todo.md).
    ``num_res == 0`` (no known id range) draws everything from the pool. Draws equal to
    ``avoid`` are re-rolled once uniformly.
    """
    if num_res <= 0:
        return hard_pool[torch.randint(0, len(hard_pool), (num_events,), device=device)]
    neg = torch.randint(0, num_res, (num_events,), device=device) + res_lo

    if hard_pool is not None and len(hard_pool) > 1 and hard_ratio > 0.0:
        num_hard = int(num_events * hard_ratio)
        if num_hard > 0:
            neg[:num_hard] = hard_pool[torch.randint(0, len(hard_pool), (num_hard,), device=device)]

    if avoid is not None:
        collide = neg == avoid
        if collide.any():
            neg[collide] = torch.randint(0, num_res, (int(collide.sum()),), device=device) + res_lo

    return neg


def _uniformity_loss(x, t: float = 2.0) -> torch.Tensor:
    """Wang & Isola (2020) uniformity on the hypersphere: log E[exp(-t ||x - y||^2)].
    Pushes projected node embeddings to distribute uniformly over the hypersphere,
    preventing representation collapse onto a single directional mode.
    """
    if x.size(0) <= 1:
        return torch.tensor(0.0, device=x.device)
    pdist_sq = torch.pdist(x, p=2).pow(2)
    return pdist_sq.mul(-t).exp().mean().log()


# def _sample_structural_negatives(num_events, num_res, res_lo, device, *, avoid=None):
#     """Uniformly random resource destinations (standard temporal link-prediction negatives).
#
#     Uses only the resource id-range, never the generator's habitual/authorization sets, so
#     the objective learns each entity's access distribution rather than the evaluation's
#     anomaly definition.
#     """
#     neg = torch.randint(0, num_res, (num_events,), device=device) + res_lo
#     if avoid is not None:
#         # Re-roll draws equal to the true dst (false negatives); once suffices at num_res >> 1.
#         collide = neg == avoid
#         if collide.any():
#             neg[collide] = (
#                 torch.randint(0, num_res, (int(collide.sum()),), device=device) + res_lo
#             )
#     return neg


@dataclass
class StreamData:
    """A ZTA access stream for :func:`train_tgn`, decoupled from the generator (e.g. LANL).

    Node ids are ``0..num_nodes-1``; ``keys[i]`` is slot ``i``'s registry key (hashed-identity
    embedding). Access-edge negatives are drawn from ``[neg_lo, neg_lo + neg_num)``.
    ``source_nodes`` / ``config_nodes`` / ``device_nodes`` are optional: ``None`` drops their
    edges and objectives (LANL keeps only ``user -> dst``). The ``usr_*`` / ``dev_*`` /
    ``cfg_*`` / ``src_*`` ranges drive the binding-edge negatives (``num == 0``: in-batch only). ``scenario`` is the synthetic
    benign-context bitmask (``None`` skips the scenario evals).
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
    src_lo: int = 0
    src_num: int = 0
    # Evaluation-only ground truth (synthetic stream): see SyntheticStream.incident.
    incident: torch.Tensor | None = None
    theft_variant: torch.Tensor | None = None


def stream_to_data(s) -> StreamData:
    """Wrap a :class:`SyntheticStream` as :class:`StreamData`.

    Lets a caller pass an already built or reshaped stream (e.g. ``eval_common.tail_stream``)
    to ``train_tgn(dataset=...)``.
    """
    return StreamData(
        user=s.user, dst=s.dst, t=s.t, msg=s.msg, y=s.y, types=s.types,
        node_features=s.node_features, keys=s.keys, num_nodes=s.num_nodes,
        neg_lo=s.res_lo, neg_num=s.res_num,
        device_nodes=s.device, source_nodes=s.source, config_nodes=s.config,
        scenario=s.scenario,
        usr_lo=s.user_lo, usr_num=s.user_num, dev_lo=s.dev_lo, dev_num=s.dev_num,
        cfg_lo=s.cfg_lo, cfg_num=s.cfg_num, src_lo=s.src_lo, src_num=s.src_num,
        incident=s.incident, theft_variant=s.theft_variant,
    )


def _synthetic_stream_data(cfg: TGNConfig) -> StreamData:
    """:class:`StreamData` from the synthetic generator configured by ``cfg`` (default path)."""
    return stream_to_data(generate_streaming_data(**stream_kwargs_from_cfg(cfg)))


def train_tgn(cfg: TGNConfig | None = None, *, dataset: "StreamData | None" = None,
              use_struct_head=True, use_hash_identity=True, use_hist_feats=True,
              use_precursor=True, use_config_node=True, save=True, return_scores=False,
              log_dir: str | None = None):
    """Train + evaluate the streaming TGN; returns a metrics dict.

    Keyword flags drive the ablations (``tests/ablations``): ``use_struct_head``,
    ``use_hash_identity``, ``use_hist_feats`` and ``use_precursor`` toggle model components;
    ``use_config_node=False`` drops the config node, so the same stream runs on the
    ``source -> device`` chain. ``save=False`` skips persisting the artifact (ablations must
    not overwrite the full-model checkpoint). ``dataset`` injects an external
    :class:`StreamData` (e.g. LANL) instead of the synthetic generator.
    """
    if cfg is None:
        cfg = TGNConfig()

    writer = None
    if SummaryWriter is not None:
        if log_dir is None:
            tag = "synthetic" if dataset is None else "custom"
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join("runs", f"tgn_{tag}_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"[tensorboard] Logging to: {log_dir}")

    # Full seeding. The scatter-add in TransformerConv / TGNMemory has no deterministic CUDA
    # kernel (hence warn_only): residual run-to-run noise must be measured by repeated runs.
    # Set CUBLAS_WORKSPACE_CONFIG=:4096:8 for cuBLAS determinism.
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if dataset is None:
        print("Generating streaming data...")
        data = _synthetic_stream_data(cfg)
    else:
        print("Using injected dataset stream...")
        data = dataset
    user_arr, dst, t, msg, y, types = data.user, data.dst, data.t, data.msg, data.y, data.types
    device_arr, source_arr, scenario = data.device_nodes, data.source_nodes, data.scenario
    config_arr = data.config_nodes
    # Config-node ablation: the whole pipeline falls back to the source→device chain.
    if not use_config_node:
        config_arr = None
    node_features = data.node_features
    total_nodes = data.num_nodes
    capacity = total_nodes + cfg.capacity_headroom
    neg_lo, neg_num = data.neg_lo, data.neg_num

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Entity registry: ``data.keys[i]`` is the external key of slot ``i``.
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

    # Static node features of the preregistered entities; slots for entities first seen
    # at serving time stay zero until they supply their own.
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
    # Kill-chain precursor knobs (serving-time prior; see serve_tgn.precursor_shift).
    model.precursor_half_life = cfg.precursor_half_life
    model.precursor_max_shift = cfg.precursor_max_shift
    model.edge_combine = cfg.edge_combine
    if not (use_struct_head and use_hash_identity and use_hist_feats and use_precursor):
        print(f"[ablation] use_struct_head={use_struct_head} use_hash_identity={use_hash_identity} "
              f"use_hist_feats={use_hist_feats} use_precursor={use_precursor}")

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-5)

    # Chronological split (the stream is already time-ordered).
    n = len(dst)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)
    train_end, val_end = n_train, n_train + n_val
    bs = cfg.batch_size

    K = cfg.infonce_k
    # Node-id ranges (lo, num) of the binding endpoints; num == 0 → in-batch draws only.
    usr_rng, dev_rng = (data.usr_lo, data.usr_num), (data.dev_lo, data.dev_num)
    cfg_rng, src_rng = (data.cfg_lo, data.cfg_num), (data.src_lo, data.src_num)

    def _batch_loss(i: int, *, uniformity: bool):
        """Self-supervised loss of train batch ``i`` and a closure committing its benign events.

        Shared by Stage 1 and the Stage 2 MLP fine-tuning. The closure is called after the
        optimiser step (predict-then-update). Returns ``None`` for a batch without benign events.
        """
        start_idx, end_idx = i * bs, i * bs + bs

        b_user = user_arr[start_idx:end_idx].to(device)
        b_dst = dst[start_idx:end_idx].to(device)
        b_t = t[start_idx:end_idx].to(device)
        b_msg = msg[start_idx:end_idx].to(device)
        b_y = y[start_idx:end_idx].to(device)
        b_device = device_arr[start_idx:end_idx].to(device) if device_arr is not None else None
        b_source = source_arr[start_idx:end_idx].to(device) if source_arr is not None else None
        b_config = config_arr[start_idx:end_idx].to(device) if config_arr is not None else None

        # Trust (node_feat[:, 14]) stays at the neutral 1.0: never derived from labels.
        benign_mask = b_y == 0
        if not benign_mask.any():
            return None

        p_user = b_user[benign_mask]
        p_dst = b_dst[benign_mask]
        p_t = b_t[benign_mask]
        p_msg = b_msg[benign_mask]
        p_device = b_device[benign_mask] if b_device is not None else None
        p_source = b_source[benign_mask] if b_source is not None else None
        p_config = b_config[benign_mask] if b_config is not None else None

        # Structural negatives, K per positive and per edge:
        #   user→resource   random resources  (habitual accesses: lateral movement)
        #   device→user     other users  | other devices  (hosted users | a thief's machine)
        #   config→user     other users  | other configs  (a thief's client differs from the victim's)
        #   config→device   other devices | other configs (a new tool on a known device)
        #   source→config   other configs | other sources (clients behind an IP; roaming = tolerance)
        # Binding edges corrupt the tail (P(tail | head)) and, with head_negatives, the head
        # (P(head | tail)): credential theft keeps the victim and swaps in the attacker's
        # device / client / IP, i.e. a head corruption. In-batch draws (binding_hard_ratio)
        # are frequency-weighted, like a mimicked popular fleet client.
        # The bindings are required: a thief reaching the victim's habitual resources
        # looks benign on the access edge; the anomaly lives in the bindings.
        P = len(p_user)
        neg_res = _sample_structural_negatives(
            P * K, neg_num, neg_lo, device, avoid=p_dst.repeat_interleave(K),
            hard_pool=p_dst, hard_ratio=0.25
        )
        user_rep = p_user.repeat_interleave(K)

        has_bind = p_device is not None and data.dev_num > 0
        # A missing device skips only the device's own edges (score_event parity).
        has_src = p_source is not None
        has_config = p_config is not None and data.cfg_num > 0

        def _corrupt(pos, rng):
            return _sample_structural_negatives(
                P * K, rng[1], rng[0], device, avoid=pos.repeat_interleave(K),
                hard_pool=pos, hard_ratio=cfg.binding_hard_ratio,
            )

        # (head, tail, tail negatives, head negatives | None), in the loss order.
        bind_edges = []

        def _bind(head, tail, h_rng, t_rng):
            neg_h = _corrupt(head, h_rng) if cfg.head_negatives else None
            bind_edges.append((head, tail, _corrupt(tail, t_rng), neg_h))

        if has_bind:
            _bind(p_device, p_user, dev_rng, usr_rng)  # device → user
        if has_config:
            _bind(p_config, p_user, cfg_rng, usr_rng)  # config → user
            if has_bind:
                _bind(p_config, p_device, cfg_rng, dev_rng)  # config → device
            if has_src:
                _bind(p_source, p_config, src_rng, cfg_rng)  # source → config
        if has_src and has_bind and not has_config:
            _bind(p_source, p_device, src_rng, dev_rng)  # source → device (config-node ablation)

        # Expand every involved node to its stored temporal neighbourhood and embed
        # once; the heads below differ only in which endpoints / message they score,
        # sharing the same history-conditioned embeddings.
        parts = [p_user, p_dst, neg_res] + [x for e in bind_edges for x in e if x is not None]
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
            d_pair = model.pair_delta_t(s_list, d_list, t_list, device)
            d_src = model.src_delta_t(src_nodes, t_nodes, device)
            return model.score(
                z, nf, h_idx, assoc[src_nodes], assoc[dst_nodes], msgs, d_pair, d_src, hist_feats,
            )

        def _neg_logits(src_b, dst_b, msgs, hist_feats, is_true):
            """(P, K) negative logits. With ``mask_seen_negatives``, draws that are no
            negatives leave the softmax (-inf): the true endpoint (a collision the re-roll
            missed), or a pair already in the benign history (a shared device, a user's second
            machine), read from the model's own counters, never from labels."""
            logits = _edge_logits(src_b, dst_b, tv_rep, msgs, hist_feats)
            return _mask_negatives(logits, src_b, dst_b, is_true).view(P, K)

        def _mask_negatives(logits, src_b, dst_b, is_true):
            if not cfg.mask_seen_negatives:
                return logits
            seen = torch.tensor(
                [model.pair_count.get(pr, 0) > 0 for pr in zip(src_b.tolist(), dst_b.tolist())],
                dtype=torch.bool, device=device,
            )
            return logits.masked_fill(seen | is_true, float("-inf"))

        target = torch.zeros(P, dtype=torch.long, device=device)

        def _infonce(pos, neg):
            return F.cross_entropy(torch.cat([pos.unsqueeze(1), neg], dim=1), target)

        # Stage 1 only: the projector is frozen in Stage 2.
        struct_aux = uniformity and model.use_struct_head and cfg.struct_aux_weight > 0

        def _struct_infonce(head, tail, corruptions):
            """Auxiliary InfoNCE on the structural term alone, over the same corruptions
            ``(heads, tails, is_true)`` as the edge's main InfoNCE."""
            pos = model.struct_logit(z, assoc[head], assoc[tail])
            negs = [
                _mask_negatives(model.struct_logit(z, assoc[h], assoc[t]), h, t, is_true).view(P, K)
                for h, t, is_true in corruptions
            ]
            return cfg.struct_aux_weight * _infonce(pos, torch.cat(negs, dim=1))

        zeros_msg = torch.zeros_like(p_msg)
        zeros_msg_rep = torch.zeros(P * K, p_msg.size(1), device=device)
        msg_rep = p_msg.repeat_interleave(K, dim=0)

        # --- ACCESS EDGE user→resource (full message + device-aux history) ---
        hist_acc_pos = model.compute_hist_feats(u_l, dpos_l, device, aux_src_ids=dev_l)
        hist_acc_neg = model.compute_hist_feats(
            user_rep.tolist(), neg_res.tolist(), device,
            aux_src_ids=p_device.repeat_interleave(K).tolist() if has_bind else None,
        )
        pos_access = _edge_logits(p_user, p_dst, p_t, p_msg, hist_acc_pos)
        neg_access = _neg_logits(
            user_rep, neg_res, msg_rep, hist_acc_neg, neg_res == p_dst.repeat_interleave(K),
        )

        # --- CONTEXTUAL NEGATIVES: Gaussian noise on the message, a different mechanism
        # from the eval's discrete signal flips; keeps the feature head using the message.
        neg_msg = p_msg + torch.randn_like(p_msg) * 0.5
        neg_out_ctx = _edge_logits(p_user, p_dst, p_t, neg_msg, hist_acc_pos)

        # --- SELF-SUPERVISED LOSS ---
        #   * InfoNCE ranking per edge: among {true endpoint, K random alternatives}
        #     the true one must score most-benign given the src's history. AP-aligned
        #     (a relative/soft target, unlike a hard 0/1 negative).
        #   * positive BCE anchors: keep benign logits high so the FPR-calibrated
        #     threshold is meaningful (InfoNCE alone fixes only relative order).
        #   * contextual BCE: off-manifold message ⇒ anomalous.
        loss = (
            _infonce(pos_access, neg_access)
            + F.binary_cross_entropy_with_logits(pos_access, torch.ones_like(pos_access))
            + F.binary_cross_entropy_with_logits(neg_out_ctx, torch.zeros_like(neg_out_ctx))
        )

        if struct_aux:
            loss = loss + _struct_infonce(
                p_user, p_dst, [(user_rep, neg_res, neg_res == p_dst.repeat_interleave(K))],
            )

        # --- UNIFORMITY REGULARIZATION (Wang & Isola, ICML 2020) ---
        # Per node type: on the mixed set the term also repels a user from their resources.
        if uniformity and model.use_struct_head:
            groups = (
                [p_user.unique(), p_dst.unique()] if cfg.uniformity_per_type
                else [torch.cat([p_user, p_dst]).unique()]
            )
            unif = sum(
                _uniformity_loss(F.normalize(model.struct_proj(z[assoc[g]]), dim=-1)) for g in groups
            )
            loss = loss + 0.05 * unif / len(groups)

        def _binding_loss(head, tail, neg_t, neg_h):
            """One InfoNCE per zero-message binding edge — the true pair above its tail and
            head corruptions in a single softmax (same per-edge weight as tail-only) — plus
            the positive BCE."""
            h_l, t_l = head.tolist(), tail.tolist()
            pos = _edge_logits(head, tail, p_t, zeros_msg, model.compute_hist_feats(h_l, t_l, device))
            h_rep, t_rep = head.repeat_interleave(K), tail.repeat_interleave(K)
            corruptions = [(h_rep, neg_t, neg_t == t_rep)]
            if neg_h is not None:
                corruptions.append((neg_h, t_rep, neg_h == h_rep))
            negs = [
                _neg_logits(h, t, zeros_msg_rep, model.compute_hist_feats(h.tolist(), t.tolist(), device), is_true)
                for h, t, is_true in corruptions
            ]
            edge_loss = (
                _infonce(pos, torch.cat(negs, dim=1))
                + F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
            )
            if struct_aux:
                edge_loss = edge_loss + _struct_infonce(head, tail, corruptions)
            return edge_loss

        for head, tail, neg_t, neg_h in bind_edges:
            loss = loss + _binding_loss(head, tail, neg_t, neg_h)

        def commit():
            """Predict-then-update: commit the benign events (memory, neighbours, recency,
            counters) in serving edge order: source→config, config→device, config→user,
            device→user, user→resource."""
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
            if has_src and has_bind and not has_config:
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
                if has_src and has_bind and not has_config:
                    pairs.append((src_l[j], dev_l[j]))
                if has_bind:
                    pairs.append((dev_l[j], u))
                for a, b in pairs:
                    model.last_contact[(a, b)] = tv_j
                    model.pair_count[(a, b)] = model.pair_count.get((a, b), 0) + 1
                    model.src_count[a] = model.src_count.get(a, 0) + 1
                if has_bind:
                    # aux (device, resource) habituality counter -> no temporal edge.
                    model.pair_count[(dev_l[j], d)] = model.pair_count.get((dev_l[j], d), 0) + 1

        return loss, commit

    # One-class: labels select the (benign) training set; see the module docstring.
    print("--- ONE-CLASS TRAINING START (benign traffic only) ---")
    _t_train0 = time.perf_counter()  # wall time of the gradient loop only (no calibration / replay)
    global_step = 0
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
            out = _batch_loss(i, uniformity=True)
            if out is None:
                continue
            loss, commit = out
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            total_loss += batch_loss
            global_step += 1
            if writer is not None and (global_step % 10 == 0 or i == num_train_batches - 1):
                writer.add_scalar("Train/Batch_Loss", batch_loss, global_step)
            # refresh=False: store the postfix but let the bar's own throttled refresh
            # (mininterval) draw it — otherwise every batch forces a line in non-TTY logs.
            epoch_bar.set_postfix(loss=f"{batch_loss:.4f}", refresh=False)
            commit()

        epoch_loss = total_loss / max(num_train_batches, 1)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:02d} | Train Loss: {epoch_loss:.4f} | LR: {current_lr:.6f}")
        if writer is not None:
            writer.add_scalar("Train/Epoch_Loss", epoch_loss, epoch)
            writer.add_scalar("Train/LR", current_lr, epoch)
        scheduler.step() # learning rate update using CosineAnnealingLR fun
    train_seconds = time.perf_counter() - _t_train0

    # TODO: checks that model improves with new settings
    #print(f"Epoch {epoch:02d} | Train Loss: {total_loss / max(num_train_batches, 1):.4f}")
    #train_seconds = time.perf_counter() - _t_train0

    # ============= FineTuning ==============
    if getattr(cfg, "ft_epochs", 0) > 0:
        print(f"\n--- TWO-STAGE TRAINING: FASE 2 (Finetuning MLP per {cfg.ft_epochs} epoche, GNN congelata) ---")

        # Freezing layers
        model.gnn.requires_grad_(False)
        model.memory.requires_grad_(False)
        model.hash_emb.requires_grad_(False)

        if hasattr(model, "struct_proj"):
            model.struct_proj.requires_grad_(False)

        model.link_pred.requires_grad_(True)
        ft_lr = getattr(cfg, "ft_learning_rate", 1e-4)
        optimizer_ft = AdamW(model.link_pred.parameters(), lr=ft_lr)
        # scheduler_ft = CosineAnnealingLR(optimizer_ft, T_max=cfg.ft_epochs, eta_min=1e-6)

        global_step_ft = 0
        for ft_epoch in range(1, cfg.ft_epochs + 1):
            model.memory.reset_state()
            model.neighbor_loader.reset_state()
            model.last_contact.clear()
            model.pair_count.clear()
            model.src_count.clear()
            model.train()

            total_ft_loss=0.0
            ft_bar = _pbar(range(num_train_batches), desc=f"Stage2 Epoch {ft_epoch:02d}/{cfg.ft_epochs} [MLP]")
            for i in ft_bar:
                optimizer_ft.zero_grad()
                # No uniformity term: the projector is frozen in Stage 2.
                out = _batch_loss(i, uniformity=False)
                if out is None:
                    continue
                loss, commit = out
                loss.backward()
                optimizer_ft.step()
                batch_ft_loss = loss.item()
                total_ft_loss += batch_ft_loss
                global_step_ft += 1
                if writer is not None and (global_step_ft % 10 == 0 or i == num_train_batches - 1):
                    writer.add_scalar("Stage2/Batch_Loss", batch_ft_loss, global_step_ft)
                ft_bar.set_postfix(loss=f"{batch_ft_loss:.4f}", refresh=False)
                commit()

            # current_ft_lr = scheduler_ft.get_last_lr()[0]
            # print(f"Stage 2 Epoch {ft_epoch:02d} | MLP Loss: {total_ft_loss / max(num_train_batches, 1):.4f} | LR: {current_ft_lr: .7f}")
            # scheduler_ft.step()
            stage2_epoch_loss = total_ft_loss / max(num_train_batches, 1)
            print(f"Stage 2 Epoch {ft_epoch:02d} | MLP Loss: {stage2_epoch_loss:.4f}")
            if writer is not None:
                writer.add_scalar("Stage2/Epoch_Loss", stage2_epoch_loss, ft_epoch)

    # Trained weights before calibration/eval: a crash there no longer costs the training.
    if writer is not None:
        torch.save(model.state_dict(), os.path.join(log_dir, "weights_pre_calib.pt"))

    # ------ THRESHOLD CALIBRATION (held-out benign slice) ------
    print("\n--- THRESHOLD CALIBRATION (on the benign validation stream) ---")
    def _slice(arr, lo, hi):
        return arr[lo:hi] if arr is not None else None

    # Calibration replays with the test gate (gate_by_label=False), so the thresholds are
    # fitted on the score distribution they are applied to. The score-driven precursor needs
    # a threshold that does not exist yet, hence one fixed-point iteration: pass A (sensor
    # alarms only) gives provisional thresholds, pass B replays the same slice under them.
    def _snapshot_runtime():
        """Deep copy of the mutable runtime state a replay advances (pass B restarts from it)."""
        return {
            "memory": copy.deepcopy(model.memory.state_dict()),
            "msg_s_store": copy.deepcopy(model.memory.msg_s_store),
            "msg_d_store": copy.deepcopy(model.memory.msg_d_store),
            "neighbor": {k: (v.clone() if torch.is_tensor(v) else v)
                         for k, v in model.neighbor_loader.state().items()},
            "last_contact": dict(model.last_contact),
            "pair_count": dict(model.pair_count),
            "src_count": dict(model.src_count),
            "recent_alert": dict(model.recent_alert),
        }

    def _restore_runtime(snap):
        """Restore a :func:`_snapshot_runtime` copy."""
        model.memory.load_state_dict(copy.deepcopy(snap["memory"]))
        model.memory.msg_s_store = copy.deepcopy(snap["msg_s_store"])
        model.memory.msg_d_store = copy.deepcopy(snap["msg_d_store"])
        model.neighbor_loader.load_state(
            {k: (v.clone() if torch.is_tensor(v) else v) for k, v in snap["neighbor"].items()}
        )
        model.last_contact = dict(snap["last_contact"])
        model.pair_count = dict(snap["pair_count"])
        model.src_count = dict(snap["src_count"])
        model.recent_alert = dict(snap["recent_alert"])

    model.recent_alert.clear()
    pre_cal_state = _snapshot_runtime()

    def _cal_replay(desc, thr=None, thr_dirty=None, thr_arm=None, gate_by_label=False, **kw):
        """Replay the validation slice from the pre-calibration state."""
        _restore_runtime(pre_cal_state)
        return _replay(
            model, _slice(source_arr, train_end, val_end), _slice(device_arr, train_end, val_end),
            user_arr[train_end:val_end], dst[train_end:val_end], t[train_end:val_end],
            msg[train_end:val_end], y[train_end:val_end], device, gate_by_label=gate_by_label,
            threshold=thr, threshold_dirty=thr_dirty, threshold_arm=thr_arm,
            config_nodes=_slice(config_arr, train_end, val_end),
            batch_size=cfg.eval_batch_size, desc=desc, **kw,
        )

    def _fit_thresholds(scores, labels):
        return fit_thresholds(scores, labels, types[train_end:val_end].numpy(),
                              msg[train_end:val_end].numpy(), cfg)

    # Pass A: test gate (signal-clean commits) and sensor-only arming: the feedback does not
    # depend on any threshold, so the per-edge reference fitted on it is exact and its
    # scores can be recomputed instead of replayed.
    scores_a, labels_a, extra_a = _cal_replay(
        "Calibration pass A (val replay, sensor-only arming)", return_edge_logits=True,
    )
    set_edge_calibration(model, None)
    if cfg.edge_calibration:
        ref = (labels_a == 0) & ~_rule_baseline(msg[train_end:val_end].numpy()).astype(bool)
        set_edge_calibration(model, {
            kind: fit_edge_calibration(v[ref], tail_q=cfg.edge_calib_tail_q)
            for kind, v in extra_a["edge_logits"].items()
        })
        comb = combine_edge_logits(model, {
            kind: torch.as_tensor(v) for kind, v in extra_a["edge_logits"].items()
        }).numpy()
        scores_a = anomaly_score(comb + extra_a["shift"])
        print(f"Per-edge benign calibration fitted on {int(ref.sum())} clean benign val events "
              f"({', '.join(extra_a['edge_logits'])}) | combine={cfg.edge_combine}")
    val_scores, val_labels, extra_v = scores_a, labels_a, extra_a
    threshold, threshold_dirty, threshold_clean_unsup, benign_val_scores = _fit_thresholds(
        val_scores, val_labels
    )
    # Passes B: replay under the current thresholds with the test's arming rule, so the
    # precursor shifts (and the benign tail they create) match the test replay; refit the
    # thresholds. The score arm threshold, when enabled, stays frozen at its pass-A value
    # (sensor-only arming, independent of any threshold): refitting it feeds back on itself
    # (arming → heavier benign tail → higher arm threshold → less arming → ...) and oscillates.
    # threshold_clean_unsup keeps its refit value: it is the label-free 1%-FPR threshold
    # reported as the global baseline and persisted in the calibration metadata.
    thr_arm = threshold_clean_unsup if cfg.precursor_arm_on_score else None
    for it in range(cfg.calib_iters):
        val_scores, val_labels, extra_v = _cal_replay(
            f"Calibration pass B{it + 1}/{cfg.calib_iters} (val replay, test gate + arming)",
            thr=threshold, thr_dirty=threshold_dirty, thr_arm=thr_arm,
            return_edge_logits=True,
        )
        threshold, threshold_dirty, threshold_clean_unsup, benign_val_scores = _fit_thresholds(
            val_scores, val_labels
        )
        print(f"  pass B{it + 1}: thr_clean={threshold:.6f} thr_dirty={threshold_dirty:.6f} "
              f"thr_unsup={threshold_clean_unsup:.6f} (thr_arm={thr_arm})")
    model.threshold_arm = thr_arm
    val_types = types[train_end:val_end].numpy()
    val_msg = msg[train_end:val_end].numpy()
    val_clean = ~_rule_baseline(val_msg).astype(bool)
    cal_mask = val_clean & ((val_labels == 0) | (val_types == 3))
    cal_scores = val_scores[cal_mask]
    cal_labels = (val_types[cal_mask] == 3).astype(int)
    print(
        f"Benign val score: mean={benign_val_scores.mean():.4f} "
        f"p95={np.quantile(benign_val_scores, 0.95):.4f}"
    )
    print(
        f"threshold@cost_ratio={cfg.cost_ratio}: {threshold:.4f} "
        f"[PRIMARY — persisted and used by serving for signal-clean events; requires "
        f"labelled lateral movement in the validation window; "
        f"clean cal: n_benign={int((cal_labels == 0).sum())} n_lateral={int(cal_labels.sum())}] | "
        f"threshold_dirty@FPR={cfg.target_fpr}: {threshold_dirty:.4f} "
        f"[signal-dirty events] | "
        f"threshold_clean_unsup@FPR={cfg.target_fpr}: {threshold_clean_unsup:.4f} "
        f"[label-free alternative; arms the precursor only with precursor_arm_on_score]"
    )
    # Lateral recall/FPR trade-off the clean threshold was picked from (operator/OPA reference).
    if cal_labels.sum() > 0:
        print("  clean-stream lateral recall vs benign FPR (reference curve):")
        for thr, rec, fpr in recall_fpr_curve(cal_scores, cal_labels, n_points=6):
            print(f"    thr={thr:.4f} | lateral_recall={rec:.3f} | benign_fpr={fpr:.4f}")

    # --- STREAMING EVALUATION (event-by-event, predicted-benign gating) ------
    if cfg.eval_batch_size > 1:
        print(
            f"\n[!] eval_batch_size={cfg.eval_batch_size} > 1: the scores are the "
            f"BATCHED APPROXIMATION (fast, but NOT identical to sequential serving: "
            f"intra-batch staleness). For final / publishable numbers use eval_batch_size=1.",
            file=sys.stderr,
        )
    _mode = f"batch={cfg.eval_batch_size}" if cfg.eval_batch_size > 1 else "per-event"
    print(f"\n--- INFERENCE / ANOMALY DETECTION PHASE START ({_mode}) ---")
    # Memory + neighbour history legitimately continue from the (benign) calibration
    # slice; the alert state does not.
    model.recent_alert.clear()  # don't let calibration-slice alerts pre-condition the test stream
    test_scores, test_labels, extra_t = _replay(
        model, _slice(source_arr, val_end, n), _slice(device_arr, val_end, n),
        user_arr[val_end:], dst[val_end:], t[val_end:], msg[val_end:], y[val_end:],
        device, config_nodes=_slice(config_arr, val_end, n),
        threshold=threshold, threshold_dirty=threshold_dirty,
        threshold_arm=model.threshold_arm, gate_by_label=False,
        batch_size=cfg.eval_batch_size, desc="Inferenza (replay test)",
        return_edge_logits=True,
    )

    test_types = types[val_end:].numpy()
    test_msg = msg[val_end:].numpy()

    # Signal-routed decision (matches serving): clean events -> cost-sensitive threshold,
    # dirty events -> conservative FPR threshold.
    dirty_test = _rule_baseline(test_msg).astype(bool)
    test_preds = routed_predict(test_scores, dirty_test, threshold, threshold_dirty)

    auc = roc_auc_score(test_labels, test_scores)
    ap = average_precision_score(test_labels, test_scores)
    precision, recall = binary_metrics(test_preds, test_labels, threshold=1)
    print(f"Test Stream | AUC: {auc:.4f} | AP: {ap:.4f}")
    print(f"Routed decision | Precision: {precision:.4f} | Recall: {recall:.4f}")

    # --- HEADLINE: lateral recall, global FPR threshold vs routed cost-sensitive decision ---
    lat = test_types == 3
    benign_test = test_labels == 0
    old_preds = (test_scores >= threshold_clean_unsup).astype(int)
    old_lat_recall = float(old_preds[lat].mean()) if lat.any() else float("nan")
    new_lat_recall = float(test_preds[lat].mean()) if lat.any() else float("nan")
    old_fpr = float(old_preds[benign_test].mean()) if benign_test.any() else float("nan")
    new_fpr = float(test_preds[benign_test].mean()) if benign_test.any() else float("nan")
    print("\n--- LATERAL RECALL: GLOBAL-FPR THRESHOLD  vs  COST-SENSITIVE ROUTING ---")
    print(f"  before (global @FPR={cfg.target_fpr}): lateral_recall={old_lat_recall:.4f} | benign_fpr={old_fpr:.4f}")
    print(f"  after  (routed cost-sensitive)       : lateral_recall={new_lat_recall:.4f} | benign_fpr={new_fpr:.4f}")
    # Aggregate recall at the global target_fpr threshold, the one the baselines report.
    pos_test = test_labels == 1
    old_agg_recall = float(old_preds[pos_test].mean()) if pos_test.any() else float("nan")
    print(f"  aggregate recall (global @FPR={cfg.target_fpr}): {old_agg_recall:.4f}")

    # --- PER-ANOMALY-TYPE BREAKDOWN ------------------------------------------
    # Per-type AUC/AP are benign (type 0) vs that type, so an aggregate cannot mask a weak
    # class. The vs-rule column says whether the signal-only rule baseline also catches it:
    #   policy, benign-denied  OPA-owned (decided upstream): sanity column only;
    #   contextual             rule-trivial (broken JA3 / sensor);
    #   lateral, cred-theft    rule-blind (signal-clean): the model's real target.
    print("\n--- PER-ANOMALY-TYPE METRICS ---")
    vs_rule = {
        1: "OPA-owned  ", 2: "rule-trivial", 3: "rule-blind ", 4: "rule-blind ",
        5: "volume-tell", 6: "OPA-owned  ",
    }
    per_type = {}
    benign = test_types == 0
    # 5 = exfiltration (external datasets only), kept apart from lateral: its transfer volume
    # would add a trivially separable sub-population. 6 = benign OPA denial (label 1, no attack).
    for type_id, name in ((1, "policy"), (2, "contextual"), (3, "lateral"),
                          (4, "cred-theft"), (5, "exfil"), (6, "benign-denied")):
        sel = benign | (test_types == type_id)
        s_sel, l_sel = test_scores[sel], (test_types[sel] == type_id).astype(int)
        if l_sel.sum() == 0:
            continue
        t_auc = roc_auc_score(l_sel, s_sel)
        t_ap = average_precision_score(l_sel, s_sel)
        # Recall at the routed operational decision (not a single global threshold).
        preds_sel = test_preds[sel]
        t_recall = float(preds_sel[l_sel == 1].mean()) if (l_sel == 1).any() else 0.0
        # Recall at the global target_fpr threshold (what baselines report); ``recall`` above
        # is the routed one: never mix the two in one table column.
        t_recall_global = float(old_preds[sel][l_sel == 1].mean())
        per_type[name] = {"auc": t_auc, "ap": t_ap, "recall": t_recall,
                          "recall_global": t_recall_global, "n": int(l_sel.sum())}
        print(f"  {name:10s} | {vs_rule[type_id]} | n={int(l_sel.sum()):4d} | AUC: {t_auc:.4f} | "
              f"AP: {t_ap:.4f} | Recall@thr: {t_recall:.4f} | Recall@FPR{cfg.target_fpr}: "
              f"{t_recall_global:.4f}")

    val_types_np = types[train_end:val_end].numpy()
    val_msg_np = msg[train_end:val_end].numpy()
    precursor_report("val", val_scores, val_labels, val_types_np, val_msg_np,
                     extra_v["shift"], threshold, threshold_dirty)
    precursor_report("test", test_scores, test_labels, test_types, test_msg,
                     extra_t["shift"], threshold, threshold_dirty)
    combine_report(model, extra_v, val_types_np, val_msg_np, extra_t, test_types, cfg.target_fpr)

    # Incident-level detection (synthetic ground truth only): an attack is stopped by its
    # first blocked event. At the routed decision and at the global val-1%-FPR threshold.
    incident_metrics = {}
    if data.incident is not None:
        test_incident = data.incident[val_end:].numpy()
        test_variant = data.theft_variant[val_end:].numpy()
        incident_metrics["routed"] = incident_report(
            "routed decision", test_preds, test_scores, test_types, test_incident, test_variant)
        incident_metrics["global"] = incident_report(
            f"global @FPR={cfg.target_fpr}", old_preds, test_scores, test_types, test_incident,
            test_variant)
    if writer is not None:
        # Test-stream scores for offline analysis / cross-branch comparison.
        np.savez_compressed(
            os.path.join(log_dir, "test_eval.npz"), scores=test_scores, labels=test_labels,
            types=test_types, preds_routed=test_preds, preds_global=old_preds,
            threshold=threshold, threshold_dirty=threshold_dirty,
            threshold_global=threshold_clean_unsup, test_start=val_end,
        )

    if writer is not None:
        eval_step = cfg.epochs + getattr(cfg, "ft_epochs", 0)
        writer.add_scalar("Eval/AUC", auc, eval_step)
        writer.add_scalar("Eval/AP", ap, eval_step)
        writer.add_scalar("Eval/Precision_routed", precision, eval_step)
        writer.add_scalar("Eval/Recall_routed", recall, eval_step)
        if not np.isnan(new_lat_recall):
            writer.add_scalar("Eval/Lateral_Recall_routed", new_lat_recall, eval_step)
        if not np.isnan(old_lat_recall):
            writer.add_scalar("Eval/Lateral_Recall_global", old_lat_recall, eval_step)
        if not np.isnan(new_fpr):
            writer.add_scalar("Eval/FPR_routed", new_fpr, eval_step)
        if not np.isnan(old_fpr):
            writer.add_scalar("Eval/FPR_global", old_fpr, eval_step)

        for type_name, m in per_type.items():
            writer.add_scalar(f"PerType/{type_name}_AUC", m["auc"], eval_step)
            writer.add_scalar(f"PerType/{type_name}_AP", m["ap"], eval_step)
            writer.add_scalar(f"PerType/{type_name}_Recall_routed", m["recall"], eval_step)

    # --- COLD-START CONDITIONING (lateral) -----------------------------------
    # Lateral recall split by whether the actor had benign history before the event (warmed)
    # or not (cold: detection not yet possible). Labels are used up to val_end; after that
    # the partition follows the system's own routed decision, as in deployment.
    actor_arr = device_arr if device_arr is not None else user_arr
    pred_full = np.zeros(len(y), dtype=np.int64)
    pred_full[val_end:] = test_preds
    src_seen_test = causal_src_seen(
        actor_arr.numpy(), y.numpy(), label_horizon=val_end, pred=pred_full
    )[val_end:]
    lat_mask = test_types == 3
    lat_pred = test_preds  # routed operational decision
    warmed = lat_mask & src_seen_test
    cold = lat_mask & ~src_seen_test
    cold_start = {"n_warmed": int(warmed.sum()), "n_cold": int(cold.sum())}
    cold_start["recall_warmed"] = float(lat_pred[warmed].mean()) if warmed.any() else float("nan")
    cold_start["recall_cold"] = float(lat_pred[cold].mean()) if cold.any() else float("nan")
    if warmed.any() and lat_mask.any():
        warmed_sel = (test_types == 0) | warmed
        cold_start["auc_warmed"] = float(roc_auc_score((test_types[warmed_sel] == 3).astype(int), test_scores[warmed_sel]))
    print("\n--- LATERAL: COLD-START CONDITIONING ---")
    print(f"  warmed src (has benign history): n={cold_start['n_warmed']:4d} | "
          f"recall@thr={cold_start['recall_warmed']:.4f} | AUC={cold_start.get('auc_warmed', float('nan')):.4f}")
    print(f"  cold   src (no history yet)     : n={cold_start['n_cold']:4d} | "
          f"recall@thr={cold_start['recall_cold']:.4f}  (detection not yet possible)")

    # --- SCENARIO-LEVEL EVALUATION -------------------------------------------
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
        new_user = benign_t & ((scen_test & SCEN_NEW_USER) != 0)
        scenario_metrics["fpr_new_user"] = _fpr(new_user)
        scenario_metrics["n_new_user"] = int(new_user.sum())

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
              f"wiped-cookie={scenario_metrics['fpr_wiped']:.4f} (n={scenario_metrics['n_wiped']}) | "
              f"new-user={scenario_metrics['fpr_new_user']:.4f} (n={scenario_metrics['n_new_user']})")
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
            "precursor_max_shift": cfg.precursor_max_shift,
            "use_resource_risk": cfg.use_resource_risk,
            "use_source_internal": cfg.use_source_internal,
            "guest_device_fallback": cfg.guest_device_fallback,
            "use_precursor": use_precursor,
            "edge_calibration": cfg.edge_calibration,
            "edge_combine": cfg.edge_combine,
            "head_negatives": cfg.head_negatives,
            "binding_hard_ratio": cfg.binding_hard_ratio,
            "mask_seen_negatives": cfg.mask_seen_negatives,
            "precursor_arm_on_score": cfg.precursor_arm_on_score,
        }
        op_new = operating_point(test_scores, test_labels, test_types, threshold)
        model.recent_alert.clear()
        save_model(
            model, registry, threshold, hp, TGN_CHECKPOINT_PATH, TGN_STATS_PATH,
            threshold_dirty=threshold_dirty,
            calibration={"mode": "cost", "cost_ratio": cfg.cost_ratio,
                         "clean_fpr_cap": cfg.clean_fpr_cap, "target_fpr": cfg.target_fpr,
                         # Label-free alternative for deployments without red-team labels.
                         "threshold_clean_unsup": threshold_clean_unsup},
            operating_point=op_new,
        )
        print(f"\nSaved checkpoint -> {TGN_CHECKPOINT_PATH}")
        print(f"Saved stats      -> {TGN_STATS_PATH}")

    if writer is not None:
        writer.close()

    return {
        "threshold": threshold,
        "threshold_dirty": threshold_dirty,
        "threshold_clean_unsup": threshold_clean_unsup,
        "agg_auc": auc,
        "agg_ap": ap,
        "agg_precision": precision,
        "agg_recall": recall,
        # Recall at the global target_fpr threshold (baseline comparison); agg_recall above
        # is the routed operational recall.
        "agg_recall_global": old_agg_recall,
        "lateral_recall_before": old_lat_recall,
        "lateral_recall_after": new_lat_recall,
        "fpr_before": old_fpr,
        "fpr_after": new_fpr,
        "per_type": per_type,
        "cold_start": cold_start,
        "incident": incident_metrics,
        "scenario": scenario_metrics,
        # Wall time of the gradient loop alone (the data-budget curve's cost axis): the call
        # as a whole is dominated by the fixed validation / test replays.
        "train_seconds": train_seconds,
        "use_struct_head": use_struct_head,
        "use_hash_identity": use_hash_identity,
        "use_hist_feats": use_hist_feats,
        "use_precursor": use_precursor,
        "use_config_node": use_config_node,
        # Raw per-event test scores (events [test_start, n) of the stream), for evaluation
        # protocols whose positive/negative sets differ from per_type's (e.g. LMDEval on OpTC).
        **({"test_scores": test_scores, "test_start": val_end} if return_scores else {}),
    }


if __name__ == "__main__":
    train_tgn()
