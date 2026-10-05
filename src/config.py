"""Central configuration: artifact paths and :class:`TGNConfig`, the single source of the
synthetic-generator knobs and of the TGN hyper-parameters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Repository root = parent of ``src/`` (this file lives at src/config.py).
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR: Path = REPO_ROOT / "public"

# Streaming TGN deployment artifacts.
TGN_CHECKPOINT_PATH: Path = ARTIFACTS_DIR / "tgn_checkpoint.pt"
TGN_STATS_PATH: Path = ARTIFACTS_DIR / "tgn_stats.json"


@dataclass(frozen=True)
class TGNConfig:
    """Generator knobs (mapped by ``stream_synthetic.stream_kwargs_from_cfg``) and TGN
    hyper-parameters.

    The model memory holds the training entities plus ``capacity_headroom`` slots, so
    unseen entities can be admitted at inference time through the
    :class:`~graphagate.model.registry.NodeRegistry`.
    """

    # --- Synthetic stream: entity counts and length -----------------------------------
    # SOURCE = client network (IP), DEVICE = hardware identity (TPM id or device cookie):
    # an IP change is not a new machine, a new device on a known user stands out.
    num_users: int = 100
    # Anonymous (unauthenticated) visitors, one user node each: part of the node space.
    num_guests: int = 1000
    num_devices: int = 160
    num_sources: int = 300
    # Resource catalogue size (real routes + synthetic estate); ZTAStreamSimulator asserts
    # it is at least the number of real routes. Keys are the URIs the orchestrator sends.
    num_resources: int = 1000
    # Client configurations (TLS/JA3). Low cardinality, so a small habitual pool; each
    # machine uses 1-2. Chain: source → config → device → user → resource (+ config → user).
    num_configs: int = 40
    num_events: int = 200000

    # Spare device slots, recycled round-robin: cookie wipes (a machine re-keyed as a cold
    # node) and credential-theft attacker devices draw from them.
    num_wipe_slots: int = 128
    num_theft_slots: int = 64

    # Behavioural dynamics (all benign except p_cred_theft):
    #   p_roam          — benign event issued from a non-home IP (smart working / 5G);
    #   p_shared_device — fraction of devices used by 2-4 users (control-room machine);
    #   p_cookie_wipe   — per-event chance a cookie-identified device wipes its cookie
    #                     (re-keyed as a cold node; benign, must not become a false positive);
    #   p_cred_theft    — per-event chance a credential-theft incident starts (etype 4).
    p_roam: float = 0.10
    p_shared_device: float = 0.20
    p_cookie_wipe: float = 0.001
    p_cred_theft: float = 0.0012

    # Collapse every non-TPM device onto one shared ``dev:guest`` node (mirror of
    # ``conf:guest``). Off: that node would carry ~75% of all events, hiding new
    # device->user bindings and disabling cookie wipes. Serving honours the checkpoint's flag.
    guest_device_fallback: bool = False

    # --- Open world and difficulty (stream_synthetic.ZTAStreamSimulator) --------------
    # Without them benign entities are all seen early while attackers bring fresh IP/JA3
    # slots, and a "never seen" lookup beats the TGN (tasks/runs/generator_rule_audit.log).
    #
    # Benign churn — novelty must be a common BENIGN event:
    #   p_new_source     — share of roaming events from a never-seen IP (mobile/CGNAT);
    #   p_config_release — per-step chance a client release gives one habitual JA3 a new
    #                      version; machines adopt it at p_config_adopt per use;
    #   p_hotdesk        — share of events where a user signs in on a machine not theirs;
    #   p_sensor_fp      — IDS probe false-positive rate on non-recon traffic.
    # Mimetic credential theft — the attacker runs a common client / a known egress /
    # replays the victim's stolen session cookie (pass-the-cookie):
    #   p_theft_mimic_config, p_theft_known_source, p_theft_session_replay.
    # Session-replay kits ship residential proxies, so most thefts egress from address
    # space the fleet also uses; at 0.5 src|usr_new alone crossed the audit's 0.85 AUC.
    # Kill chain — p_compromise is a global per-step intrusion rate with remediation after
    # the dwell (None = per-request hazard, never remediated). A lateral event pivots with a
    # harvested credential at p_lateral_foreign_cred; a credential comes from the machine's
    # logon cache at p_harvest_cached, else from a user who never signed in there.
    # p_lateral_role_spoof (a role-claim tell) stays at 0.
    # Fresh slot pools (shared by benign churn and attackers, recycled when exhausted):
    num_new_sources: int = 6000
    num_new_configs: int = 1024
    p_new_source: float = 0.3
    p_config_release: float = 0.00025
    p_config_adopt: float = 0.05
    p_hotdesk: float = 0.02
    p_sensor_fp: float = 0.01
    p_theft_mimic_config: float = 0.7
    p_theft_known_source: float = 0.7
    p_theft_session_replay: float = 0.5
    p_compromise: float | None = 0.0006
    p_lateral_foreign_cred: float = 0.7
    p_harvest_cached: float = 0.7
    p_lateral_role_spoof: float = 0.0
    p_lateral_new_config: float = 0.3
    # The service account (user 0) runs on this many dedicated server machines.
    num_service_machines: int | None = 3
    # Device posture mix (no cert / cert / cert+TPM). Unmanaged BYOD machines are common;
    # with too few of them a never-attested device is itself a theft tell (the attacker's
    # fresh cookie is always tier 0), readable off one static node feature.
    tier_mix: tuple[float, float, float] = (0.35, 0.4, 0.25)
    # Per-step chance an active theft incident emits its next request. Faster incidents
    # shrink the victim's inter-request gap (read by the TGN as the user's recency) until
    # it alone identifies the class.
    p_theft_interleave: float = 0.06
    # Never-seen users throughout the stream: num_new_users hires, one per equal stratum,
    # and (ramp_guests) visitors admitted progressively like the device fleet.
    # 24 hires / 100 users over ~10 months ≈ 2.4%/month, within the JOLTS 2025 hire rates.
    num_new_users: int = 24
    ramp_guests: bool = True
    # Share of events from a one-off client with a never-seen JA3 (new app, CLI tool), so a
    # new JA3 is not a theft tell. num_new_configs is sized to avoid recycling in one stream.
    p_benign_new_config: float = 0.003

    # Probability that a benign event is an authorised non-habitual access (exploration),
    # so "non-habitual" is not a lateral-movement label.
    benign_explore_prob: float = 0.15

    # Static-feature toggles (the slot stays 0 when off): resource risk in node_feat[4],
    # RFC1918 internal bit of the source in node_feat[5]. source_internal can mask
    # credential theft (roaming normalises external bindings), hence off.
    use_resource_risk: bool = True
    use_source_internal: bool = False

    # TGN Architecture
    node_feat_dim: int = 16
    # [ja3, s1, s2, s3, method, role, clearance]: request-time fields only (the PDP decides
    # before any response exists).
    msg_dim: int = 7
    time_dim: int = 32
    memory_dim: int = 256
    num_hops: int = 3
    # Attention heads per TransformerConv (persisted in the checkpoint).
    gnn_heads: int = 4
    # Hidden Linear layers of the feature head before the output (persisted).
    link_pred_hidden_layers: int = 3
    struct_proj_hidden_layers: int = 2
    hash_buckets: int = 100000
    hash_dim: int = 16
    # Temporal neighbours kept per node by the bounded in-memory neighbour loader.
    neighbor_size: int = 30
    # Interaction-history features: one triplet for the scored pair, one for the
    # (device, resource) pair on the access edge (ZTATemporalGraphNetwork.compute_hist_feats).
    hist_feat_dim: int = 6
    # InfoNCE ranking objective: number of random-destination negatives per positive.
    # The lateral signal is "rank the true dst above K alternatives given src history".
    infonce_k: int = 5
    # Δt clamp and never-seen sentinel (seconds, one week) for both recency inputs. A
    # constant sentinel, not t_now, keeps train/val/test and serving encodings stationary
    # (see ZTATemporalGraphNetwork.pair_delta_t).
    delta_t_cap: float = 604800.0

    # Kill-chain precursor prior (serving-time, not trained): after an alert, up to
    # ``precursor_max_shift`` nats are added to the entity's anomaly logit, halving every
    # ``precursor_half_life`` seconds (serve_tgn.precursor_shift). 72 h maximised lateral AUC
    # in tasks/tmp/precursor_sweep.py (6 h would favour credential theft instead).
    precursor_half_life: float = 259200.0
    precursor_max_shift: float = 4.0
    # Multiplicative equivalent for the baselines, whose scores are not logits
    # (eval_common.causal_precursor_factor); same half-life.
    precursor_max_boost: float = 2.0

    # Optimisation.
    batch_size: int = 200
    epochs: int = 15    # Training epochs (all network)
    ft_epochs: int = 5  # Finetune epochs (only linkPredictor)
    learning_rate: float = 1e-3
    ft_learning_rate: float = 1e-4
    # Offline replay batch size (calibration + test, never serving). 1 = exact per-event;
    # larger blocks score against the start-of-batch memory (train_tgn._replay).
    eval_batch_size: int = 1

    # Chronological split fractions (test = 1 - train - val).
    train_frac: float = 0.7
    val_frac: float = 0.1

    # Spare memory slots reserved for entities first seen at inference time.
    capacity_headroom: int = 50000
    # Benign FPR of the conservative threshold for signal-dirty events.
    target_fpr: float = 0.01
    # Signal-clean threshold minimises ``cost_ratio * FN + FP`` (a miss costs more than a
    # re-challenge); calibration.cost_sensitive_threshold.
    cost_ratio: float = 20.0
    # Cap on the clean-stream benign FPR during that search.
    clean_fpr_cap: float = 0.05
    # Per-edge benign calibration before the max over edges (serve_tgn.calibrated_edge_logit);
    # edge_calib_tail_q = quantile above which the reference gets an exponential tail.
    edge_calibration: bool = True
    edge_calib_tail_q: float = 0.99

    seed: int = 42

    # Checkpoint schema (4 = 5-node chain source → config → device → user → resource plus
    # config → user); other versions are rejected at load time. node_feat columns:
    # [2]=device tier, [3]=always 0 (regression invariant), [4]=resource risk,
    # [5]=source internal bit, [14]=trust (neutral 1.0).
    schema_version: int = 4
