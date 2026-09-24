"""Central configuration: hyper-parameters and paths for the streaming TGN.

The streaming Temporal Graph Network (v4 schema: the 5-node causal chain) consumes
synthetic ZTA access events
and scores each interaction online. Its memory is sized for ``capacity`` slots so
previously unseen entities can be admitted at inference time via the
:class:`~graphagate.model.registry.NodeRegistry`.
"""

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
    """Streaming Temporal Graph Network hyper-parameters.

    The model memory is sized for ``capacity`` slots (= training entities +
    ``capacity_headroom``) so previously unseen ZTA entities can be admitted at
    inference time through the :class:`~graphagate.model.registry.NodeRegistry`.
    """

    # Synthetic stream shape (entity counts + number of events). Since the v3 split
    # (kept in v4), the old single "IP/device" entity is split into SOURCE (network
    # context: the client IP) and DEVICE (hardware context: TPM id or persistent device
    # cookie), so an IP change (smart working) no longer looks like a brand-new
    # machine, while a new device suddenly binding to a known user (credential theft)
    # stands out.
    num_users: int = 100
    # Anonymous (unauthenticated) visitors, one user node each: part of the node space.
    num_guests: int = 1000
    num_devices: int = 160
    num_sources: int = 300
    # MUST equal len(RESOURCE_URIS) in stream_synthetic.py: resource node keys are
    # the exact route URIs the security-orchestrator sends as key_dst.
    num_resources: int = 1000
    # v4 schema: the client CONFIGURATION (TLS/JA3 fingerprint) is a 5th node role,
    # inserted into the causal chain as ``source → config → device → user → resource``
    # (plus a ``config → user`` binding). JA3 has limited cardinality (a handful of
    # browsers / tools share a fingerprint), so a modest pool of habitual configs is
    # realistic; each machine is assigned 1-2 of them (a never-seen config on a known
    # device / for a known user is the lateral-movement / credential-theft tell).
    num_configs: int = 40
    num_events: int = 200000

    # Spare device-node slots for the generator's dynamic scenarios: a cookie wipe
    # re-keys a machine (new cold device node), a credential-theft incident brings a
    # never-seen attacker device + attacker IP.
    # Since v5 both come from one recycled pool (plus the theft share of the fresh
    # source / config pools below).
    num_wipe_slots: int = 128
    num_theft_slots: int = 64

    # Behavioural dynamics of the 4-node stream (all benign except p_cred_theft):
    #   p_roam          — benign event issued from a non-home IP (smart working / 5G);
    #   p_shared_device — fraction of devices used by 2-3 users (control-room machine);
    #   p_cookie_wipe   — per-event chance a cookie-identified device wipes its cookie
    #                     (re-keyed as a cold node; benign, must not become a false positive);
    #   p_cred_theft    — per-event chance a credential-theft incident starts: a new IP +
    #                     new device issue requests as an existing victim user (etype 4).
    p_roam: float = 0.10
    p_shared_device: float = 0.20
    p_cookie_wipe: float = 0.001
    p_cred_theft: float = 0.0012

    # Collapse every non-TPM device onto a single shared ``dev:guest`` node (mirror of
    # ``conf:guest``) instead of giving each TPM-less machine its own cookie (``ck:``)
    # identity. OFF since v5: with it on, ~70% of the fleet shares one device node that
    # carries 72-78% of all events, so a new device->user binding (the lateral pivot) is
    # invisible on most machines and the cookie-wipe scenario never fires. The earlier
    # A/B (tests/ablations/run_guest_device_eval.py) compared a stream WITH wipes against
    # one where wipes are impossible by construction, so it cannot justify the default.
    # Serving still honours the flag stored in the checkpoint.
    guest_device_fallback: bool = False

    # --- v5 open world + difficulty knobs (stream_synthetic.ZTAStreamSimulator) -------
    # Without these, every benign entity is seen within the first few percent of the
    # stream while every attacker brings globally fresh IP/JA3 slots, and a set-membership
    # lookup ("never seen this IP") beats the TGN (tasks/runs/generator_rule_audit.log).
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
    # Kill chain — p_compromise is a global per-step intrusion rate with remediation
    # after exfiltration (None = the v4 per-visit hazard with no remediation, which left
    # ~95% of machines compromised and ~28% of events anomalous). A lateral event pivots
    # with a harvested credential (Euler/LANL sense) at p_lateral_foreign_cred; each
    # harvested credential comes from the machine's logon cache at p_harvest_cached (a
    # device->user binding already seen), otherwise from a user who never signed in there.
    # p_lateral_role_spoof is the v4 role-claim tell, kept at 0.
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
    # shrink the victim's inter-request gap (msg[9]) until it alone identifies the class.
    p_theft_interleave: float = 0.06
    # Never-seen users at every point of the stream (training and inference), not only
    # in the first few percent: num_new_users registered users are hired mid-stream, one
    # per equal stratum of the stream, and with ramp_guests anonymous visitors keep
    # arriving over the whole stream like the device fleet.
    # 24 hires on 100 users over a ~10-month stream is ~2.4%/month, inside the JOLTS 2025
    # annual-average hires rates (finance 2.1%, professional services 4.6%); with 12 on 50
    # users only ~8 hires fell in training, too few for the model to learn a cold employee.
    num_new_users: int = 24
    ramp_guests: bool = True
    # Share of events issued by a one-off client whose JA3 nobody presented before (new
    # app, CLI tool, updater). Without it a globally never-seen JA3 is half theft in the
    # test window (11 benign vs 11 theft, seed 2000). The fresh config pool is sized so
    # these draws do not recycle slots within one stream.
    p_benign_new_config: float = 0.003

    # De-degeneration knob: probability that a *benign* event performs an
    # authorised-but-non-habitual access (legitimate exploration). With this > 0 the
    # task is no longer the tautology "non-habitual ⟺ malicious" — lateral movement
    # must be told apart from benign novelty by its temporal/relational *pattern*,
    # not by novelty alone. See stream_synthetic.generate_streaming_data.
    benign_explore_prob: float = 0.15

    # v3 static-feature ablation toggles (node_feat_dim is unchanged either way; the
    # slot is simply left at 0 when off). resource_risk (node_feat[*,4]) = per-resource
    # sensitivity baked from getResourceSensitivity. source_internal (node_feat[*,5]) =
    # RFC1918 internal/external bit of the client network. NB: source_internal can
    # *mask* credential theft (benign external roaming normalises external→device
    # bindings, the very edge that exposes a stolen-credential attacker), so it is
    # ablatable and validated against the cred-theft gate before being enabled.
    use_resource_risk: bool = True
    use_source_internal: bool = False

    # TGN Architecture
    node_feat_dim: int = 16
    # [ja3, s1, s2, s3, method, role, clearance, bytes_in, bytes_out, log1p(user Δt)/10]
    # — request-time fields only; see the stream_synthetic module docstring.
    msg_dim: int = 10
    time_dim: int = 32
    memory_dim: int = 256
    num_hops: int = 3
    # Attention heads of each TransformerConv in the GraphAttentionEmbedding. Persisted
    # in the checkpoint hyper-parameters so serving rebuilds the same architecture.
    gnn_heads: int = 4
    # Number of hidden Linear layers in the feature-head MLP (LinkPredictor) before the
    # output unit. 2 = the historical layout (lin_in + 1 mid + out). Also persisted.
    link_pred_hidden_layers: int = 2
    hash_buckets: int = 100000
    hash_dim: int = 16
    # Temporal neighbours kept per node in the (bounded, in-memory) neighbour loader.
    # Enables message passing over each entity's recent interaction history → the
    # structural signal for lateral-movement detection. No graph DB required.
    neighbor_size: int = 30
    # Explicit interaction-history features per scored event (runtime-derivable,
    # non-circular): [log1p(pair_count), log1p(src_count), pair_count/(src_count+1)]
    # for the scored edge, plus the same triplet for an auxiliary (device, resource)
    # pair on the access edge — the per-device habituality signal that previously
    # lived on the direct device→resource edge (kept as counters, not as a 4th
    # temporal edge). See ZTATemporalGraphNetwork.compute_hist_feats.
    hist_feat_dim: int = 6
    # InfoNCE ranking objective: number of random-destination negatives per positive.
    # The lateral signal is "rank the true dst above K alternatives given src history".
    infonce_k: int = 5
    # Recency cap, in the stream's own clock unit (seconds). Both Δt inputs — pair
    # recency and src activity — are clamped here, and a pair/entity never observed
    # before is given exactly this value as a sentinel. Encoding "never seen" as
    # Δt = t_now (the absolute clock) instead would grow monotonically along the
    # stream: train / val / test would see systematically different encodings of the
    # same state and a long-running server would drift away from both. It is also a
    # shortcut in the InfoNCE objective — every random negative would carry an
    # absolute-clock Δt while every positive carries a small one, making the ranking
    # task solvable from one scalar. One week: a pair silent for longer is operationally indistinguishable
    # from one never seen, which is exactly what the sentinel asserts.
    delta_t_cap: float = 604800.0

    # Kill-chain precursor prior (serving-time, not trained). Lateral movement follows a
    # recon alert on the same entity; we add up to ``precursor_max_shift`` nats to that
    # entity's anomaly logit right after it alerts, decaying with half-life
    # ``precursor_half_life`` (seconds). See serve_tgn.precursor_shift.
    # The shift is additive on the logit because the prior is a prior on the odds; the
    # earlier multiplicative form acted on ``1 - sigmoid(logit)`` and was then clipped to
    # 1.0, so it was a no-op on the saturated events and negligible on the low-scoring
    # ones. Being bounded by ``precursor_max_shift`` it cannot drive a score to 1.0 on its
    # own, which is what forced the previous 600 s half-life; the half-life can now be set
    # from the measured recon->lateral delay instead (median 8.7 h, p90 56.5 h on the
    # dev stream, see tasks/tmp/lateral_chain_diag.py).
    precursor_half_life: float = 600.0
    precursor_max_shift: float = 2.0
    # Multiplicative equivalent, used only by the baselines' causal mirror
    # (eval_common.causal_precursor_factor): their scores are not probabilities, so an
    # additive logit shift is not defined on them. Kept so the baselines receive the same
    # prior with the same half-life.
    precursor_max_boost: float = 2.0

    # Optimisation.
    batch_size: int = 200
    epochs: int = 15
    learning_rate: float = 1e-3
    # Offline streaming-eval batch size (calibration + test replay only — NOT the online
    # serving path, which stays strictly sequential). 1 = exact per-event behaviour; larger
    # values score a whole block against the start-of-batch memory snapshot then commit the
    # updates in batch (the standard batched-TGN regime the model is already trained under),
    # saturating the GPU on large datasets like LANL. See train_tgn._replay.
    eval_batch_size: int = 1

    # Chronological split fractions (test = 1 - train - val).
    train_frac: float = 0.7
    val_frac: float = 0.1

    # Spare memory slots reserved for entities first seen at inference time.
    capacity_headroom: int = 50000
    # Decision-threshold calibration target (benign false-positive rate). Used for the
    # conservative *signal-dirty* threshold (events whose edge signal already fires).
    target_fpr: float = 0.01
    # Cost-sensitive calibration for the *signal-clean* stream, where lateral movement is
    # indistinguishable from benign except by temporal pattern. The clean threshold minimises
    # ``cost_ratio * FN + FP`` (FN = missed detection ≫ FP = false alarm the orchestrator can
    # re-challenge), turning the ~0.72 lateral AUC ranking into operational recall. See
    # graphagate.calibration.cost_sensitive_threshold.
    cost_ratio: float = 20.0
    # Guardrail on the cost-sensitive search: cap the *clean-stream* benign false-positive rate.
    # Lateral AUC ~0.72 is a steep recall/FPR trade-off, so an uncapped cost ratio would chase
    # recall to an absurd FPR; the cap fixes the operating point at the max recall achievable
    # while no more than this fraction of benign clean events are re-challenged by the orchestrator.
    clean_fpr_cap: float = 0.05

    seed: int = 42

    # Event schema version persisted in the checkpoint hyper-parameters. v4 = the
    # 5-node schema: the client CONFIGURATION (JA3) node is inserted into the causal
    # chain as ``source → config → device → user → resource`` plus a ``config → user``
    # binding (config keys namespaced ``conf:<ja3>`` / ``conf:guest``). v3 was the
    # 4-node / 3-edge schema with type-namespaced keys (src:/ipdev:/tpm:/ck:). node_feat
    # index map: [2]=device tier, [3]=unused (held at 0.0 on purpose — a resource index
    # would be a single-feature shortcut, and the zero is a regression invariant),
    # [4]=resource risk, [5]=source network internal/external, [14]=trust.
    # Earlier checkpoints (v1/v2/v3) are rejected at load time.
    schema_version: int = 4

    @property
    def total_nodes(self) -> int:
        return (
            self.num_users + self.num_guests
            + self.num_devices + self.num_wipe_slots + self.num_theft_slots
            + self.num_sources + self.num_theft_slots + self.num_new_sources
            + self.num_configs + self.num_theft_slots + self.num_new_configs
            + self.num_resources
        )

    @property
    def capacity(self) -> int:
        return self.total_nodes + self.capacity_headroom
