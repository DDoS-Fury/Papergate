"""Central configuration: hyper-parameters and paths for the streaming TGN.

The streaming Temporal Graph Network (v2) consumes synthetic ZTA access events
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

    # Synthetic stream shape (entity counts + number of events). v2 schema: the old
    # single "IP/device" entity is split into SOURCE (network context: the client IP)
    # and DEVICE (hardware context: TPM id or persistent device cookie), so an IP
    # change (smart working) no longer looks like a brand-new machine, while a new
    # device suddenly binding to a known user (credential theft) stands out.
    num_users: int = 50
    num_devices: int = 80
    num_sources: int = 150
    # MUST equal len(RESOURCE_URIS) in stream_synthetic.py: resource node keys are
    # the exact route URIs the security-orchestrator sends as key_dst.
    num_resources: int = 19
    num_events: int = 200000

    # Spare device-node slots for the generator's dynamic scenarios: a cookie wipe
    # re-keys a machine (new cold device node), a credential-theft incident brings a
    # never-seen attacker device + attacker IP.
    num_wipe_slots: int = 16
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
    p_cookie_wipe: float = 0.0003
    p_cred_theft: float = 0.0012

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
    msg_dim: int = 7
    time_dim: int = 32
    memory_dim: int = 256
    num_hops: int = 3
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

    # Kill-chain precursor prior (serving-time, not trained). Lateral movement follows a
    # recon alert on the same entity; we multiply an entity's anomaly score by up to
    # (1 + precursor_max_boost) right after it alerts, decaying with half-life
    # ``precursor_half_life`` (seconds). See serve_tgn.precursor_boost.
    # Production timing note: the synthetic clock draws ~300 s exponential gaps, so the
    # swept optimum (100000 s) maps to ~a day of real traffic — one alert would keep
    # multiplying a real user's scores toward 1.0 for hours (observed as score==1.0 on
    # every request after a single cold-start alert). 600 s keeps the kill-chain prior
    # for recon→lateral bursts without long-lived score poisoning.
    precursor_half_life: float = 600.0
    precursor_max_boost: float = 2.0

    # Optimisation.
    batch_size: int = 200
    epochs: int = 15
    learning_rate: float = 1e-3

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
    # re-challenge), turning the ~0.76 lateral AUC ranking into operational recall. See
    # graphagate.calibration.cost_sensitive_threshold.
    cost_ratio: float = 20.0
    # Guardrail on the cost-sensitive search: cap the *clean-stream* benign false-positive rate.
    # Lateral AUC ~0.70 is a steep recall/FPR trade-off, so an uncapped cost ratio would chase
    # recall to an absurd FPR; the cap fixes the operating point at the max recall achievable
    # while no more than this fraction of benign clean events are re-challenged by the orchestrator.
    clean_fpr_cap: float = 0.05

    seed: int = 42

    # Event schema version persisted in the checkpoint hyper-parameters. v3 = the
    # 4-node / 3-edge schema with type-namespaced keys (src:/ipdev:/tpm:/ck:) plus two
    # new static node features — node_feat[*,4]=resource RISK, node_feat[*,5]=source
    # network internal/external. node_feat index map: [2]=device tier, [3]=resource
    # priority, [4]=resource risk, [5]=source internal, [14]=trust. Earlier checkpoints
    # (v1 2-edge, v2 bare-IP keys / no risk-internal feats) are rejected at load time.
    schema_version: int = 3

    @property
    def total_nodes(self) -> int:
        return (
            self.num_users
            + self.num_devices + self.num_wipe_slots + self.num_theft_slots
            + self.num_sources + self.num_theft_slots
            + self.num_resources
        )

    @property
    def capacity(self) -> int:
        return self.total_nodes + self.capacity_headroom
