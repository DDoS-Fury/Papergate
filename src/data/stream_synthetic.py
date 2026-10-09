"""Synthetic ZTA access stream generator: 5 node roles, 5 edges per request.

Every HTTP request involves five entities mapped to a contiguous node index space:
    [0, U)                -> Users      (identity context: user credentials)
    [U, U+D)              -> Devices    (hardware context: TPM id or device cookie)
    [U+D, U+D+S)          -> Sources    (network context: client IP)
    [U+D+S, U+D+S+C)      -> Configs    (client context: TLS/JA3 fingerprint)
    [U+D+S+C, U+D+S+C+R)  -> Resources  (data context: route URI)

Requests unroll into a causal chain:
    source_ip -> config -> device -> user -> resource
accompanied by a ``config -> user`` binding. The access edge ``user -> resource``
carries the 7-dimensional request message; structural binding edges carry zero messages.

Who decides what is allowed: :mod:`graphagate.data.access_policy` (reference Bell-LaPadula
model + resource catalogue). This module only simulates who sends which request.

Entities and Open-World Dynamics:
  * **CONFIG**: client software identity (``conf:<ja3>`` or generic ``conf:guest``).
  * **Roaming** (``p_roam``): benign requests from non-home IPs (e.g. remote work / 5G).
  * **Shared Devices** (``p_shared_device``): multi-user workstations where device history bridges users.
  * **Cookie Wipes** (``p_cookie_wipe``): non-TPM machines re-keying to fresh cookie slots.
  * **Open-World Churn**: benign evolution (new roaming IPs, JA3 updates, new hires, visitors)
    and attacks draw from shared fresh pools with identical key formats to prevent trivial novelty lookup shortcuts.
  * **Credential Theft** (``p_cred_theft``, etype 4): attacker issues requests as a legitimate
    victim user using mimetic techniques (common client, fleet egress IP, or replayed session cookies).
  * **Lateral Movement** (etype 3): pivot using harvested credentials (cached or foreign) or
    non-habitual accesses on compromised hosts.

Anomaly Types (``types``, see :class:`EventType`):
  * 0 = Benign
  * 1 = Policy violation (denied by authorization rules)
  * 2 = Contextual anomaly (reconnaissance / scanner probes)
  * 3 = Lateral movement
  * 4 = Credential theft
  * 6 = Benign policy denial (user error, denied by policy but not an attack)
  * (5 is reserved for external dataset exfiltration labels)

Scenario Bitmask (``scenario``):
  1 = roaming, 2 = wiped cookie device (cold), 4 = shared device, 8 = recently hired user.

Edge Message Layout (7-dim):
  ``[ja3, s1, s2, s3, method, role, clearance]``
  Strictly restricted to telemetry available to the PDP *before* request dispatch (no response fields).

Key Generator Invariants (validated in ``tests/test_leakage_audit.py``):
  * **No univariate shortcuts**: no single feature cleanly separates attacks (benign and attack
    traffic share identical destination marginals via Zipf popularity weighting).
  * **No exact-value fingerprints**: no deterministic constants assigned per anomaly class.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import torch

from graphagate.data import access_policy
from graphagate.data.access_policy import (
    NUM_BASE_ROUTES,
    ROLE_CLEARANCE,
    ROLES,
    build_resource_universe,
)
from graphagate.netclass import GUEST_DEVICE, ip_is_internal


def stream_kwargs_from_cfg(cfg) -> dict:
    """Extract generator kwargs from a TGNConfig instance.

    Single source of truth mapping TGNConfig to generate_streaming_data, ensuring
    evaluation baselines, live generator, and audits share the exact entity space
    and parameters of the trained model.
    """
    return dict(
        num_users=cfg.num_users,
        num_guests=cfg.num_guests,
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
        num_new_sources=cfg.num_new_sources,
        num_new_configs=cfg.num_new_configs,
        p_new_source=cfg.p_new_source,
        p_config_release=cfg.p_config_release,
        p_config_adopt=cfg.p_config_adopt,
        p_hotdesk=cfg.p_hotdesk,
        p_theft_mimic_config=cfg.p_theft_mimic_config,
        p_theft_known_source=cfg.p_theft_known_source,
        p_theft_session_replay=cfg.p_theft_session_replay,
        p_compromise=cfg.p_compromise,
        p_lateral_foreign_cred=cfg.p_lateral_foreign_cred,
        p_harvest_cached=cfg.p_harvest_cached,
        p_lateral_role_spoof=cfg.p_lateral_role_spoof,
        p_lateral_new_config=cfg.p_lateral_new_config,
        p_sensor_fp=cfg.p_sensor_fp,
        num_service_machines=cfg.num_service_machines,
        tier_mix=cfg.tier_mix,
        p_theft_interleave=cfg.p_theft_interleave,
        num_new_users=cfg.num_new_users,
        ramp_guests=cfg.ramp_guests,
        p_benign_new_config=cfg.p_benign_new_config,
    )


# Configuration overrides reproducing the legacy closed-world baseline
# (closed-world benign traffic, fresh attacker slots, role spoofing) for comparative audits.
V4_KNOBS = dict(
    guest_device_fallback=True, num_wipe_slots=16, p_cookie_wipe=0.0003,
    num_new_sources=0, num_new_configs=0, p_new_source=0.0, p_config_release=0.0,
    p_hotdesk=0.0, p_sensor_fp=0.0,
    p_theft_mimic_config=0.0, p_theft_known_source=0.0, p_theft_session_replay=0.0,
    p_compromise=None, p_lateral_foreign_cred=0.0, p_harvest_cached=0.0,
    p_lateral_role_spoof=0.5,
    p_lateral_new_config=0.5, num_service_machines=None, tier_mix=(0.2, 0.5, 0.3),
    p_theft_interleave=0.15, num_new_users=0, ramp_guests=False,
    p_benign_new_config=0.0,
)


class EventType(IntEnum):
    """Anomaly type of an event (``SyntheticStream.types``); 5 is reserved for external exfiltration."""

    BENIGN = 0
    POLICY = 1
    CONTEXT = 2
    LATERAL = 3
    CRED_THEFT = 4
    BENIGN_DENIAL = 6


class KillPhase(IntEnum):
    """Kill-chain phase of a compromised machine."""

    RECON = 1
    LATERAL = 2
    DWELL = 4


# Scenario bitmask annotations for benign-context evaluation.
SCEN_ROAMING = 1   # Non-home IP (remote work / 5G)
SCEN_WIPED = 2     # Recently re-keyed cookie (cold device node)
SCEN_SHARED = 4    # Workstation shared across multiple users
SCEN_NEW_USER = 8  # Recently onboarded user (cold user node)

# Credential-theft variant bits (event field ``theft_variant``): the mimicry the attacker used.
THEFT_REPLAY = 1     # replayed one of the victim's device cookies (pass-the-cookie)
THEFT_MIMIC_CFG = 2  # the victim's / a common fleet client instead of a fresh JA3
THEFT_KNOWN_SRC = 4  # a fleet egress IP instead of a fresh one

_WIPE_COLD_EVENTS = 25      # Events until a re-keyed device is considered warm
_NEW_USER_COLD_EVENTS = 25  # Events until a new user node is considered warm

# Per-request probabilities of the benign and attack branches of ``step``.
_P_GUEST_CONFIG = 0.05          # benign request from an unfingerprinted client (conf:guest)
_P_ANONYMOUS = 0.15             # benign request from an anonymous visitor
_P_SERVICE_ACCOUNT = 0.05       # benign request is the service account's cronjob
_P_BENIGN_DENIAL = 0.02         # registered user hits a denied route by mistake (etype 6)
_P_ATTACK_ON_COMPROMISED = 0.3  # compromised host emits attack traffic (benign otherwise)
_P_LEGACY_COMPROMISE = 0.005    # per-request compromise hazard when p_compromise is None

_SERVICE_USER = 0     # user 0 is the service account
_SERVICE_CONFIG = 1   # its fixed client config (conf:0001)
_OFFICE_SUBNET = 30   # source slots [0, 30) are office NAT IPs (src:10.0.0.x)
_ZIPF_EXPONENT = 1.2  # resource popularity ~ 1 / rank^1.2

# Circadian inter-arrival scale (seconds between requests).
_DAY_S = 86400
_WORK_START_S, _WORK_END_S = 8 * 3600, 18 * 3600
_SCALE_WORK, _SCALE_NIGHT, _SCALE_WEEKEND = 45.0, 600.0, 1200.0

# Static node feature columns (node_features is [num_nodes, 16]).
_NODE_FEAT_DIM = 16
_NF_DEVICE_TIER = 2
_NF_RESOURCE_RISK = 4
_NF_SOURCE_INTERNAL = 5
_NF_TRUST = 14


def _zipf_weight(rank):
    """Unnormalised popularity of the resource at ``rank`` (0 = most popular)."""
    return 1.0 / ((rank + 1.0) ** _ZIPF_EXPONENT)


@dataclass
class _Request:
    """Entities of the request being generated; branches may rewrite any of them."""

    machine: int
    device: int
    user: int
    role: str
    clearance: int
    source: int
    config: int
    scenario: int


@dataclass
class _TheftIncident:
    """Active credential-theft session: attacker context replayed with the victim's identity."""

    victim: int
    device: int
    source: int
    config: int
    remaining: int  # requests left in the session
    incident: int = -1  # incident id (event field ``incident``)
    variant: int = 0    # THEFT_* bits


class ZTAStreamSimulator:
    """Stateful generator for synthetic ZTA access event streams.

    Powers both the offline tensor stream (generate_streaming_data) and live API
    streaming. Simulates progressive device/entity admission, benign organizational
    dynamics, and stealthy attack kill chains. ``step()`` returns one event dict.
    """

    def __init__(
        self,
        num_users: int = 1000,
        num_guests: int = 1000,
        num_devices: int = 2000,
        num_sources: int = 1500,
        num_configs: int = 400,
        num_resources: int = 19,
        num_wipe_slots: int = 16,
        num_theft_slots: int = 64,
        benign_explore_prob: float = 0.15,
        p_roam: float = 0.10,
        p_shared_device: float = 0.30,
        p_cookie_wipe: float = 0.0003,
        p_cred_theft: float = 0.0012,
        admission_horizon: int | None = None,
        seed: int | None = None,
        start_time: int = 0,
        use_resource_risk: bool = True,
        use_source_internal: bool = False,
        guest_device_fallback: bool = False,
        # Open-world / difficulty knobs (module docstring). Defaults give the closed-world
        # process (V4_KNOBS minus guest_device_fallback); TGNConfig sets the published values.
        num_new_sources: int = 0,
        num_new_configs: int = 0,
        p_new_source: float = 0.0,
        p_config_release: float = 0.0,
        p_config_adopt: float = 0.05,
        p_hotdesk: float = 0.0,
        p_theft_mimic_config: float = 0.0,
        p_theft_known_source: float = 0.0,
        p_theft_session_replay: float = 0.0,
        p_compromise: float | None = None,
        p_lateral_foreign_cred: float = 0.0,
        p_harvest_cached: float = 0.0,
        p_lateral_role_spoof: float = 0.5,
        p_lateral_new_config: float = 0.5,
        p_sensor_fp: float = 0.0,
        num_service_machines: int | None = None,
        tier_mix: tuple[float, float, float] = (0.2, 0.5, 0.3),
        p_theft_interleave: float = 0.15,
        num_new_users: int = 0,
        ramp_guests: bool = False,
        p_benign_new_config: float = 0.0,
    ):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self._build_catalogue(num_resources, seed)
        self.num_registered_users = num_users
        self.num_guests = num_guests
        self.num_users = num_users + num_guests
        self.num_devices = num_devices
        self.num_sources = num_sources
        self.num_configs = num_configs
        self.num_resources = num_resources
        self.num_wipe_slots = num_wipe_slots
        self.num_theft_slots = num_theft_slots
        self.num_new_sources = num_new_sources
        self.num_new_configs = num_new_configs
        self.guest_device_fallback = guest_device_fallback
        self.benign_explore_prob = benign_explore_prob
        self.p_roam = p_roam
        self.p_cookie_wipe = p_cookie_wipe
        self.p_cred_theft = p_cred_theft
        self.admission_horizon = admission_horizon
        self.use_resource_risk = use_resource_risk
        self.p_new_source = p_new_source
        self.p_config_release = p_config_release
        self.p_config_adopt = p_config_adopt
        self.p_hotdesk = p_hotdesk
        self.p_theft_mimic_config = p_theft_mimic_config
        self.p_theft_known_source = p_theft_known_source
        self.p_theft_session_replay = p_theft_session_replay
        self.p_theft_interleave = p_theft_interleave
        self.p_compromise = p_compromise
        self.p_lateral_foreign_cred = p_lateral_foreign_cred
        self.p_harvest_cached = p_harvest_cached
        self.p_lateral_role_spoof = p_lateral_role_spoof
        self.p_lateral_new_config = p_lateral_new_config
        self.p_sensor_fp = p_sensor_fp
        self.ramp_guests = ramp_guests
        self.p_benign_new_config = p_benign_new_config

        # Build order is fixed: every step below draws from the global RNGs.
        # Popularity rank is a random permutation, so resource ids do not encode frequency.
        pop_rank = np.random.permutation(num_resources).astype(np.float64)
        self._res_pop_weight = _zipf_weight(pop_rank)
        self._build_layout()
        self._build_population(tier_mix, p_shared_device, num_new_users, num_service_machines)
        self._build_network(num_service_machines)
        self._build_keys()
        self._build_node_features(use_source_internal)
        self._build_behaviour()
        self._init_state(start_time)

    # --- Construction ---
    def _build_catalogue(self, num_resources: int, seed: int | None) -> None:
        """Per-run resource catalogue: the real routes plus a synthetic estate of the right size."""
        n_generated = num_resources - NUM_BASE_ROUTES
        assert n_generated >= 0, (
            f"num_resources ({num_resources}) must be >= the {NUM_BASE_ROUTES} "
            f"real routes: set TGNConfig.num_resources accordingly"
        )
        (
            self.route_methods,
            self.security_matrix,
            self.resource_uris,
            self.resource_risk,
        ) = build_resource_universe(n_generated, seed if seed is not None else 42)
        assert num_resources == len(self.resource_uris), (
            f"num_resources ({num_resources}) must equal the built catalogue size "
            f"({len(self.resource_uris)})"
        )

    def _build_layout(self) -> None:
        """Node index layout: [users][device slots][source slots][config slots][resources]."""
        self.user_lo = 0
        self.dev_lo = self.num_users
        self.dev_slots = self.num_devices + self.num_wipe_slots + self.num_theft_slots
        self.src_lo = self.dev_lo + self.dev_slots
        # Trailing slots form shared fresh pools for benign churn and attackers
        self.src_slots = self.num_sources + self.num_theft_slots + self.num_new_sources
        self.cfg_lo = self.src_lo + self.src_slots
        self.cfg_slots = self.num_configs + self.num_theft_slots + self.num_new_configs
        self.res_lo = self.cfg_lo + self.cfg_slots
        self.num_nodes = self.res_lo + self.num_resources

    def _build_population(self, tier_mix, p_shared_device, num_new_users, num_service_machines) -> None:
        """Roles, machine tiers, scheduled hires and the desk owners of each machine."""
        self.user_roles = [str(np.random.choice(ROLES)) for _ in range(self.num_registered_users)]
        self.user_roles.extend(["guest"] * self.num_guests)
        self.user_clearances = [ROLE_CLEARANCE[r] for r in self.user_roles]

        # Tier: 0=unmanaged, 1=cert-only, 2=TPM-backed.
        self.machine_tiers = [int(np.random.choice([0, 1, 2], p=tier_mix))
                              for _ in range(self.num_devices)]
        # Desk owners (user 0 reserved as dedicated service account if service machines exist)
        self._humans = (
            list(range(1, self.num_registered_users))
            if num_service_machines is not None and self.num_registered_users > 1
            else list(range(self.num_registered_users))
        )
        # Mid-stream hires: (admission step, user) spread across the admission horizon
        self._pending_hires: list[tuple[int, int]] = []
        if num_new_users > 0 and self.admission_horizon:
            cand = [u for u in self._humans if u != 0]
            k = min(num_new_users, max(len(cand) - 1, 0))
            hires = [int(u) for u in np.random.permutation(cand)[:k]]
            self._pending_hires = [
                (int((i + np.random.rand()) * self.admission_horizon / k), u)
                for i, u in enumerate(hires)
            ]
            pending = set(hires)
            self._humans = [u for u in self._humans if u not in pending]
        pending = {u for _, u in self._pending_hires}
        self._registered = [u for u in range(self.num_registered_users) if u not in pending]
        self._user_age: dict[int, int] = {}
        self.machine_users: list[list[int]] = []
        for m in range(self.num_devices):
            users = [self._humans[m % len(self._humans)]]
            if np.random.rand() < p_shared_device:
                extra = np.random.randint(1, 4)
                pool = [u for u in self._humans if u not in users]
                users += list(np.random.choice(pool, size=min(extra, len(pool)), replace=False))
            self.machine_users.append(users)

    def _build_network(self, num_service_machines) -> None:
        """Per-machine home IPs and habitual client configs; service machines."""
        # Home IPs: one office IP (RFC1918 NAT) plus one remote/home IP
        self._num_office = min(_OFFICE_SUBNET, self.num_sources)
        self._office_locals = list(range(self._num_office))
        self.machine_home_ips: list[set[int]] = []
        for m in range(self.num_devices):
            home = {int(np.random.choice(self._office_locals))}
            if self.num_sources > self._num_office:
                home.add(int(np.random.randint(self._num_office, self.num_sources)))
            self.machine_home_ips.append(home)

        # Habitual client configurations (TLS/JA3) per machine; 0 is conf:guest
        cfg_pool = list(range(1, self.num_configs)) if self.num_configs > 1 else [0]
        self.machine_configs: list[list[int]] = []
        for m in range(self.num_devices):
            k = min(int(np.random.randint(1, 3)), len(cfg_pool))
            cfgs = np.random.choice(cfg_pool, size=k, replace=False)
            self.machine_configs.append([int(c) for c in cfgs])

        # Machines running the service account's cronjobs
        self.service_machines = (
            None if num_service_machines is None
            else list(range(min(num_service_machines, self.num_devices)))
        )

    def _build_keys(self) -> None:
        """External key of every node slot (what the orchestrator would send)."""
        self.keys: list[str | None] = [None] * self.num_nodes
        for u in range(self.num_registered_users):
            self.keys[self.user_lo + u] = f"user_{u:04d}"
        for g in range(self.num_guests):
            self.keys[self.user_lo + self.num_registered_users + g] = f"guest_{g:04d}"
        # Device keys: TPM-backed or random opaque cookies (ck:<hex>)
        self._used_cookies: set[str] = set()
        for m in range(self.num_devices):
            tier = self.machine_tiers[m]
            self.keys[self.dev_lo + m] = f"tpm:{m:04d}" if tier == 2 else self._new_cookie()
        for k in range(self.num_wipe_slots + self.num_theft_slots):
            self.keys[self.dev_lo + self.num_devices + k] = f"_spare_dev_{k}"
        # Source keys: internal office IPs (10.0.0.x) and external/CGNAT IPs (100.64.x.x)
        for s in range(self.num_sources):
            self.keys[self.src_lo + s] = (
                f"src:10.0.0.{s}" if s < self._num_office else f"src:100.64.{s // 256}.{s % 256}"
            )
        for k in range(self.num_theft_slots + self.num_new_sources):
            self.keys[self.src_lo + self.num_sources + k] = self._fresh_ip_key(self.num_sources + k)
        # Config keys: 0=conf:guest, 1..num_configs=habitual, trailing=fresh pool
        self.keys[self.cfg_lo] = "conf:guest"
        for c in range(1, self.num_configs):
            self.keys[self.cfg_lo + c] = f"conf:{c:04d}"
        for k in range(self.num_theft_slots + self.num_new_configs):
            self.keys[self.cfg_lo + self.num_configs + k] = f"conf:{self.num_configs + k:04d}"
        for r in range(self.num_resources):
            self.keys[self.res_lo + r] = self.resource_uris[r]

    def _build_node_features(self, use_source_internal: bool) -> None:
        """Static node features: device tier, resource risk, internal-source flag, trust = 1.0."""
        nf = torch.zeros(self.num_nodes, _NODE_FEAT_DIM)
        nf[:, _NF_TRUST] = 1.0  # neutral constant
        for m in range(self.num_devices):
            nf[self.dev_lo + m, _NF_DEVICE_TIER] = self.machine_tiers[m] / 2.0
        if self.use_resource_risk:
            for r in range(self.num_resources):
                nf[self.res_lo + r, _NF_RESOURCE_RISK] = self.resource_risk[self.resource_uris[r]]
        if use_source_internal:
            for s in range(self.src_slots):
                is_internal = ip_is_internal(self.keys[self.src_lo + s])
                nf[self.src_lo + s, _NF_SOURCE_INTERNAL] = 1.0 if is_internal else 0.0
        self.node_features = nf

    def _build_behaviour(self) -> None:
        """Per-user habitual actions (half of what the role allows) and action caches."""
        self._valid_cache: dict[str, list[tuple[int, int]]] = {}
        self._violation_cache: dict[str, list[tuple[int, int]]] = {}
        self._user_action_cache: dict[int, tuple[list, list]] = {}
        self._zipf_probs: dict[tuple, np.ndarray] = {}
        self.user_habitual: list[set[tuple[int, int]]] = []
        for u in range(self.num_users):
            valid = self._policy_valid_actions(self.user_roles[u])
            if valid:
                k = max(1, len(valid) // 2)
                hab_idx = np.random.choice(len(valid), size=k, replace=False)
                self.user_habitual.append({valid[j] for j in hab_idx})
            else:
                self.user_habitual.append(set())
        self._refresh_action_space()

    def _init_state(self, start_time: int) -> None:
        """Mutable state: clock, device slots, fresh-slot pools, compromise bookkeeping."""
        self.t = start_time
        self.step_count = 0
        self.machine_slot = {m: self.dev_lo + m for m in range(self.num_devices)}
        # Optional fallback: non-TPM machines share a single guest device node
        self._guest_dev_slot: int | None = None
        if self.guest_device_fallback:
            non_tpm = [m for m in range(self.num_devices) if self.machine_tiers[m] < 2]
            if non_tpm:
                guest = self.dev_lo + non_tpm[0]
                self._guest_dev_slot = guest
                self.keys[guest] = GUEST_DEVICE
                self.node_features[guest, _NF_DEVICE_TIER] = 0.0
                for m in non_tpm:
                    self.machine_slot[m] = guest
                    if self.dev_lo + m != guest:
                        self.keys[self.dev_lo + m] = f"_guest_unused_dev_{m:04d}"
                        self.node_features[self.dev_lo + m, _NF_DEVICE_TIER] = 0.0
        # Fresh-slot allocators (recycled round-robin upon pool exhaustion)
        self._dev_pool = [self.dev_lo + self.num_devices + k
                          for k in range(self.num_wipe_slots + self.num_theft_slots)]
        self._src_pool = [self.src_lo + self.num_sources + k
                          for k in range(self.num_theft_slots + self.num_new_sources)]
        self._cfg_pool = [self.num_configs + k
                          for k in range(self.num_theft_slots + self.num_new_configs)]
        self._next_dev = self._next_src = self._next_cfg = 0
        self._slot_age: dict[int, int] = {}
        self.compromised_state: dict[int, int] = {}           # machine -> KillPhase
        self.compromise_incident: dict[int, int] = {}         # machine -> incident id
        self._next_incident = 0  # evaluation-only ids: no RNG draw, the stream is unchanged
        self.compromised_chain_remaining: dict[int, int] = {} # machine -> steps left in phase
        self.compromised_dwell: dict[int, int] = {}           # machine -> dwell events before remediation
        self.harvested_creds: dict[int, list[int]] = {}       # machine -> dumped credentials
        self.machine_logons: dict[int, set[int]] = {}         # machine -> hot-desk users
        self._active_thefts: list[_TheftIncident] = []
        self._cfg_upgrade: dict[int, int] = {}                # active JA3 release migrations
        self._admitted = self.num_devices                     # machines admitted so far

    # --- Policy and action sampling ---
    def policy_allows(self, role: str, method: int, uri: str) -> bool:
        """:func:`access_policy.policy_allows` against this simulator's catalogue."""
        return access_policy.policy_allows(role, method, uri, self.route_methods, self.security_matrix)

    def _policy_valid_actions(self, role: str):
        """``(resource_idx, method)`` actions OPA would ALLOW for this role (cached)."""
        if role not in self._valid_cache:
            self._valid_cache[role] = [
                (r, m)
                for r, uri in enumerate(self.resource_uris)
                for m in self.route_methods[uri]
                if self.policy_allows(role, m, uri)
            ]
        return self._valid_cache[role]

    def _policy_violations(self, role: str):
        """Denied (resource_idx, method) actions on protected routes for role (cached)."""
        if role not in self._violation_cache:
            self._violation_cache[role] = [
                (r, m)
                for r, uri in enumerate(self.resource_uris)
                for m in self.route_methods[uri]
                if uri in self.security_matrix and not self.policy_allows(role, m, uri)
            ]
        return self._violation_cache[role]

    def _user_actions(self, user: int, role: str):
        """Allowed actions split into (habitual, non_habitual) for user (cached)."""
        if user not in self._user_action_cache:
            valid = self._policy_valid_actions(role)
            hab = self.user_habitual[user]
            self._user_action_cache[user] = (
                [a for a in valid if a in hab],
                [a for a in valid if a not in hab],
            )
        return self._user_action_cache[user]

    def _refresh_action_space(self) -> None:
        """Every (resource, method) pair, and the public subset anonymous visitors use."""
        self._all_actions = [
            (r, m) for r, uri in enumerate(self.resource_uris) for m in self.route_methods[uri]
        ]
        self._anon_actions = [
            (r, m) for r, uri in enumerate(self.resource_uris)
            if uri not in self.security_matrix for m in self.route_methods[uri]
        ]

    def _zipf_choice(self, choices: list, key: tuple):
        """Sample an action weighted by resource popularity rank (Zipf law).

        Rank is decoupled from resource index so benign and attack traffic share
        identical destination marginals.
        """
        if not choices:
            return None
        p = self._zipf_probs.get(key)
        if p is None or len(p) != len(choices):
            w = self._res_pop_weight[[r for r, _ in choices]]
            p = w / w.sum()
            self._zipf_probs[key] = p
        idx = int(np.random.choice(len(choices), p=p))
        return choices[idx]

    # --- Fresh-slot allocation (shared by benign churn and attackers) ---
    def _new_cookie(self) -> str:
        """Unused opaque device cookie ``ck:<48-bit hex>``."""
        while True:
            key = f"ck:{random.getrandbits(48):012x}"
            if key not in self._used_cookies:
                self._used_cookies.add(key)
                return key

    @staticmethod
    def _fresh_ip_key(i: int) -> str:
        """Construct key for external/CGNAT source slot i."""
        return f"src:100.{64 + i // 65536}.{(i // 256) % 256}.{i % 256}"

    def _alloc_dev(self, tier: int) -> int | None:
        """Allocate a fresh device slot with a new opaque cookie token."""
        busy = set(self.machine_slot.values()) | {t.device for t in self._active_thefts}
        for _ in range(len(self._dev_pool)):
            slot = self._dev_pool[self._next_dev % len(self._dev_pool)]
            self._next_dev += 1
            if slot not in busy:
                break
        else:
            return None
        self.keys[slot] = self._new_cookie()
        self.node_features[slot, _NF_DEVICE_TIER] = tier / 2.0
        self._slot_age[slot] = 0
        return slot

    def _alloc_src(self) -> int:
        """Allocate a fresh client IP slot (round-robin)."""
        slot = self._src_pool[self._next_src % len(self._src_pool)]
        self._next_src += 1
        return slot

    def _alloc_cfg(self) -> int:
        """Allocate a fresh client JA3 config slot (round-robin); returns the local index."""
        local = self._cfg_pool[self._next_cfg % len(self._cfg_pool)]
        self._next_cfg += 1
        return local

    # --- Benign organisational dynamics ---
    def _maybe_wipe_cookie(self, machine: int) -> None:
        """Simulate cookie wipe: re-key cookie-identified machine to a fresh device slot."""
        if (
            not self.guest_device_fallback  # guest devices have no per-machine cookie
            and self.machine_tiers[machine] < 2
            and self._dev_pool
            and random.random() < self.p_cookie_wipe
        ):
            slot = self._alloc_dev(self.machine_tiers[machine])
            if slot is not None:
                self.machine_slot[machine] = slot

    def _maybe_release_config(self) -> None:
        """Simulate client software update (fleet-wide JA3 migration to fresh config)."""
        if self.p_config_release <= 0 or random.random() >= self.p_config_release:
            return
        in_use = sorted({c for cfgs in self.machine_configs for c in cfgs} - set(self._cfg_upgrade))
        if in_use:
            self._cfg_upgrade[int(random.choice(in_use))] = self._alloc_cfg()

    def _habitual_config(self, machine: int) -> int:
        """Global slot of machine's habitual config (occasionally conf:guest or pending upgrade)."""
        if random.random() < _P_GUEST_CONFIG:
            return self.cfg_lo  # conf:guest
        cfgs = self.machine_configs[machine]
        j = random.randrange(len(cfgs))
        new = self._cfg_upgrade.get(cfgs[j])
        if new is not None and random.random() < self.p_config_adopt:
            cfgs[j] = new
        return self.cfg_lo + int(cfgs[j])

    def _fleet_config(self) -> int:
        """Global slot of a config sampled with fleet popularity."""
        m = int(np.random.randint(0, self._admitted))
        return self.cfg_lo + int(random.choice(self.machine_configs[m]))

    def _new_tool_config(self, machine: int) -> int:
        """Global slot of a config not habitually used by machine (new tool tell)."""
        habit = set(self.machine_configs[machine])
        others = [c for c in range(1, self.num_configs) if c not in habit]
        if not others:
            return self._habitual_config(machine)
        return self.cfg_lo + int(random.choice(others))

    def _maybe_onboard(self) -> None:
        """Onboard next scheduled hire once their admission step is reached."""
        if not self._pending_hires or self.step_count < self._pending_hires[0][0]:
            return
        _, u = self._pending_hires.pop(0)
        seats = [m for m in range(self._admitted)
                 if self.service_machines is None or m not in self.service_machines]
        m = int(random.choice(seats)) if seats else 0
        self.machine_users[m].append(u)
        self._humans.append(u)
        self._registered.append(u)
        self._user_age[u] = 0

    def _admitted_machines(self) -> int:
        """Number of physical machines admitted by current step (all without a horizon)."""
        if not self.admission_horizon:
            return self.num_devices
        return min(self.num_devices,
                   int(self.step_count / self.admission_horizon * self.num_devices) + 1)

    def _admitted_guests(self) -> int:
        """Number of anonymous visitor identities admitted by current step."""
        if not (self.ramp_guests and self.admission_horizon):
            return self.num_guests
        return min(self.num_guests,
                   int(self.step_count / self.admission_horizon * self.num_guests) + 1)

    def _benign_sensors(self) -> tuple[float, float, float]:
        """Generate baseline sensor probe values (s1, s2, s3) misfiring at p_sensor_fp."""
        if self.p_sensor_fp <= 0:
            return 0.0, 0.0, 0.0
        s = (np.random.rand(3) < self.p_sensor_fp).astype(float)
        return float(s[0]), float(s[1]), float(s[2])

    # --- Compromise lifecycle ---
    def _compromise(self, machine: int) -> None:
        """Initialize compromise on machine: harvest cached or foreign credentials for pivot."""
        self.compromised_state[machine] = KillPhase.RECON
        self.compromise_incident[machine] = self._new_incident()
        cached = sorted(set(self.machine_users[machine]) | self.machine_logons.get(machine, set()))
        foreign = [u for u in self._registered if u not in cached]
        k = int(np.random.randint(1, 4))
        n_cached = int(np.random.binomial(k, self.p_harvest_cached)) if self.p_harvest_cached > 0 else 0
        creds = []
        for pool, n in ((cached, n_cached), (foreign, k - n_cached)):
            n = min(n, len(pool))
            if n:
                creds += [int(u) for u in np.random.choice(pool, size=n, replace=False)]
        if creds:
            self.harvested_creds[machine] = creds

    def _remediate(self, machine: int) -> None:
        """Clean the machine: drop all compromise bookkeeping."""
        for d in (self.compromised_state, self.compromised_chain_remaining,
                  self.compromised_dwell, self.harvested_creds, self.compromise_incident):
            d.pop(machine, None)

    def _new_incident(self) -> int:
        """Next incident id: one per theft session and per machine compromise episode."""
        self._next_incident += 1
        return self._next_incident - 1

    def _advance_kill_chain(self, machine: int) -> str:
        """Advance the machine's kill chain by one request; return the anomaly kind to emit.

        With ``p_compromise`` (multi-incident mode): recon 1-3 requests, lateral 5-11,
        dwell 0-4, then remediation. Without it (legacy): one recon request, then lateral,
        then dwell forever.
        """
        phase = self.compromised_state[machine]
        multi = self.p_compromise is not None
        if phase == KillPhase.RECON:
            kind = "context"  # sensor probes
            recon_left = self.compromised_chain_remaining.get(machine)
            if multi and recon_left is None:
                recon_left = int(np.random.randint(1, 4))
            if not multi or recon_left <= 1:
                self.compromised_state[machine] = KillPhase.LATERAL
                self.compromised_chain_remaining[machine] = int(np.random.randint(5, 12))
            else:
                self.compromised_chain_remaining[machine] = recon_left - 1
        elif phase == KillPhase.LATERAL:
            kind = "lateral"
            self.compromised_chain_remaining[machine] -= 1
            if self.compromised_chain_remaining[machine] <= 0:
                self.compromised_state[machine] = KillPhase.DWELL
                if multi:
                    self.compromised_dwell[machine] = int(np.random.randint(0, 5))
        else:
            kind = np.random.choice(["policy", "context", "lateral"])
        if (
            self.compromised_state.get(machine) == KillPhase.DWELL and multi
            and self.compromised_dwell.get(machine, 0) <= 0
        ):
            self._remediate(machine)
        elif phase == KillPhase.DWELL and multi:
            self.compromised_dwell[machine] -= 1
        return kind

    # --- Event construction ---
    @staticmethod
    def _msg(ja3: float, sensors, method: int, role: str, clearance: int) -> list[float]:
        """7-dim access-edge message ``[ja3, s1, s2, s3, method, role, clearance]``."""
        s1, s2, s3 = sensors
        return [ja3, float(s1), float(s2), float(s3), float(method),
                ROLES.index(role) / (len(ROLES) - 1), clearance / 4.0]

    def _emit(self, req: _Request, res_idx: int, method: int, ja3: float, sensors,
              etype: EventType) -> dict:
        """Event for ``req`` accessing ``res_idx``; label is 1 for every non-benign type."""
        return self._event(
            source=req.source, config=req.config, device=req.device, user=req.user,
            res_idx=res_idx, feat=self._msg(ja3, sensors, method, req.role, req.clearance),
            label=int(etype != EventType.BENIGN), etype=etype, scenario=req.scenario,
        )

    def _event(self, *, source, config, device, user, res_idx, feat, label, etype, scenario,
               incident=-1, theft_variant=0):
        """Event dict (global node ids + external keys); ages cold device/user nodes.

        ``incident`` (-1 = none) and ``theft_variant`` are evaluation-only ground truth.
        """
        if device in self._slot_age:
            self._slot_age[device] += 1
        if user in self._user_age:
            if self._user_age[user] < _NEW_USER_COLD_EVENTS:
                scenario |= SCEN_NEW_USER
            self._user_age[user] += 1

        dst = self.res_lo + res_idx
        return {
            "source": source, "config": config, "device": device, "user": user, "dst": dst,
            "t": self.t, "features": feat, "label": label, "etype": int(etype),
            "scenario": scenario, "incident": incident, "theft_variant": theft_variant,
            "key_source": self.keys[source], "key_config": self.keys[config],
            "key_device": self.keys[device],
            "key_user": self.keys[user], "key_dst": self.keys[dst],
        }

    # --- One event step ---
    def step(self) -> dict:
        """Generate the next event: credential theft, benign request, or kill-chain attack."""
        self._advance_clock()

        theft = self._maybe_theft_event()
        if theft is not None:
            return theft

        if self.p_compromise is not None and random.random() < self.p_compromise:
            # Compromise a clean admitted machine at the global rate
            victim_m = int(np.random.randint(0, self._admitted))
            if victim_m not in self.compromised_state:
                self._compromise(victim_m)
        req = self._sample_request()
        if (
            self.p_compromise is None  # legacy per-request hazard
            and np.random.rand() < _P_LEGACY_COMPROMISE and req.machine not in self.compromised_state
        ):
            self._compromise(req.machine)

        # Compromised hosts blend in with benign traffic most of the time
        if req.machine in self.compromised_state and np.random.rand() < _P_ATTACK_ON_COMPROMISED:
            # Read before the kill chain advances: remediation drops the episode id.
            incident = self.compromise_incident[req.machine]
            event = self._attack_event(req)
            event["incident"] = incident
            return event
        return self._benign_event(req)

    def _current_interarrival_scale(self) -> float:
        """Scale parameter for exponential inter-arrival time reflecting circadian activity."""
        day_sec = self.t % _DAY_S
        weekday = (self.t // _DAY_S) % 7
        if weekday >= 5:
            return _SCALE_WEEKEND
        if _WORK_START_S <= day_sec <= _WORK_END_S:
            return _SCALE_WORK
        return _SCALE_NIGHT

    def _advance_clock(self) -> None:
        """Advance time and the admission horizon; apply fleet-wide churn (releases, hires)."""
        self.t += max(1, int(np.random.exponential(scale=self._current_interarrival_scale())))
        self.step_count += 1
        self._admitted = self._admitted_machines()
        self._maybe_release_config()
        self._maybe_onboard()

    def _maybe_theft_event(self) -> dict | None:
        """Continue an active theft session (interleaved with user traffic) or start a new one."""
        if self._active_thefts and random.random() < self.p_theft_interleave:
            return self._emit_theft_event(random.choice(self._active_thefts))
        if (
            random.random() < self.p_cred_theft and self._src_pool and self._cfg_pool
            and (self._dev_pool or self._guest_dev_slot is not None)
        ):
            incident = self._start_theft()
            self._active_thefts.append(incident)
            return self._emit_theft_event(incident)
        return None

    def _start_theft(self) -> _TheftIncident:
        """Pick a victim and the attacker's device, IP and client (fresh or mimicked)."""
        victim = self._registered[int(np.random.randint(0, len(self._registered)))]
        victim_machines = [
            m for m in range(self._admitted) if victim in self.machine_users[m]
            and self.machine_tiers[m] < 2  # TPM-bound identities cannot be stolen
        ]
        replay_m = None
        variant = 0
        if victim_machines and random.random() < self.p_theft_session_replay:
            # Pass-the-cookie: replay victim device cookie
            replay_m = int(random.choice(victim_machines))
            dev_slot = self.machine_slot[replay_m]
            variant |= THEFT_REPLAY
        elif self.guest_device_fallback and self._guest_dev_slot is not None:
            dev_slot = self._guest_dev_slot
        else:
            cert = any(self.machine_tiers[m] == 1 for m in victim_machines)
            dev_slot = self._alloc_dev(
                tier=1 if cert and random.random() < self.p_theft_mimic_config else 0
            )
            if dev_slot is None:
                dev_slot = self.machine_slot[int(np.random.randint(0, self._admitted))]
        # Attacker network: fleet egress IP or fresh IP
        if random.random() < self.p_theft_known_source:
            src_slot = self.src_lo + int(np.random.randint(self._num_office, self.num_sources))
            variant |= THEFT_KNOWN_SRC
        else:
            src_slot = self._alloc_src()
        if random.random() < self.p_theft_mimic_config:
            # Replay victim client config or sample common fleet client
            cfg_slot = (
                self.cfg_lo + int(random.choice(self.machine_configs[replay_m]))
                if replay_m is not None else self._fleet_config()
            )
            variant |= THEFT_MIMIC_CFG
        else:
            cfg_slot = self.cfg_lo + self._alloc_cfg()
        return _TheftIncident(victim=victim, device=dev_slot, source=src_slot, config=cfg_slot,
                              remaining=int(np.random.randint(3, 7)),
                              incident=self._new_incident(), variant=variant)

    def _emit_theft_event(self, incident: _TheftIncident) -> dict:
        """Emit one credential-theft request: attacker IP/device/config with victim credentials."""
        u = incident.victim
        role, clr = self.user_roles[u], self.user_clearances[u]
        # Destination drawn from valid actions for role to preserve destination marginal
        valid = self._policy_valid_actions(role)
        res_idx, method = (self._zipf_choice(valid, ("valid", role)) if valid else (0, 0))
        # Credential theft is policy-clean and signal-clean; exposed only by broken binding
        feat = self._msg(1.0, self._benign_sensors(), method, role, clr)
        incident.remaining -= 1
        if incident.remaining <= 0:
            self._active_thefts.remove(incident)
        return self._event(
            source=incident.source, config=incident.config, device=incident.device, user=u,
            res_idx=res_idx, feat=feat, label=1, etype=EventType.CRED_THEFT, scenario=0,
            incident=incident.incident, theft_variant=incident.variant,
        )

    def _set_user(self, req: _Request, user: int) -> None:
        """Make ``user`` the requester, with the role and clearance of their account."""
        req.user = user
        req.role, req.clearance = self.user_roles[user], self.user_clearances[user]

    def _sample_request(self) -> _Request:
        """Pick machine (progressively admitted), user, source IP and client config."""
        machine = int(np.random.randint(0, self._admitted))
        self._maybe_wipe_cookie(machine)
        dev_slot = self.machine_slot[machine]
        user = int(random.choice(self.machine_users[machine]))
        if self.p_hotdesk > 0 and random.random() < self.p_hotdesk:
            # Hot-desking: registered user signing in on another machine
            user = int(random.choice(self._humans))
            self.machine_logons.setdefault(machine, set()).add(user)

        scenario = 0
        if len(self.machine_users[machine]) > 1:
            scenario |= SCEN_SHARED
        if self._slot_age.get(dev_slot, _WIPE_COLD_EVENTS) < _WIPE_COLD_EVENTS:
            scenario |= SCEN_WIPED

        home = self.machine_home_ips[machine]
        if random.random() < self.p_roam:
            scenario |= SCEN_ROAMING
            if self.p_new_source > 0 and self._src_pool and random.random() < self.p_new_source:
                # Fresh roaming IP from shared pool
                source = self._alloc_src()
            else:
                src_local = int(np.random.randint(0, self.num_sources))
                while src_local in home:
                    src_local = int(np.random.randint(0, self.num_sources))
                source = self.src_lo + src_local
        else:
            source = self.src_lo + random.choice(sorted(home))

        # Client config (TLS/JA3): habitual fingerprint or occasional fresh client
        config = self._habitual_config(machine)
        if self.p_benign_new_config > 0 and self._cfg_pool and random.random() < self.p_benign_new_config:
            config = self.cfg_lo + self._alloc_cfg()

        return _Request(machine=machine, device=dev_slot, user=user,
                        role=self.user_roles[user], clearance=self.user_clearances[user],
                        source=source, config=config, scenario=scenario)

    # --- Benign branch ---
    def _benign_event(self, req: _Request) -> dict:
        """Benign request: service cronjob, user mistake (etype 6), visitor or habitual access."""
        is_anonymous = random.random() < _P_ANONYMOUS

        if random.random() < _P_SERVICE_ACCOUNT:
            event = self._service_account_event(req)
            if event is not None:
                return event
            # No allowed action: continue as a regular request of the service account

        if not is_anonymous and random.random() < _P_BENIGN_DENIAL:
            invalid = self._policy_violations(req.role)
            if invalid:
                res_idx, method = self._zipf_choice(invalid, ("viol", req.role))
                return self._emit(req, res_idx, method, 1.0, self._benign_sensors(),
                                  EventType.BENIGN_DENIAL)

        if is_anonymous:
            req.user = self.num_registered_users + int(np.random.randint(0, self._admitted_guests()))
            req.role, req.clearance = "guest", 0
            req.config = self.cfg_lo  # conf:guest
            if self._guest_dev_slot is not None:
                req.device = self._guest_dev_slot
            res_idx, method = self._zipf_choice(self._anon_actions, ("anon",))
        else:
            valid = self._policy_valid_actions(req.role)
            habit, non_habit = self._user_actions(req.user, req.role)
            # Benign exploration: authorized non-habitual access
            if non_habit and random.random() < self.benign_explore_prob:
                res_idx, method = self._zipf_choice(non_habit, ("nonhabit", req.user))
            elif habit:
                res_idx, method = self._zipf_choice(habit, ("habit", req.user))
            elif valid:
                res_idx, method = self._zipf_choice(valid, ("valid", req.role))
            else:
                res_idx, method = 0, 0  # public-path fallback
        return self._emit(req, res_idx, method, 1.0, self._benign_sensors(), EventType.BENIGN)

    def _service_account_event(self, req: _Request) -> dict | None:
        """Deterministic cronjob of the service account (its first allowed action).

        Rewrites ``req`` to the service identity (and a service machine, if any) even
        when no event is emitted, so the caller continues as the service account.
        """
        self._set_user(req, _SERVICE_USER)
        req.config = self.cfg_lo + _SERVICE_CONFIG
        if self.service_machines is not None:
            sm = int(random.choice(self.service_machines))
            req.machine, req.device = sm, self.machine_slot[sm]
            req.source = self.src_lo + random.choice(sorted(self.machine_home_ips[sm]))
            req.scenario = SCEN_SHARED if len(self.machine_users[sm]) > 1 else 0
        valid = self._policy_valid_actions(req.role)
        if not valid:
            return None
        res_idx, method = valid[0]
        return self._emit(req, res_idx, method, 1.0, self._benign_sensors(), EventType.BENIGN)

    # --- Attack branch ---
    def _attack_event(self, req: _Request) -> dict:
        """Kill-chain request from a compromised machine: lateral, policy or context anomaly.

        Falls back lateral -> policy (no target) -> context (no denied route for the role).
        """
        kind = self._advance_kill_chain(req.machine)

        if kind == "lateral":
            event = self._lateral_event(req)
            if event is not None:
                return event
            kind = "policy"

        if kind == "policy":
            # Policy denial for this role (read-up, write-down, missing compartment)
            invalid = self._policy_violations(req.role)
            if invalid:
                res_idx, method = self._zipf_choice(invalid, ("viol", req.role))
                return self._emit(req, res_idx, method, 1.0, self._benign_sensors(),
                                  EventType.POLICY)

        # Contextual anomaly (scanner probes / recon)
        valid = self._policy_valid_actions(req.role)
        if valid:
            res_idx, method = self._zipf_choice(valid, ("valid", req.role))
        else:
            res_idx, method = self._zipf_choice(self._all_actions, ("all",))
        ja3 = 0.0 if np.random.rand() > 0.5 else 1.0
        # Sensor probe trigger rates on external scan
        s1 = 1.0 if np.random.rand() > 0.2 else 0.0
        s2 = 1.0 if np.random.rand() > 0.5 else 0.0
        s3 = 1.0 if np.random.rand() > 0.8 else 0.0
        return self._emit(req, res_idx, method, ja3, (s1, s2, s3), EventType.CONTEXT)

    def _lateral_event(self, req: _Request) -> dict | None:
        """Lateral movement: pivot on harvested credentials or own non-habitual access.

        Returns None when there is no target; ``req`` keeps any pivot identity.
        """
        creds = [u for u in self.harvested_creds.get(req.machine, ()) if u != req.user]
        pivot = bool(creds) and random.random() < self.p_lateral_foreign_cred
        if pivot:
            # Pivot using harvested credentials (cached or foreign identity)
            self._set_user(req, int(random.choice(creds)))
            valid = self._policy_valid_actions(req.role)
            target = self._zipf_choice(valid, ("valid", req.role)) if valid else None
        else:
            # Own-identity lateral movement (authorized non-habitual access)
            _habit, non_habit = self._user_actions(req.user, req.role)
            target = self._zipf_choice(non_habit, ("nonhabit", req.user)) if non_habit else None
        if target is None:
            return None

        res_idx, method = target
        ja3 = 1.0
        # Lateral movement: stealth — legitimate credentials and protocols,
        # rarely triggers the IDS (the network must study the graph).
        # --- OLD PARAMS ---
        #s1 = 0.0
        #s2 = 1.0 if np.random.rand() > 0.98 else 0.0  # 2%
        #s3 = 1.0 if np.random.rand() > 0.90 else 0.0  # 10%

        # --- NEW PARAMS ---
        s1, s2, s3 = self._benign_sensors()

        if not pivot and random.random() < self.p_lateral_role_spoof:
            # Role claim disagreement (tested via p_lateral_role_spoof)
            allowed_roles = [
                r for r in ROLES
                if r != req.role and self.policy_allows(r, method, self.resource_uris[res_idx])
            ]
            if allowed_roles:
                req.role = random.choice(allowed_roles)
                req.clearance = ROLE_CLEARANCE[req.role]
        if random.random() < self.p_lateral_new_config:
            # New tool execution on host
            req.config = self._new_tool_config(req.machine)
        return self._emit(req, res_idx, method, ja3, (s1, s2, s3), EventType.LATERAL)

    # --- Live API ---
    def add_resource(self, uri: str, methods: set[int], classification: str | None = None,
                     categories: set[str] | None = None, risk: float = 0.5) -> None:
        """Append a resource node at runtime (lowest popularity); refreshes the action caches."""
        if uri in self.resource_uris:
            return

        self.route_methods[uri] = methods
        if classification and categories:
            self.security_matrix[uri] = (classification, categories)
        self.resource_risk[uri] = risk
        self.resource_uris.append(uri)
        self.keys.append(uri)

        new_feat = torch.zeros(1, _NODE_FEAT_DIM)
        new_feat[0, _NF_TRUST] = 1.0
        if self.use_resource_risk:
            new_feat[0, _NF_RESOURCE_RISK] = risk
        self.node_features = torch.cat([self.node_features, new_feat], dim=0)

        new_weight = np.array([_zipf_weight(float(len(self._res_pop_weight)))], dtype=np.float64)
        self._res_pop_weight = np.concatenate([self._res_pop_weight, new_weight])

        self.num_resources += 1
        self.num_nodes += 1

        self._valid_cache.clear()
        self._violation_cache.clear()
        self._user_action_cache.clear()
        self._zipf_probs.clear()
        self._refresh_action_space()


@dataclass
class SyntheticStream:
    """Tensorised event stream and node indexing metadata for training and evaluation."""

    source: torch.Tensor        # [N] global source (IP) node ids
    config: torch.Tensor        # [N] global config (JA3) node ids
    device: torch.Tensor        # [N] global device node ids
    user: torch.Tensor          # [N] global user node ids
    dst: torch.Tensor           # [N] global resource node ids
    t: torch.Tensor             # [N] timestamps
    msg: torch.Tensor           # [N, msg_dim=7] edge messages (access edge)
    y: torch.Tensor             # [N] binary labels
    types: torch.Tensor         # [N] EventType values
    scenario: torch.Tensor      # [N] benign-context bitmask (SCEN_*)
    node_features: torch.Tensor  # [num_nodes, 16]
    keys: list = field(repr=False)
    num_nodes: int = 0
    user_lo: int = 0
    user_num: int = 0
    dev_lo: int = 0
    dev_num: int = 0
    src_lo: int = 0
    src_num: int = 0
    cfg_lo: int = 0
    cfg_num: int = 0
    res_lo: int = 0
    res_num: int = 0
    # Evaluation-only ground truth: incident id per event (-1 = none; one per theft session
    # and per machine compromise episode) and the THEFT_* bits of credential-theft events.
    incident: torch.Tensor | None = None
    theft_variant: torch.Tensor | None = None


def generate_streaming_data(num_events: int = 50000, **sim_kwargs) -> SyntheticStream:
    """Run a :class:`ZTAStreamSimulator` for ``num_events`` steps and tensorise the stream.

    ``sim_kwargs`` are simulator arguments (build them with :func:`stream_kwargs_from_cfg`);
    the admission horizon is the stream length.
    """
    sim = ZTAStreamSimulator(admission_horizon=num_events, **sim_kwargs)
    events = [sim.step() for _ in range(num_events)]

    def col(name, dtype=torch.long):
        return torch.tensor([e[name] for e in events], dtype=dtype)

    # Spare slots never touched by the run keep placeholder keys; that is fine — the
    # registry just preregisters them and serving re-keys slots dynamically anyway.
    return SyntheticStream(
        source=col("source"), config=col("config"), device=col("device"),
        user=col("user"), dst=col("dst"),
        t=col("t"),
        msg=torch.tensor([e["features"] for e in events], dtype=torch.float),
        y=col("label"), types=col("etype"), scenario=col("scenario"),
        node_features=sim.node_features, keys=list(sim.keys),
        num_nodes=sim.num_nodes,
        user_lo=sim.user_lo, user_num=sim.num_users,
        dev_lo=sim.dev_lo, dev_num=sim.dev_slots,
        src_lo=sim.src_lo, src_num=sim.src_slots,
        cfg_lo=sim.cfg_lo, cfg_num=sim.cfg_slots,
        res_lo=sim.res_lo, res_num=sim.num_resources,
        incident=col("incident"), theft_variant=col("theft_variant"),
    )
