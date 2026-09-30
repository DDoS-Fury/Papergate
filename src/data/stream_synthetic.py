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

Anomaly Types (``types``):
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

import numpy as np
import torch

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

ROLES = ["guest", "operator", "manager", "admin"]
CLEARANCES = ["PUBLIC", "INTERNAL", "CONFIDENTIAL", "SECRET", "TOP_SECRET"]
WRITE_METHODS = {1, 2, 3, 4}  # POST/PUT/DELETE/PATCH (0=GET is the only read)

# --- Reference Authorization Model (Bell-LaPadula + Compartments) --------------------
# Reference security policy model used for ground-truth authorization decisions:
#   * Simple Security Property: no read-up on GET (clearance >= classification).
#   * *-Property: no write-down on writes (clearance <= classification).
#   * Compartments: required categories must be a subset of the subject's categories.
#   * Trusted Guard: sanitized write-down exception allowing admin POST on TRUSTED_GUARD.
# Note: Deployed OPA policies should mirror this reference model to keep the benign
# manifold aligned with PDP decisions.
SECURITY_LEVELS = {
    "PUBLIC": 0, "INTERNAL": 1, "CONFIDENTIAL": 2, "SECRET": 3, "TOP_SECRET": 4,
}
ROLE_CLEARANCE = {  # Clearance derives directly from role
    "guest": 0, "operator": 1, "manager": 2, "admin": 4,
}
ROLE_CATEGORIES = {  # Compartments granted per role
    "guest": set(),
    "operator": {"hr", "ops"},
    "manager": {"hr", "ops", "finance"},
    "admin": {"hr", "ops", "finance", "nuclear", "security"},
}

TRUSTED_GUARD = "/api/v1/trusted-guard/sanitized-delete-personnel"

# Protected routes: uri -> (classification, required categories)
SECURITY_MATRIX = {
    "/api/v1/personnel":          ("INTERNAL",     {"hr"}),
    "/api/v1/documents":          ("CONFIDENTIAL", {"finance"}),
    "/api/v1/nuclear-materials":  ("TOP_SECRET",   {"nuclear"}),
    "/api/v1/reactor-parameters": ("TOP_SECRET",   {"nuclear", "security"}),
    TRUSTED_GUARD:                ("SECRET",       {"security"}),
}

# Methods served per route (0=GET, 1=POST, 2=PUT, 3=DELETE, 4=PATCH).
# Protected routes expose methods matching their security profile; public/auth routes
# are accessible to all roles and gated by PDP risk evaluation.
_GET, _POST = {0}, {1}
ROUTE_METHODS = {
    "/": _GET,
    "/materials": _GET,
    "/reserved": _GET,
    "/login": _GET,
    "/register": _GET,
    "/static": _GET,
    "/favicon.ico": _GET,
    "/api/v1/auth/register": _POST,
    "/api/v1/auth/login": _POST,
    "/api/v1/auth/verify-otp": _POST,
    "/api/v1/auth/register/begin": _POST,
    "/api/v1/auth/register/finish": _POST,
    "/api/v1/auth/login/begin": _POST,
    "/api/v1/auth/login/finish": _POST,
    "/api/v1/personnel": {0, 1},              # GET, POST
    "/api/v1/documents": {0, 1, 3},           # GET, POST, DELETE
    "/api/v1/nuclear-materials": {0, 1, 3},   # GET, POST, DELETE
    "/api/v1/reactor-parameters": {0, 1, 3},  # GET, POST, DELETE
    TRUSTED_GUARD: {1},                       # POST only
}

# Base orchestrator routes. Simulator instances expand this into a larger synthetic
# resource catalogue in a seed-dependent manner (see build_resource_universe).
_BASE_ROUTE_METHODS = dict(ROUTE_METHODS)
_BASE_SECURITY_MATRIX = dict(SECURITY_MATRIX)

_GENERATED_CATEGORIES = ("public", "hr", "finance", "nuclear", "ops", "security")
_GENERATED_CLASSIFICATION = {
    "hr": ("INTERNAL", {"hr"}),
    "finance": ("CONFIDENTIAL", {"finance"}),
    "ops": ("INTERNAL", {"ops"}),
    "nuclear": ("TOP_SECRET", {"nuclear"}),
    "security": ("SECRET", {"security"}),
}

# Inherent resource risk per security level (stored at node_features[:, 4]).
_CLASSIFICATION_RISK = {
    "INTERNAL": 0.5, "CONFIDENTIAL": 0.6, "SECRET": 0.8, "TOP_SECRET": 0.9,
}
# Risk overrides for base orchestrator routes.
_RISK_OVERRIDES = {
    "/api/v1/personnel": 0.6,
    "/api/v1/documents": 0.7,
    "/api/v1/nuclear-materials": 0.7,
    "/api/v1/reactor-parameters": 1.0,
    TRUSTED_GUARD: 1.0,
}


def build_resource_universe(num_generated: int = 981, seed: int = 42):
    """Build the resource catalogue combining base routes with synthetic endpoints.

    Returns (route_methods, security_matrix, resource_uris, resource_risk).
    Generation uses random.Random(seed) so each run obtains a distinct resource catalog.
    Resource keys match normalized URI paths sent by the orchestrator.
    """
    route_methods = dict(_BASE_ROUTE_METHODS)
    security_matrix = dict(_BASE_SECURITY_MATRIX)

    rng = random.Random(seed)
    for i in range(num_generated):
        cat = rng.choice(_GENERATED_CATEGORIES)
        if cat == "public":
            uri = f"/api/v2/public/resource_{i}"
            route_methods[uri] = {0, 1}
        else:
            uri = f"/internal/{cat}/doc_{i}"
            route_methods[uri] = {0, 1, 3}
            security_matrix[uri] = _GENERATED_CLASSIFICATION[cat]

    resource_uris = list(route_methods)
    resource_risk = {uri: 0.0 for uri in resource_uris}
    for uri, (cls, _cats) in security_matrix.items():
        resource_risk[uri] = _CLASSIFICATION_RISK[cls]
    resource_risk.update(_RISK_OVERRIDES)
    return route_methods, security_matrix, resource_uris, resource_risk


# Module-level defaults (seed 42) for policy/netclass testing.
ROUTE_METHODS, SECURITY_MATRIX, RESOURCE_URIS, RESOURCE_RISK = build_resource_universe()


def policy_allows(role: str, method: int, uri: str) -> bool:
    """Evaluate access authorization under the reference Bell-LaPadula policy.

    Public routes allow all roles. Protected routes verify:
      1. Route serves the requested HTTP method.
      2. Role possesses all required compartments (categories).
      3. Bell-LaPadula: no read-up on GET, no write-down on writes,
         with trusted-guard sanitized write-down exception for admin.
    """
    if method not in ROUTE_METHODS.get(uri, set()):
        return False
    if uri not in SECURITY_MATRIX:
        return True  # public / auth / static route
    classification, categories = SECURITY_MATRIX[uri]
    if not categories.issubset(ROLE_CATEGORIES[role]):
        return False
    clr = ROLE_CLEARANCE[role]
    obj = SECURITY_LEVELS[classification]
    if uri == TRUSTED_GUARD:
        return role == "admin" and method == 1  # sanitized write-down (admin POST only)
    if method == 0:  # GET — Simple Security Property (no read-up)
        return clr >= obj
    return clr <= obj  # writes — *-Property (no write-down)

# Scenario bitmask annotations for benign-context evaluation.
SCEN_ROAMING = 1   # Non-home IP (remote work / 5G)
SCEN_WIPED = 2     # Recently re-keyed cookie (cold device node)
SCEN_SHARED = 4    # Workstation shared across multiple users
SCEN_NEW_USER = 8  # Recently onboarded user (cold user node)

_WIPE_COLD_EVENTS = 25      # Events until a re-keyed device is considered warm
_NEW_USER_COLD_EVENTS = 25  # Events until a new user node is considered warm

# Probability that a benign request uses an unfingerprinted client (conf:guest)
_P_GUEST_CONFIG = 0.05


class ZTAStreamSimulator:
    """Stateful generator for synthetic ZTA access event streams.

    Powers both the offline tensor stream (generate_streaming_data) and live API
    streaming. Simulates progressive device/entity admission, benign organizational
    dynamics, and stealthy attack kill chains.
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
        # --- v5 realism / difficulty knobs (see the "Open world" section of the module
        # docstring). Every default below reproduces the v4 *process*; TGNConfig sets the
        # published values.
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

        # Per-instance resource catalogue drawn under this run's seed (see
        # ``build_resource_universe``): the real routes plus a synthetic estate sized so
        # the total matches ``num_resources``.
        n_generated = num_resources - len(_BASE_ROUTE_METHODS)
        assert n_generated >= 0, (
            f"num_resources ({num_resources}) must be >= the {len(_BASE_ROUTE_METHODS)} "
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
        self.num_registered_users = num_users
        self.num_guests = num_guests
        self.num_users = num_users + num_guests
        self.num_devices = num_devices
        self.num_sources = num_sources
        self.num_configs = num_configs
        self.num_resources = num_resources
        self.num_wipe_slots = num_wipe_slots
        self.num_theft_slots = num_theft_slots
        self.guest_device_fallback = guest_device_fallback
        self.benign_explore_prob = benign_explore_prob
        self.p_roam = p_roam
        self.p_cookie_wipe = p_cookie_wipe
        self.p_cred_theft = p_cred_theft
        self.admission_horizon = admission_horizon
        self.use_resource_risk = use_resource_risk
        self.num_new_sources = num_new_sources
        self.num_new_configs = num_new_configs
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

        # --- Resource Popularity (Decoupled from index) ---
        # Popularity follows a Zipf distribution. Rank is a random permutation of the
        # resource index space to prevent destination IDs from encoding frequency.
        pop_rank = np.random.permutation(num_resources).astype(np.float64)
        self._res_pop_weight = 1.0 / ((pop_rank + 1.0) ** 1.2)

        # --- Node Index Layout: [users][device slots][source slots][config slots][resources] ---
        self.user_lo = 0
        self.dev_lo = self.num_users
        self.dev_slots = num_devices + num_wipe_slots + num_theft_slots
        self.src_lo = self.dev_lo + self.dev_slots
        # Trailing slots form shared fresh pools for benign churn and attackers
        self.src_slots = num_sources + num_theft_slots + num_new_sources
        self.cfg_lo = self.src_lo + self.src_slots
        self.cfg_slots = num_configs + num_theft_slots + num_new_configs
        self.res_lo = self.cfg_lo + self.cfg_slots
        self.num_nodes = self.res_lo + num_resources

        # --- Users and Clearances ---
        self.user_roles = [str(np.random.choice(ROLES)) for _ in range(self.num_registered_users)]
        self.user_roles.extend(["guest"] * self.num_guests)
        self.user_clearances = [ROLE_CLEARANCE[r] for r in self.user_roles]

        # --- Physical Machines ---
        # Tier: 0=unmanaged, 1=cert-only, 2=TPM-backed.
        self.machine_tiers = [int(np.random.choice([0, 1, 2], p=tier_mix))
                              for _ in range(num_devices)]
        # Desk owners (user 0 reserved as dedicated service account if service machines exist)
        self._humans = (
            list(range(1, self.num_registered_users))
            if num_service_machines is not None and self.num_registered_users > 1
            else list(range(self.num_registered_users))
        )
        # Mid-stream hires: registered users arriving progressively across the admission horizon
        self._pending_hires: list[tuple[int, int]] = []
        if num_new_users > 0 and admission_horizon:
            cand = [u for u in self._humans if u != 0]
            k = min(num_new_users, max(len(cand) - 1, 0))
            hires = [int(u) for u in np.random.permutation(cand)[:k]]
            self._pending_hires = [
                (int((i + np.random.rand()) * admission_horizon / k), u)
                for i, u in enumerate(hires)
            ]
            pending = set(hires)
            self._humans = [u for u in self._humans if u not in pending]
        pending = {u for _, u in self._pending_hires}
        self._registered = [u for u in range(self.num_registered_users) if u not in pending]
        self._user_age: dict[int, int] = {}
        self.machine_users: list[list[int]] = []
        for m in range(num_devices):
            users = [self._humans[m % len(self._humans)]]
            if np.random.rand() < p_shared_device:
                extra = np.random.randint(1, 4)
                pool = [u for u in self._humans if u not in users]
                users += list(np.random.choice(pool, size=min(extra, len(pool)), replace=False))
            self.machine_users.append(users)

        # Home IPs: office subnet (RFC1918 NAT) and remote/home subnet
        num_office = min(30, num_sources)
        self._office_locals = list(range(num_office))
        self.machine_home_ips: list[set[int]] = []
        for m in range(num_devices):
            home = {int(np.random.choice(self._office_locals))}
            if num_sources > num_office:
                home.add(int(np.random.randint(num_office, num_sources)))
            self.machine_home_ips.append(home)

        # Habitual client configurations (TLS/JA3) per machine
        cfg_pool = list(range(1, num_configs)) if num_configs > 1 else [0]
        self.machine_configs: list[list[int]] = []
        for m in range(num_devices):
            k = min(int(np.random.randint(1, 3)), len(cfg_pool))
            cfgs = np.random.choice(cfg_pool, size=k, replace=False)
            self.machine_configs.append([int(c) for c in cfgs])

        # Service machines for user 0 (automated tasks / cronjobs)
        self.service_machines = (
            None if num_service_machines is None
            else list(range(min(num_service_machines, num_devices)))
        )

        # --- External Keys per Node Slot ---
        self.keys: list[str | None] = [None] * self.num_nodes
        for u in range(self.num_registered_users):
            self.keys[self.user_lo + u] = f"user_{u:04d}"
        for g in range(self.num_guests):
            self.keys[self.user_lo + self.num_registered_users + g] = f"guest_{g:04d}"
        # Device keys: TPM-backed or random opaque cookies (ck:<hex>)
        self._used_cookies: set[str] = set()
        for m in range(num_devices):
            tier = self.machine_tiers[m]
            self.keys[self.dev_lo + m] = f"tpm:{m:04d}" if tier == 2 else self._new_cookie()
        for k in range(num_wipe_slots + num_theft_slots):
            self.keys[self.dev_lo + num_devices + k] = f"_spare_dev_{k}"
        # Source keys: internal office IPs (10.0.0.x) and external/CGNAT IPs (100.64.x.x)
        for s in range(num_sources):
            self.keys[self.src_lo + s] = (
                f"src:10.0.0.{s}" if s < num_office else f"src:100.64.{s // 256}.{s % 256}"
            )
        for k in range(num_theft_slots + num_new_sources):
            self.keys[self.src_lo + num_sources + k] = self._fresh_ip_key(num_sources + k)
        # Config keys: 0=conf:guest, 1..num_configs=habitual, trailing=fresh pool
        self.keys[self.cfg_lo] = "conf:guest"
        for c in range(1, num_configs):
            self.keys[self.cfg_lo + c] = f"conf:{c:04d}"
        for k in range(num_theft_slots + num_new_configs):
            self.keys[self.cfg_lo + num_configs + k] = f"conf:{num_configs + k:04d}"
        for r in range(num_resources):
            self.keys[self.res_lo + r] = self.resource_uris[r]

        # --- Static Node Features (16-dim) ---
        # Map: [2]=device tier, [3]=reserved (0.0), [4]=resource risk,
        # [5]=internal source IP flag (1.0 internal, 0.0 external), [14]=trust score default (1.0).
        nf = torch.zeros(self.num_nodes, 16)
        nf[:, 14] = 1.0  # trust score default
        for m in range(num_devices):
            nf[self.dev_lo + m, 2] = self.machine_tiers[m] / 2.0
        for r in range(num_resources):
            if use_resource_risk:
                nf[self.res_lo + r, 4] = self.resource_risk[self.resource_uris[r]]
        if use_source_internal:
            for s in range(self.src_slots):
                nf[self.src_lo + s, 5] = 1.0 if ip_is_internal(self.keys[self.src_lo + s]) else 0.0
        self.node_features = nf

        # --- Behaviour Model ---
        # Valid actions per role under policy; habitual action subset per user
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

        # Complete candidate (resource, method) action space
        self._all_actions = [
            (r, m) for r, uri in enumerate(self.resource_uris) for m in self.route_methods[uri]
        ]

        # --- Mutable State ---
        self.t = start_time
        self.step_count = 0
        self.machine_slot = {m: self.dev_lo + m for m in range(num_devices)}
        # Optional fallback: non-TPM machines share a single guest device node
        self._guest_dev_slot: int | None = None
        if self.guest_device_fallback:
            non_tpm = [m for m in range(num_devices) if self.machine_tiers[m] < 2]
            if non_tpm:
                guest = self.dev_lo + non_tpm[0]
                self._guest_dev_slot = guest
                self.keys[guest] = GUEST_DEVICE
                self.node_features[guest, 2] = 0.0
                for m in non_tpm:
                    self.machine_slot[m] = guest
                    if self.dev_lo + m != guest:
                        self.keys[self.dev_lo + m] = f"_guest_unused_dev_{m:04d}"
                        self.node_features[self.dev_lo + m, 2] = 0.0
        # Fresh-slot allocators (recycled round-robin upon pool exhaustion)
        self._dev_pool = [self.dev_lo + num_devices + k for k in range(num_wipe_slots + num_theft_slots)]
        self._src_pool = [self.src_lo + num_sources + k for k in range(num_theft_slots + num_new_sources)]
        self._cfg_pool = [num_configs + k for k in range(num_theft_slots + num_new_configs)]
        self._next_dev = self._next_src = self._next_cfg = 0
        self._slot_age: dict[int, int] = {}
        self.compromised_state: dict[int, int] = {}           # machine -> kill-chain phase
        self.compromised_chain_remaining: dict[int, int] = {} # machine -> steps left in phase
        self.compromised_dwell: dict[int, int] = {}           # machine -> dwell events before remediation
        self.harvested_creds: dict[int, list[int]] = {}       # machine -> dumped credentials
        self.machine_logons: dict[int, set[int]] = {}         # machine -> hot-desk users
        self._active_thefts: list[dict] = []
        self._cfg_upgrade: dict[int, int] = {}                # active JA3 release migrations
        self._admitted = num_devices                          # admitted machines horizon

    # --- helpers ---
    def policy_allows(self, role: str, method: int, uri: str) -> bool:
        if method not in self.route_methods.get(uri, set()):
            return False
        if uri not in self.security_matrix:
            return True
        classification, categories = self.security_matrix[uri]
        if not categories.issubset(ROLE_CATEGORIES[role]):
            return False
        clr = ROLE_CLEARANCE[role]
        obj = SECURITY_LEVELS[classification]
        if uri == TRUSTED_GUARD:
            return role == "admin" and method == 1
        if method == 0:
            return clr >= obj
        return clr <= obj

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

    # --- Fresh-Slot Allocation (shared by benign churn and attackers) ---
    def _new_cookie(self) -> str:
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
        busy = set(self.machine_slot.values()) | {t["dev_slot"] for t in self._active_thefts}
        for _ in range(len(self._dev_pool)):
            slot = self._dev_pool[self._next_dev % len(self._dev_pool)]
            self._next_dev += 1
            if slot not in busy:
                break
        else:
            return None
        self.keys[slot] = self._new_cookie()
        self.node_features[slot, 2] = tier / 2.0
        self._slot_age[slot] = 0
        return slot

    def _alloc_src(self) -> int:
        """Allocate a fresh client IP slot (round-robin)."""
        slot = self._src_pool[self._next_src % len(self._src_pool)]
        self._next_src += 1
        return slot

    def _alloc_cfg(self) -> int:
        """Allocate a fresh client JA3 config slot (round-robin)."""
        local = self._cfg_pool[self._next_cfg % len(self._cfg_pool)]
        self._next_cfg += 1
        return local

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

    def _admitted_guests(self) -> int:
        """Number of anonymous visitor identities admitted by current step."""
        if not (self.ramp_guests and self.admission_horizon):
            return self.num_guests
        return min(self.num_guests,
                   int(self.step_count / self.admission_horizon * self.num_guests) + 1)

    def _compromise(self, machine: int) -> None:
        """Initialize compromise on machine: harvest cached or foreign credentials for pivot."""
        self.compromised_state[machine] = 1
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
        for d in (self.compromised_state, self.compromised_chain_remaining,
                  self.compromised_dwell, self.harvested_creds):
            d.pop(machine, None)

    def _benign_sensors(self) -> tuple[float, float, float]:
        """Generate baseline sensor probe values (s1, s2, s3) misfiring at p_sensor_fp."""
        if self.p_sensor_fp <= 0:
            return 0.0, 0.0, 0.0
        s = (np.random.rand(3) < self.p_sensor_fp).astype(float)
        return float(s[0]), float(s[1]), float(s[2])

    def _emit_theft_event(self, incident: dict) -> dict:
        """Emit one credential-theft request: attacker IP/device/config with victim credentials."""
        u = incident["victim"]
        role, clr = self.user_roles[u], self.user_clearances[u]
        # Destination drawn from valid actions for role to preserve destination marginal
        valid = self._policy_valid_actions(role)
        res_idx, method = (self._zipf_choice(valid, ("valid", role)) if valid else (0, 0))
        # Credential theft is policy-clean and signal-clean; exposed only by broken binding
        s1, s2, s3 = self._benign_sensors()
        feat = [1.0, s1, s2, s3, float(method),
                ROLES.index(role) / (len(ROLES) - 1), clr / 4.0]
        incident["remaining"] -= 1
        if incident["remaining"] <= 0:
            self._active_thefts.remove(incident)
        return self._event(
            source=incident["src_slot"], config=incident["cfg_slot"],
            device=incident["dev_slot"], user=u,
            res_idx=res_idx, feat=feat, label=1, etype=4, scenario=0,
        )

    def _event(self, *, source, config, device, user, res_idx, feat, label, etype, scenario):
        if device in self._slot_age:
            self._slot_age[device] += 1
        if user in self._user_age:
            if self._user_age[user] < _NEW_USER_COLD_EVENTS:
                scenario |= SCEN_NEW_USER
            self._user_age[user] += 1

        dst = self.res_lo + res_idx
        return {
            "source": source, "config": config, "device": device, "user": user, "dst": dst,
            "t": self.t, "features": feat, "label": label, "etype": etype,
            "scenario": scenario,
            "key_source": self.keys[source], "key_config": self.keys[config],
            "key_device": self.keys[device],
            "key_user": self.keys[user], "key_dst": self.keys[dst],
        }

    # --- One Event Step ---
    def _current_interarrival_scale(self) -> float:
        """Scale parameter for exponential inter-arrival time reflecting circadian activity."""
        day_sec = self.t % 86400
        weekday = (self.t // 86400) % 7
        is_weekend = weekday >= 5
        
        if is_weekend:
            return 1200.0  # Slow weekend traffic
            
        # Work hours: 08:00 to 18:00
        if 28800 <= day_sec <= 64800:
            return 45.0  # High work hour traffic
        else:
            return 600.0  # Slow night traffic

    def step(self) -> dict:
        self.t += max(1, int(np.random.exponential(scale=self._current_interarrival_scale())))
        self.step_count += 1

        # Interleave active credential theft requests to maintain natural user timing
        if self.admission_horizon:
            max_m = min(
                self.num_devices,
                int(self.step_count / self.admission_horizon * self.num_devices) + 1,
            )
        else:
            max_m = self.num_devices
        self._admitted = max_m
        self._maybe_release_config()
        self._maybe_onboard()

        if self._active_thefts and random.random() < self.p_theft_interleave:
            return self._emit_theft_event(random.choice(self._active_thefts))
        if (
            random.random() < self.p_cred_theft and self._src_pool and self._cfg_pool
            and (self._dev_pool or self._guest_dev_slot is not None)
        ):
            victim = self._registered[int(np.random.randint(0, len(self._registered)))]
            victim_machines = [
                m for m in range(self._admitted) if victim in self.machine_users[m]
                and self.machine_tiers[m] < 2  # TPM-bound identities cannot be stolen
            ]
            replay_m = None
            if victim_machines and random.random() < self.p_theft_session_replay:
                # Pass-the-cookie: replay victim device cookie
                replay_m = int(random.choice(victim_machines))
                dev_slot = self.machine_slot[replay_m]
            elif self.guest_device_fallback and self._guest_dev_slot is not None:
                dev_slot = self._guest_dev_slot
            else:
                cert = any(self.machine_tiers[m] == 1 for m in victim_machines)
                dev_slot = self._alloc_dev(
                    tier=1 if cert and random.random() < self.p_theft_mimic_config else 0
                )
                if dev_slot is None:
                    dev_slot = self.machine_slot[int(np.random.randint(0, max_m))]
            # Attacker network and client mimicry
            if random.random() < self.p_theft_known_source:
                src_slot = self.src_lo + int(np.random.randint(min(30, self.num_sources), self.num_sources))
            else:
                src_slot = self._alloc_src()
            if random.random() < self.p_theft_mimic_config:
                # Replay victim client config or sample common fleet client
                cfg_slot = (
                    self.cfg_lo + int(random.choice(self.machine_configs[replay_m]))
                    if replay_m is not None else self._fleet_config()
                )
            else:
                cfg_slot = self.cfg_lo + self._alloc_cfg()
            incident = {
                "victim": victim,
                "dev_slot": dev_slot,
                "src_slot": src_slot,
                "cfg_slot": cfg_slot,
                "remaining": int(np.random.randint(3, 7)),
            }
            self._active_thefts.append(incident)
            return self._emit_theft_event(incident)

        # Pick physical machine (progressively admitted), user and source IP
        if self.p_compromise is not None and random.random() < self.p_compromise:
            # Compromise clean admitted machine at global rate
            victim_m = int(np.random.randint(0, max_m))
            if victim_m not in self.compromised_state:
                self._compromise(victim_m)
        machine = int(np.random.randint(0, max_m))
        self._maybe_wipe_cookie(machine)
        dev_slot = self.machine_slot[machine]
        user = int(random.choice(self.machine_users[machine]))
        if (
            self.p_hotdesk > 0
            and random.random() < self.p_hotdesk
        ):
            # Hot-desking: registered user signing in on another machine
            user = int(random.choice(self._humans))
            self.machine_logons.setdefault(machine, set()).add(user)
        u_role, u_clearance = self.user_roles[user], self.user_clearances[user]

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

        # APT kill chain on compromised host (recon -> lateral -> dwell -> remediation)
        if (
            self.p_compromise is None  # Legacy hazard rate fallback
            and np.random.rand() < 0.005 and machine not in self.compromised_state
        ):
            self._compromise(machine)
        is_anomalous = (
            machine in self.compromised_state and np.random.rand() < 0.3
        )  # Compromised hosts blend in with benign traffic ~70% of the time

        is_anonymous = not is_anomalous and random.random() < 0.15

        if not is_anomalous:
            # Benign Service Account (cronjob) deterministic access pattern
            if random.random() < 0.05:
                user = 0
                u_role, u_clearance = self.user_roles[user], self.user_clearances[user]
                config = self.cfg_lo + 1
                if self.service_machines is not None:
                    sm = int(random.choice(self.service_machines))
                    machine, dev_slot = sm, self.machine_slot[sm]
                    source = self.src_lo + random.choice(sorted(self.machine_home_ips[sm]))
                    scenario = SCEN_SHARED if len(self.machine_users[sm]) > 1 else 0
                valid = self._policy_valid_actions(u_role)
                if valid:
                    res_idx, method = valid[0]
                    feat = [1.0, *self._benign_sensors(), float(method),
                            ROLES.index(u_role) / (len(ROLES) - 1), u_clearance / 4.0]
                    return self._event(source=source, config=config, device=dev_slot, user=user,
                                       res_idx=res_idx, feat=feat, label=0, etype=0, scenario=scenario)

            # Benign user mistake (OPA denial, etype=6)
            if not is_anonymous and random.random() < 0.02:
                invalid = self._policy_violations(u_role)
                if invalid:
                    res_idx, method = self._zipf_choice(invalid, ("viol", u_role))
                    feat = [1.0, *self._benign_sensors(), float(method),
                            ROLES.index(u_role) / (len(ROLES) - 1), u_clearance / 4.0]
                    return self._event(source=source, config=config, device=dev_slot, user=user,
                                       res_idx=res_idx, feat=feat, label=1, etype=6, scenario=scenario)

            if is_anonymous:
                user = self.num_registered_users + int(np.random.randint(0, self._admitted_guests()))
                u_role, u_clearance = "guest", 0
                config = self.cfg_lo  # conf:guest
                dev_slot = self._guest_dev_slot if self._guest_dev_slot is not None else dev_slot
                if not hasattr(self, "_anon_actions"):
                    self._anon_actions = [
                        (r, m) for r, uri in enumerate(self.resource_uris)
                        if uri not in self.security_matrix for m in self.route_methods[uri]
                    ]
                res_idx, method = self._zipf_choice(self._anon_actions, ("anon",))
            else:
                valid = self._policy_valid_actions(u_role)
                habit, non_habit = self._user_actions(user, u_role)
                # Benign exploration: authorized non-habitual access
                if non_habit and random.random() < self.benign_explore_prob:
                    res_idx, method = self._zipf_choice(non_habit, ("nonhabit", user))
                elif habit:
                    res_idx, method = self._zipf_choice(habit, ("habit", user))
                elif valid:
                    res_idx, method = self._zipf_choice(valid, ("valid", u_role))
                else:
                    res_idx, method = 0, 0  # public-path fallback

            ja3, (s1, s2, s3) = 1.0, self._benign_sensors()
            label, etype = 0, 0
        else:
            state = self.compromised_state[machine]
            multi = self.p_compromise is not None
            if state == 1:
                anomaly_type = "context"  # Recon phase (sensor probes)
                recon_left = self.compromised_chain_remaining.get(machine)
                if multi and recon_left is None:
                    recon_left = int(np.random.randint(1, 4))
                if not multi or recon_left <= 1:
                    self.compromised_state[machine] = 2
                    self.compromised_chain_remaining[machine] = int(np.random.randint(5, 12))
                else:
                    self.compromised_chain_remaining[machine] = recon_left - 1
            elif state == 2:
                anomaly_type = "lateral"  # Lateral movement phase
                self.compromised_chain_remaining[machine] -= 1
                if self.compromised_chain_remaining[machine] <= 0:
                    self.compromised_state[machine] = 4  # Post-exploitation dwell
                    if multi:
                        self.compromised_dwell[machine] = int(np.random.randint(0, 5))
            else:
                anomaly_type = np.random.choice(["policy", "context", "lateral"])
            if (
                self.compromised_state.get(machine) == 4 and self.p_compromise is not None
                and self.compromised_dwell.get(machine, 0) <= 0
            ):
                self._remediate(machine)
            elif state == 4 and self.p_compromise is not None:
                self.compromised_dwell[machine] -= 1

            if anomaly_type == "lateral":
                creds = [u for u in self.harvested_creds.get(machine, ()) if u != user]
                pivot = bool(creds) and random.random() < self.p_lateral_foreign_cred
                if pivot:
                    # Pivot using harvested credentials (cached or foreign identity)
                    user = int(random.choice(creds))
                    u_role, u_clearance = self.user_roles[user], self.user_clearances[user]
                    valid = self._policy_valid_actions(u_role)
                    target = (
                        self._zipf_choice(valid, ("valid", u_role)) if valid else None
                    )
                else:
                    # Own-identity lateral movement (authorized non-habitual access)
                    _habit, non_habit = self._user_actions(user, u_role)
                    target = (
                        self._zipf_choice(non_habit, ("nonhabit", user)) if non_habit else None
                    )
                if target is not None:
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

                    etype = 3
                    if not pivot and random.random() < self.p_lateral_role_spoof:
                        # Role claim disagreement (tested via p_lateral_role_spoof)
                        allowed_roles = [
                            r for r in ROLES
                            if r != u_role and self.policy_allows(r, method, self.resource_uris[res_idx])
                        ]
                        if allowed_roles:
                            u_role = random.choice(allowed_roles)
                            u_clearance = ROLE_CLEARANCE[u_role]
                    if random.random() < self.p_lateral_new_config:
                        # New tool execution on host
                        config = self._new_tool_config(machine)
                else:
                    anomaly_type = "policy"

            if anomaly_type == "policy":
                # Policy denial for this role (read-up, write-down, missing compartment)
                invalid = self._policy_violations(u_role)
                if not invalid:
                    anomaly_type = "context"
            if anomaly_type == "policy":
                res_idx, method = self._zipf_choice(invalid, ("viol", u_role))
                ja3, (s1, s2, s3) = 1.0, self._benign_sensors()
                etype = 1
            elif anomaly_type == "context":
                # Contextual anomaly (scanner probes / recon)
                valid = self._policy_valid_actions(u_role)
                if valid:
                    res_idx, method = self._zipf_choice(valid, ("valid", u_role))
                else:
                    res_idx, method = self._zipf_choice(self._all_actions, ("all",))
                ja3 = 0.0 if np.random.rand() > 0.5 else 1.0
                # Sensor probe trigger rates on external scan
                s1 = 1.0 if np.random.rand() > 0.2 else 0.0
                s2 = 1.0 if np.random.rand() > 0.5 else 0.0
                s3 = 1.0 if np.random.rand() > 0.8 else 0.0
                etype = 2

            label = 1

        feat = [ja3, float(s1), float(s2), float(s3), float(method),
                ROLES.index(u_role) / (len(ROLES) - 1), u_clearance / 4.0]
        return self._event(source=source, config=config, device=dev_slot, user=user,
                           res_idx=res_idx, feat=feat, label=label, etype=etype,
                           scenario=scenario)

    def add_resource(self, uri: str, methods: set[int], classification: str = None, categories: set[str] = None, risk: float = 0.5):
        """Dynamically add a new resource to the generator."""
        if uri in self.resource_uris:
            return
        
        self.route_methods[uri] = methods
        if classification and categories:
            self.security_matrix[uri] = (classification, categories)
        self.resource_risk[uri] = risk
        self.resource_uris.append(uri)
        
        self.keys.append(uri)
        
        new_feat = torch.zeros(1, 16)
        new_feat[0, 14] = 1.0  # trust score
        new_feat[0, 3] = 0.0  # slot [3] left at 0.0 to prevent label leakage (regression invariant)
        if self.use_resource_risk:
            new_feat[0, 4] = risk
            
        self.node_features = torch.cat([self.node_features, new_feat], dim=0)
        
        # Expand Zipf popularity weights for the newly added resource
        new_rank = float(len(self._res_pop_weight))
        new_weight = np.array([1.0 / ((new_rank + 1.0) ** 1.2)], dtype=np.float64)
        self._res_pop_weight = np.concatenate([self._res_pop_weight, new_weight])
        
        self.num_resources += 1
        self.num_nodes += 1
        
        # Invalidate caches so the new resource is picked up
        self._valid_cache.clear()
        self._violation_cache.clear()
        self._user_action_cache.clear()
        self._zipf_probs.clear()
        self._all_actions = [
            (r, m) for r, u in enumerate(self.resource_uris) for m in self.route_methods[u]
        ]
        if hasattr(self, "_anon_actions"):
            del self._anon_actions


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
    types: torch.Tensor         # [N] 0=benign, 1=policy, 2=contextual, 3=lateral,
                                #     4=cred-theft, 6=benign human error (denied); 5 reserved
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


def generate_streaming_data(
    num_users=1000,
    num_guests=1000,
    num_devices=2000,
    num_sources=1500,
    num_configs=400,
    num_resources=19,
    num_events=50000,
    *,
    num_wipe_slots=16,
    num_theft_slots=64,
    benign_explore_prob=0.15,
    p_roam=0.10,
    p_shared_device=0.20,
    p_cookie_wipe=0.0003,
    p_cred_theft=0.0012,
    seed=None,
    use_resource_risk=True,
    use_source_internal=False,
    guest_device_fallback=False,
    **realism,
) -> SyntheticStream:
    """Generate a reproducible offline synthetic stream using ZTAStreamSimulator."""
    sim = ZTAStreamSimulator(
        num_users=num_users, num_guests=num_guests, num_devices=num_devices, num_sources=num_sources,
        num_configs=num_configs, num_resources=num_resources, num_wipe_slots=num_wipe_slots,
        num_theft_slots=num_theft_slots, benign_explore_prob=benign_explore_prob,
        p_roam=p_roam, p_shared_device=p_shared_device, p_cookie_wipe=p_cookie_wipe,
        p_cred_theft=p_cred_theft, admission_horizon=num_events, seed=seed,
        use_resource_risk=use_resource_risk, use_source_internal=use_source_internal,
        guest_device_fallback=guest_device_fallback, **realism,
    )
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
    )
