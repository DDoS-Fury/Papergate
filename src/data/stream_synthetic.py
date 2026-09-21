"""Synthetic ZTA access stream — v4 schema: 5 node roles, 5 edges per request.

Every HTTP request involves five entities, mapped to a single node index space:

    [0, U)                    -> Users      (identity context: user id / credentials)
    [U, U+D)                  -> Devices    (hardware context: TPM id or device cookie)
    [U+D, U+D+S)              -> Sources    (network context: client IP)
    [U+D+S, U+D+S+C)          -> Configs    (client context: TLS/JA3 fingerprint)
    [U+D+S+C, U+D+S+C+R)      -> Resources  (data context: route URI)

and unrolls into the causal chain ``source_ip -> config -> device -> user -> resource``
(the configuration / JA3 fingerprint of the client is inserted between the network
source and the hardware device), **plus** a ``config -> user`` binding. The access edge
``user -> resource`` carries the request message; the four binding edges carry zero
messages.

The CONFIG node is the client's software identity (``conf:<ja3>``, or the generic
``conf:guest`` for an unfingerprinted client). A device habitually presents a small set
of configs, so a never-seen config on a known device (lateral movement with a new tool)
or for a known user (a credential thief whose client differs from the victim's) is an
extra structural tell — independent of the JA3 *validity* bit still carried in the edge
message (``features[0]``).

The split of the old single "IP/device" entity into SOURCE and DEVICE is what the
generator's dynamics exercise:

  * **roaming** (``p_roam``): a benign event from a non-home IP (smart working / 5G).
    Same device, same user — must NOT spike the anomaly score.
  * **shared devices** (``p_shared_device``): 2-3 users on one machine (control room).
    The device node bridges their histories; a compromised shared machine is visible
    on every user that touches it.
  * **cookie wipe** (``p_cookie_wipe``): a non-TPM machine loses its cookie and is
    re-keyed as a cold device node (benign; the cost of cookie-based identity).
  * **credential theft** (``p_cred_theft``, etype 4): an attacker issues requests as an
    existing victim user. Policy-clean and signal-clean — only the broken
    ``ip -> config -> device -> user`` binding pattern exposes it. Since v5 the attacker
    is mimetic: a common fleet client, a fleet egress address, or the victim's replayed
    session cookie (pass-the-cookie), each at its own rate.

Open world (v5). In v4 every benign entity was seen within the first few percent of the
stream while every attacker brought globally fresh IP / JA3 slots, so a set-membership
lookup ("never seen this IP") scored AUC 1.000 on credential theft with no learning. Now
novelty is a common BENIGN event too — never-seen roaming IPs, client releases that move
the fleet to new JA3s, hot-desking (a user on a machine that is not theirs), cookie
wipes, IDS false positives, unfingerprintable legacy clients — and benign churn and
attackers draw fresh nodes from ONE pool per role, with one key format, so neither the
slot index nor the key hash says who drew it. Lateral movement pivots with harvested
credentials (a new device->user binding, the Euler / LANL sense) instead of spoofing a
role claim; intrusions arrive at a global rate and are remediated after exfiltration, for
a ~1-2% attack prevalence. ``V4_KNOBS`` rebuilds the v4 process for before/after audits;
``graphagate.data.lookup_rules`` is the no-learning baseline the audit bounds.

Anomaly types (``types``): 0=benign, 1=policy violation (OPA-owned), 2=contextual,
3=lateral movement, 4=credential theft, 5=data exfiltration, 6=benign OPA denial (a
human mistake: ``label=1`` because OPA denies it, but not an attack). ``scenario`` is a
per-event bitmask of benign-context flags (an event can be several at once): 1=roaming,
2=recently wiped cookie device, 4=shared device.

Edge message layout (10-dim): ``[ja3, s1, s2, s3, method, role, clearance, bytes_in,
bytes_out, log1p(user Δt)/10]`` — TLS-fingerprint trust, the three Snort/sensor probes,
the HTTP method code, the requesting identity's role/clearance (possibly stolen), the
request/response volumes and the recency of this user's previous request.

Everything in this message is available to the PDP *before* the request is forwarded.
No **response** field may enter the message (e.g. the HTTP status): using it to decide
whether to allow the request is a causality violation, and a response field that takes a
class-specific constant would be a near-deterministic label channel.

Invariants this generator must preserve, regression-tested in
``tests/test_leakage_audit.py`` (which also bounds single history lookups and checks
that the role claim always matches the identity):

  * **No shortcut feature.** No single input column may separate an anomaly class on its
    own (bar the sensor probes on contextual and the resource RISK, which are legitimate
    signals by design). In particular benign and attack traffic share the same
    destination marginal, so "unusual destination" is never a free label.
  * **No exact-value fingerprint.** Per-class constants (byte volumes, status codes) let
    a model memorise the class instead of learning behaviour.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch

from graphagate.netclass import GUEST_DEVICE, ip_is_internal


def stream_kwargs_from_cfg(cfg) -> dict:
    """Generator kwargs that make a derived stream EXACTLY match the TGN stream.

    Single source of truth for the (TGNConfig -> generate_streaming_data) mapping:
    every consumer that must see the same entity space as the trained model (the
    baselines, the live test generator, the leakage audit) calls this instead of
    hand-copying the parameters. Hand-copied lists silently drifted before — the
    baselines ran 400 configs / per-cookie device keying against a model trained on
    40 configs / dev:guest.
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
        p_lateral_role_spoof=cfg.p_lateral_role_spoof,
        p_lateral_new_config=cfg.p_lateral_new_config,
        p_sensor_fp=cfg.p_sensor_fp,
        p_legacy_client=cfg.p_legacy_client,
        num_service_machines=cfg.num_service_machines,
        tier_mix=cfg.tier_mix,
        p_theft_interleave=cfg.p_theft_interleave,
    )

# The v4 generator (the one behind the pre-v5 paper numbers), expressed as knob values:
# closed-world benign traffic, globally fresh attacker slots, role-spoof lateral, no
# remediation. ``dataclasses.replace(TGNConfig(), **V4_KNOBS)`` rebuilds that process
# (not bit-identical: slot keys and RNG consumption changed) for before/after audits.
V4_KNOBS = dict(
    guest_device_fallback=True, num_wipe_slots=16, p_cookie_wipe=0.0003,
    num_new_sources=0, num_new_configs=0, p_new_source=0.0, p_config_release=0.0,
    p_hotdesk=0.0, p_sensor_fp=0.0, p_legacy_client=0.0,
    p_theft_mimic_config=0.0, p_theft_known_source=0.0, p_theft_session_replay=0.0,
    p_compromise=None, p_lateral_foreign_cred=0.0, p_lateral_role_spoof=0.5,
    p_lateral_new_config=0.5, num_service_machines=None, tier_mix=(0.2, 0.5, 0.3),
    p_theft_interleave=0.15,
)

ROLES = ["guest", "operator", "manager", "admin"]
CLEARANCES = ["PUBLIC", "INTERNAL", "CONFIDENTIAL", "SECRET", "TOP_SECRET"]
WRITE_METHODS = {1, 2, 3, 4}  # POST/PUT/DELETE/PATCH (0=GET is the only read)

# --- Reference authorization model (Bell-LaPadula + compartments) --------------------
# This module implements the REFERENCE policy model the generator labels against:
# no read-up on GET, no write-down on writes, compartment subset, plus the
# trusted-guard sanitized write-down exception. A deployment that uses the model
# in production must run an OPA policy implementing this same model, so that the
# benign manifold equals the set of accesses OPA permits and the `etype=1`
# violations are genuine OPA denials.
#
# Divergence warning: `docs/policies.txt` is a snapshot of the deployed ZTALeaks
# rego that implements a DIFFERENT model (per-route min_clearance for reads AND
# writes, min_tier device gate, a different role vocabulary, an AI-score deny
# override, and no trusted-guard route). It also requires authentication on
# /api/v1/auth/register/begin, while the reference model here keeps it public.
# The two must be reconciled before the labels produced by this generator are
# treated as decisions of the deployed PDP.
SECURITY_LEVELS = {  # rego `livelli`
    "PUBLIC": 0, "INTERNAL": 1, "CONFIDENTIAL": 2, "SECRET": 3, "TOP_SECRET": 4,
}
ROLE_CLEARANCE = {  # rego `ruoli_to_blp[*].clearance` — clearance DERIVES from the role
    "guest": 0, "operator": 1, "manager": 2, "admin": 4,
}
ROLE_CATEGORIES = {  # rego `ruoli_to_blp[*].categorie` — compartments granted to the role
    "guest": set(),
    "operator": {"hr", "ops"},
    "manager": {"hr", "ops", "finance"},
    "admin": {"hr", "ops", "finance", "nuclear", "security"},
}

TRUSTED_GUARD = "/api/v1/trusted-guard/sanitized-delete-personnel"

# rego `matrice_sicurezza`: protected route -> (classification, required categories).
SECURITY_MATRIX = {
    "/api/v1/personnel":          ("INTERNAL",     {"hr"}),
    "/api/v1/documents":          ("CONFIDENTIAL", {"finance"}),
    "/api/v1/nuclear-materials":  ("TOP_SECRET",   {"nuclear"}),
    "/api/v1/reactor-parameters": ("TOP_SECRET",   {"nuclear", "security"}),
    TRUSTED_GUARD:                ("SECRET",       {"security"}),
}

# Methods each route serves (the candidate ``(route, method)`` action space).
# methods: 0=GET, 1=POST, 2=PUT, 3=DELETE, 4=PATCH. Protected routes expose exactly the
# methods declared in the security matrix; public/auth routes are GET or POST. Public
# routes (everything outside the security matrix) are allowed for every role — OPA
# gates them on ai_score only. NB: /api/v1/auth/register/{begin,finish} are PUBLIC in
# this reference model; the deployed snapshot in docs/policies.txt instead requires
# authentication on register/begin (see the divergence warning above).
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

# The hand-written routes above are the *real* orchestrator surface. On top of them the
# simulator generates a larger synthetic estate so the resource space is not degenerate.
# That generation is **seed-dependent** (see ``build_resource_universe``): a resource
# universe fixed across seeds would make the across-seed dispersion understate the true
# variability, since every run would share the exact same route catalogue.
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

# Inherent RISK per classification (node_feat index 4). Collapsed to a per-resource
# scalar (the max over methods, since the resource node is method-agnostic). Routes
# the orchestrator never scores as sensitive (public/auth/static) are 0.0.
_CLASSIFICATION_RISK = {
    "INTERNAL": 0.5, "CONFIDENTIAL": 0.6, "SECRET": 0.8, "TOP_SECRET": 0.9,
}
# Hand-tuned overrides for the real routes (keep aligned if the Go map changes).
_RISK_OVERRIDES = {
    "/api/v1/personnel": 0.6,                # max(GET/POST 0.4, DELETE 0.6)
    "/api/v1/documents": 0.7,                # max(GET/POST 0.5, DELETE 0.7)
    "/api/v1/nuclear-materials": 0.7,        # max(0.5, DELETE 0.7)
    "/api/v1/reactor-parameters": 1.0,       # max(GET/POST 0.8, DELETE 1.0)
    TRUSTED_GUARD: 1.0,                      # sanitized delete = highest
}


def build_resource_universe(num_generated: int = 981, seed: int = 42):
    """Build the resource catalogue: the real routes plus ``num_generated`` synthetic ones.

    Returns ``(route_methods, security_matrix, resource_uris, resource_risk)``. The
    synthetic estate is drawn under ``random.Random(seed)`` so that each simulator seed
    gets its own catalogue — the resource universe is part of what a seed varies, not a
    constant shared by every run.

    Resource node keys MUST be the exact URIs the orchestrator sends as ``key_dst``
    (after its normalizeAIPath): no synthetic suffixes, one node per real route.
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


# Module-level defaults (seed 42) — the reference catalogue used by ``policy_allows`` and
# by the policy/netclass unit tests. Simulator instances build their own under their seed.
ROUTE_METHODS, SECURITY_MATRIX, RESOURCE_URIS, RESOURCE_RISK = build_resource_universe()


def policy_allows(role: str, method: int, uri: str) -> bool:
    """Reference-model allow decision for ``(role, method, route)``
    (Bell-LaPadula + compartments + trusted-guard exception).

    Public routes (anything outside :data:`SECURITY_MATRIX`) are allowed for every role;
    OPA gates them on ``ai_score`` only, which is orthogonal to identity. For protected
    routes we enforce, in order: the method must be one the route serves, the route's
    compartments must be a subset of the role's categories, and Bell-LaPadula — *no
    read-up* on GET (``clearance >= classification``) and *no write-down* on writes
    (``clearance <= classification``), with the sanitized write-down exception that lets
    ``admin`` POST the SECRET trusted-guard route.
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

# ``scenario`` bitmask flags (benign-context annotations for scenario-level eval).
SCEN_ROAMING = 1   # benign event issued from a non-home IP (smart working / 5G)
SCEN_WIPED = 2     # device cookie recently wiped: the device node is still cold
SCEN_SHARED = 4    # the device is shared by multiple users

# A wiped device node is considered "cold" for this many events on the new slot.
_WIPE_COLD_EVENTS = 25

# Per-event chance a benign request comes from an unfingerprinted client (``conf:guest``)
# rather than one of the machine's habitual configs (e.g. a fresh browser profile / a
# client whose JA3 the collector could not resolve). A low-information default, like a
# shared NAT IP — must not by itself look anomalous.
_P_GUEST_CONFIG = 0.05


class ZTAStreamSimulator:
    """Stateful per-event simulator behind both the offline tensor stream
    (:func:`generate_streaming_data`) and the live API generator (``tests/generator.py``).

    ``admission_horizon`` ramps device admission over that many steps (mirrors the
    progressive entity admission of the original generator); ``None`` admits the whole
    fleet immediately (live/serving continuation).
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
        p_lateral_role_spoof: float = 0.5,
        p_lateral_new_config: float = 0.5,
        p_sensor_fp: float = 0.0,
        p_legacy_client: float = 0.0,
        num_service_machines: int | None = None,
        tier_mix: tuple[float, float, float] = (0.2, 0.5, 0.3),
        p_theft_interleave: float = 0.15,
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
        self.p_lateral_role_spoof = p_lateral_role_spoof
        self.p_lateral_new_config = p_lateral_new_config
        self.p_sensor_fp = p_sensor_fp

        # --- resource popularity, DECOUPLED from the resource index ---
        # Access frequency follows a Zipf law, but the rank a resource gets is a random
        # permutation of the index space rather than the index itself. Otherwise the
        # resource id would encode popularity, and since benign traffic concentrates on
        # popular resources while attacks do not, the id alone would separate the classes
        # — a pure dataset artifact. See tests/test_leakage_audit.py.
        pop_rank = np.random.permutation(num_resources).astype(np.float64)
        self._res_pop_weight = 1.0 / ((pop_rank + 1.0) ** 1.2)

        # --- node index layout: [users][device slots][source slots][config slots][resources] ---
        self.user_lo = 0
        self.dev_lo = self.num_users
        self.dev_slots = num_devices + num_wipe_slots + num_theft_slots
        self.src_lo = self.dev_lo + self.dev_slots
        # Trailing source / config slots form ONE fresh pool per role, shared by benign
        # churn (never-seen roaming IPs, JA3 releases) and attackers. A single allocator
        # hands them out in arrival order with the same key format, so neither the slot
        # index nor the key string (hashed into the model input) reveals who drew it.
        self.src_slots = num_sources + num_theft_slots + num_new_sources
        self.cfg_lo = self.src_lo + self.src_slots
        # Config slot 0 is the generic ``conf:guest``; [1, num_configs) are habitual
        # fingerprints; the trailing fresh pool holds never-seen fingerprints.
        self.cfg_slots = num_configs + num_theft_slots + num_new_configs
        self.res_lo = self.cfg_lo + self.cfg_slots
        self.num_nodes = self.res_lo + num_resources

        # --- users ---
        # Clearance is NOT independent: like policy.rego it derives from the role
        # (ruoli_to_blp), so the role/clearance pair in every benign message is exactly
        # what the JWT would carry in production.
        self.user_roles = [str(np.random.choice(ROLES)) for _ in range(self.num_registered_users)]
        self.user_roles.extend(["guest"] * self.num_guests)
        self.user_clearances = [ROLE_CLEARANCE[r] for r in self.user_roles]

        # --- physical machines (stable across cookie wipes) ---
        # Tier: 0=no cert/tpm, 1=cert, 2=cert+tpm. tier-2 machines are TPM-keyed;
        # the rest are cookie-keyed (re-keyed on wipe).
        self.machine_tiers = [int(np.random.choice([0, 1, 2], p=tier_mix))
                              for _ in range(num_devices)]
        # Owner(s): base round-robin owner + extra users on shared machines.
        # With dedicated service machines, user 0 is a pure service identity: it owns no
        # desk and never hot-desks (v4 made it a human desk owner AND the fleet cronjob).
        self._humans = (
            list(range(1, self.num_registered_users))
            if num_service_machines is not None and self.num_registered_users > 1
            else list(range(self.num_registered_users))
        )
        self.machine_users: list[list[int]] = []
        for m in range(num_devices):
            users = [self._humans[m % len(self._humans)]]
            if np.random.rand() < p_shared_device:
                extra = np.random.randint(1, 4)
                pool = [u for u in self._humans if u not in users]
                users += list(np.random.choice(pool, size=min(extra, len(pool)), replace=False))
            self.machine_users.append(users)

        # Home IPs: one shared office IP (NAT: many machines behind it) plus a
        # dedicated home/dsl IP. Roaming draws any other IP from the pool (5G/estero).
        num_office = min(30, num_sources)
        self._office_locals = list(range(num_office))
        self.machine_home_ips: list[set[int]] = []
        for m in range(num_devices):
            home = {int(np.random.choice(self._office_locals))}
            if num_sources > num_office:
                home.add(int(np.random.randint(num_office, num_sources)))
            self.machine_home_ips.append(home)

        # --- habitual client configs (TLS/JA3) per machine ---
        # Config 0 is the generic ``conf:guest``; the habitual pool is [1, num_configs).
        # Each machine runs 1-2 of those (browsers/tools share fingerprints across
        # machines, so the pool is small and overlapping). A config outside a machine's
        # habitual set is a new tool on that device (lateral tell).
        cfg_pool = list(range(1, num_configs)) if num_configs > 1 else [0]
        self.machine_configs: list[list[int]] = []
        for m in range(num_devices):
            k = min(int(np.random.randint(1, 3)), len(cfg_pool))
            cfgs = np.random.choice(cfg_pool, size=k, replace=False)
            self.machine_configs.append([int(c) for c in cfgs])

        # Legacy clients: machines whose TLS stack the collector cannot fingerprint, so
        # their BENIGN traffic carries ja3=0 too. Without them "ja3 invalid" is a
        # zero-false-positive recon tell.
        self.machine_legacy = (
            [bool(np.random.rand() < p_legacy_client) for _ in range(num_devices)]
            if p_legacy_client > 0 else [False] * num_devices
        )
        # Service account (user 0): a cronjob runs on a few dedicated server machines,
        # not on the whole fleet (``None`` = v4 behaviour, any machine).
        self.service_machines = (
            None if num_service_machines is None
            else list(range(min(num_service_machines, num_devices)))
        )

        # --- external keys per node slot ---
        self.keys: list[str | None] = [None] * self.num_nodes
        for u in range(self.num_registered_users):
            self.keys[self.user_lo + u] = f"user_{u:04d}"
        for g in range(self.num_guests):
            self.keys[self.user_lo + self.num_registered_users + g] = f"guest_{g:04d}"
        # Cookie keys are opaque random tokens (``ck:<hex>``), as a browser cookie is: a
        # key encoding the machine or the allocation reason (``ck:atk-…``) would be a class
        # tell through the key hash the model consumes.
        self._used_cookies: set[str] = set()
        for m in range(num_devices):
            tier = self.machine_tiers[m]
            self.keys[self.dev_lo + m] = f"tpm:{m:04d}" if tier == 2 else self._new_cookie()
        for k in range(num_wipe_slots + num_theft_slots):
            self.keys[self.dev_lo + num_devices + k] = f"_spare_dev_{k}"
        # Source keys are namespaced ``src:<ip>`` so a client IP can never alias onto a
        # device slot in the shared NodeRegistry (the orchestrator sends the same prefix;
        # the legacy IP-fallback device uses a distinct ``ipdev:`` prefix). Office IPs are
        # RFC1918 (internal); CGNAT 100.64/10 (roaming) and TEST-NET 203.0.113 are external.
        for s in range(num_sources):
            self.keys[self.src_lo + s] = (
                f"src:10.0.0.{s}" if s < num_office else f"src:100.64.{s // 256}.{s % 256}"
            )
        # Fresh-pool sources continue the external address space: slot ``num_sources + j``
        # is the j-th never-seen IP, whoever uses it first.
        for k in range(num_theft_slots + num_new_sources):
            self.keys[self.src_lo + num_sources + k] = self._fresh_ip_key(num_sources + k)
        # Config keys: slot 0 = generic guest, [1, num_configs) = habitual JA3
        # fingerprints, trailing slots = fresh fingerprints (JA3 releases and attacker
        # tools), numbered in allocation order.
        self.keys[self.cfg_lo] = "conf:guest"
        for c in range(1, num_configs):
            self.keys[self.cfg_lo + c] = f"conf:{c:04d}"
        for k in range(num_theft_slots + num_new_configs):
            self.keys[self.cfg_lo + num_configs + k] = f"conf:{num_configs + k:04d}"
        for r in range(num_resources):
            self.keys[self.res_lo + r] = self.resource_uris[r]

        # --- static node features (16-dim) ---
        # Index map: [2]=device tier, [3]=UNUSED, [4]=resource RISK,
        # [5]=source network internal(1)/external(0), [14]=trust score.
        #
        # [3] is deliberately left at 0. A raw resource index here (dressed up as a
        # "priority") would be redundant with the RISK in [4] and, because the index
        # correlates with access frequency, would leak the label: that single column
        # reaches AUC ~0.92 on every anomaly class.
        #
        # [4] (RISK) is kept: it is a genuine ZTA attribute, known at decision time, and
        # its correlation with policy violations is semantic rather than an artifact. Its
        # standalone discriminative power is reported as a floor (tests/test_leakage_audit).
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

        # --- behaviour model ---
        # Valid actions are OPA's allow set for the user's role (device tier is realism
        # only, never a policy gate — OPA does not see it). The habitual subset is per
        # USER (the access edge is user -> resource).
        self._valid_cache: dict[str, list[tuple[int, int]]] = {}
        self._violation_cache: dict[str, list[tuple[int, int]]] = {}
        self._sensitive_cache: dict[str, list[tuple[int, int]]] = {}
        self._user_action_cache: dict[int, tuple[list, list]] = {}
        # Zipf probability vectors, memoised per action-set cache key (see _zipf_choice).
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

        # Every ``(resource, method)`` action in the catalogue. Sampling contextual
        # anomalies from this list (instead of drawing a method uniformly in 0..3) keeps
        # them inside the served (route, method) space: an unserved pair is a region
        # benign traffic never occupies and would be a free label.
        self._all_actions = [
            (r, m) for r, uri in enumerate(self.resource_uris) for m in self.route_methods[uri]
        ]

        # --- mutable state ---
        self.t = start_time
        self.step_count = 0
        self.machine_slot = {m: self.dev_lo + m for m in range(num_devices)}
        # Guest device fallback (experiment): every non-TPM machine collapses onto a
        # single shared guest device node, the mirror of ``conf:guest``. The first
        # non-TPM machine's slot becomes the canonical guest node; the rest reuse it via
        # ``machine_slot``. Their now-inert own slots keep a unique placeholder key so
        # the registry's slot<->key identity mapping (preregister) stays intact.
        self._guest_dev_slot: int | None = None
        if self.guest_device_fallback:
            non_tpm = [m for m in range(num_devices) if self.machine_tiers[m] < 2]
            if non_tpm:
                guest = self.dev_lo + non_tpm[0]
                self._guest_dev_slot = guest
                self.keys[guest] = GUEST_DEVICE
                self.node_features[guest, 2] = 0.0  # anonymous device: lowest tier
                for m in non_tpm:
                    self.machine_slot[m] = guest
                    if self.dev_lo + m != guest:
                        self.keys[self.dev_lo + m] = f"_guest_unused_dev_{m:04d}"
                        self.node_features[self.dev_lo + m, 2] = 0.0
        # Fresh-slot allocators (round-robin, recycled when exhausted: a recycled slot is
        # a node the graph has already seen, which only weakens the novelty tell).
        self._dev_pool = [self.dev_lo + num_devices + k for k in range(num_wipe_slots + num_theft_slots)]
        self._src_pool = [self.src_lo + num_sources + k for k in range(num_theft_slots + num_new_sources)]
        self._cfg_pool = [num_configs + k for k in range(num_theft_slots + num_new_configs)]  # local ids
        self._next_dev = self._next_src = self._next_cfg = 0
        self._slot_age: dict[int, int] = {}      # events seen by a re-keyed (wiped) slot
        self.last_user_t: dict[int, int] = {}
        self.compromised_state: dict[int, int] = {}  # machine -> kill-chain phase
        self.compromised_chain_remaining: dict[int, int] = {} # machine -> steps left in lateral chain
        self.compromised_dwell: dict[int, int] = {}  # machine -> post-exfil events before remediation
        self.harvested_creds: dict[int, list[int]] = {}  # machine -> users whose creds were dumped
        self._active_thefts: list[dict] = []
        # JA3 release model: habitual config (local id) -> its newer version. A machine
        # still running the old one upgrades at its next use with prob p_config_adopt.
        self._cfg_upgrade: dict[int, int] = {}
        self._admitted = num_devices  # machines admitted so far (updated each step)

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
        """``(resource_idx, method)`` actions on PROTECTED routes that OPA would DENY for
        this role — genuine policy.rego violations (read-up / write-down / missing
        compartment). Public routes can never be a policy violation."""
        if role not in self._violation_cache:
            self._violation_cache[role] = [
                (r, m)
                for r, uri in enumerate(self.resource_uris)
                for m in self.route_methods[uri]
                if uri in self.security_matrix and not self.policy_allows(role, m, uri)
            ]
        return self._violation_cache[role]

    def _sensitive_actions(self, role: str):
        """Allowed actions on PROTECTED routes for this role — the "loot" set (cached)."""
        if role not in self._sensitive_cache:
            self._sensitive_cache[role] = [
                (r, m) for r, m in self._policy_valid_actions(role)
                if self.resource_uris[r] in self.security_matrix
            ]
        return self._sensitive_cache[role]

    def _user_actions(self, user: int, role: str):
        """``(habitual, non_habitual)`` allowed actions for this user (cached)."""
        if user not in self._user_action_cache:
            valid = self._policy_valid_actions(role)
            hab = self.user_habitual[user]
            self._user_action_cache[user] = (
                [a for a in valid if a in hab],
                [a for a in valid if a not in hab],
            )
        return self._user_action_cache[user]

    def _zipf_choice(self, choices: list, key: tuple):
        """Draw an action with popularity-weighted (Zipf) probability.

        The weight comes from the resource's *popularity rank* — a random permutation of
        the index space fixed at construction — never from its position in ``choices``.
        ``key`` identifies the (cached, stable) action list so the probability vector is
        computed once per list rather than per event.

        Every destination draw in the generator, benign or anomalous, goes through here:
        benign and attack traffic must share the same destination marginal, otherwise
        "unusual destination" becomes a free label and the lateral-movement task is
        solvable without ever looking at the graph.
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

    # --- fresh-slot allocation (shared by benign churn and attackers) ---
    def _new_cookie(self) -> str:
        while True:
            key = f"ck:{random.getrandbits(48):012x}"
            if key not in self._used_cookies:
                self._used_cookies.add(key)
                return key

    @staticmethod
    def _fresh_ip_key(i: int) -> str:
        """Key of source slot ``i`` — the CGNAT/external range the roaming pool uses."""
        return f"src:100.{64 + i // 65536}.{(i // 256) % 256}.{i % 256}"

    def _alloc_dev(self, tier: int) -> int | None:
        """A fresh device slot with a new opaque cookie. Slots held by a machine or by
        an active theft incident are skipped when the pool wraps around; ``None`` if every
        slot is held (never re-key a live device node)."""
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
        """Global slot of a never-seen client IP (recycled round-robin when exhausted)."""
        slot = self._src_pool[self._next_src % len(self._src_pool)]
        self._next_src += 1
        return slot

    def _alloc_cfg(self) -> int:
        """LOCAL id of a never-seen JA3 fingerprint (recycled round-robin when exhausted)."""
        local = self._cfg_pool[self._next_cfg % len(self._cfg_pool)]
        self._next_cfg += 1
        return local

    def _maybe_wipe_cookie(self, machine: int) -> None:
        """Re-key a cookie-identified machine onto a fresh (cold) device slot."""
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
        """A client release (browser/TLS-library update) changes the JA3 of one habitual
        fingerprint: the fleet migrates to a globally never-seen config over time. This
        is what makes "new config on a known device / for a known user" a common BENIGN
        event rather than an attack-only one."""
        if self.p_config_release <= 0 or random.random() >= self.p_config_release:
            return
        in_use = sorted({c for cfgs in self.machine_configs for c in cfgs} - set(self._cfg_upgrade))
        if in_use:
            self._cfg_upgrade[int(random.choice(in_use))] = self._alloc_cfg()

    def _habitual_config(self, machine: int) -> int:
        """Global slot of a benign client config for ``machine`` (its habitual JA3, or
        occasionally the generic ``conf:guest``). A pending release is adopted here."""
        if random.random() < _P_GUEST_CONFIG:
            return self.cfg_lo  # conf:guest
        cfgs = self.machine_configs[machine]
        j = random.randrange(len(cfgs))
        new = self._cfg_upgrade.get(cfgs[j])
        if new is not None and random.random() < self.p_config_adopt:
            cfgs[j] = new
        return self.cfg_lo + int(cfgs[j])

    def _fleet_config(self) -> int:
        """Global slot of a config drawn with its fleet popularity (a common client)."""
        m = int(np.random.randint(0, self._admitted))
        return self.cfg_lo + int(random.choice(self.machine_configs[m]))

    def _new_tool_config(self, machine: int) -> int:
        """Global slot of a config ``machine`` has never used (a new tool on a known
        device — the lateral-movement config tell). Falls back to the habitual config
        when the pool offers no alternative."""
        habit = set(self.machine_configs[machine])
        others = [c for c in range(1, self.num_configs) if c not in habit]
        if not others:
            return self._habitual_config(machine)
        return self.cfg_lo + int(random.choice(others))

    def _compromise(self, machine: int) -> None:
        """Start a kill chain on ``machine``: the intruder dumps 1-3 credentials of users
        who do not own it (cached logons, a keylogger) for the lateral pivot."""
        self.compromised_state[machine] = 1
        pool = [u for u in range(self.num_registered_users) if u not in self.machine_users[machine]]
        if pool:
            k = min(int(np.random.randint(1, 4)), len(pool))
            self.harvested_creds[machine] = [int(u) for u in np.random.choice(pool, size=k, replace=False)]

    def _remediate(self, machine: int) -> None:
        for d in (self.compromised_state, self.compromised_chain_remaining,
                  self.compromised_dwell, self.harvested_creds):
            d.pop(machine, None)

    def _benign_signals(self, machine: int) -> tuple[float, float, float, float]:
        """``(ja3, s1, s2, s3)`` of a non-recon request from ``machine``: a legacy client
        is never fingerprinted, and each IDS probe misfires at ``p_sensor_fp``."""
        ja3 = 0.0 if self.machine_legacy[machine] else 1.0
        if self.p_sensor_fp <= 0:
            return ja3, 0.0, 0.0, 0.0
        s = (np.random.rand(3) < self.p_sensor_fp).astype(float)
        return ja3, float(s[0]), float(s[1]), float(s[2])

    def _emit_theft_event(self, incident: dict) -> dict:
        """One credential-theft request: attacker IP + config + device, victim identity."""
        u = incident["victim"]
        role, clr = self.user_roles[u], self.user_clearances[u]
        # The attacker holds the victim's credentials: OPA's role/clearance/category
        # checks all pass. Destination is drawn from valid actions for the role so attack
        # and benign traffic share the same destination marginal (no risk shortcut).
        valid = self._policy_valid_actions(role)
        res_idx, method = (self._zipf_choice(valid, ("valid", role)) if valid else (0, 0))
        # Byte volumes are drawn from the SAME laws as benign traffic. Credential theft is
        # policy-clean and signal-clean by construction — only the broken
        # ip -> config -> device -> user binding exposes it. Constant byte values here used
        # to identify the class with 100% precision and recall, i.e. pure label leakage.
        _ja3, s1, s2, s3 = self._benign_signals(0)
        feat = [1.0, s1, s2, s3, float(method),
                ROLES.index(role) / (len(ROLES) - 1), clr / 4.0,
                float(abs(np.random.normal(0.1, 0.05))),
                float(abs(np.random.normal(0.2, 0.1)))]
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
            
        delta_t = self.t - self.last_user_t.get(user, self.t)
        self.last_user_t[user] = self.t
        feat.append(float(np.log1p(delta_t) / 10.0))
        
        dst = self.res_lo + res_idx
        return {
            "source": source, "config": config, "device": device, "user": user, "dst": dst,
            "t": self.t, "features": feat, "label": label, "etype": etype,
            "scenario": scenario,
            "key_source": self.keys[source], "key_config": self.keys[config],
            "key_device": self.keys[device],
            "key_user": self.keys[user], "key_dst": self.keys[dst],
        }

    # --- one event ---
    def _current_interarrival_scale(self) -> float:
        """Returns a time-dependent scale for the exponential inter-arrival distribution
        to simulate circadian rhythms (higher rate during work hours)."""
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

        # Credential-theft incidents: requests from a never-seen attacker IP + device as
        # an existing victim user (signal-clean, policy-clean). They INTERLEAVE with normal
        # traffic rather than arriving back-to-back: a high emission rate made the victim's
        # inter-request gap collapse, and that gap is an edge feature — the class became
        # identifiable from ``log1p(Δt)`` alone, with no need for the binding structure that
        # is supposed to be the only thing exposing it.
        if self.admission_horizon:
            max_m = min(
                self.num_devices,
                int(self.step_count / self.admission_horizon * self.num_devices) + 1,
            )
        else:
            max_m = self.num_devices
        self._admitted = max_m
        self._maybe_release_config()

        if self._active_thefts and random.random() < self.p_theft_interleave:
            return self._emit_theft_event(random.choice(self._active_thefts))
        if (
            random.random() < self.p_cred_theft and self._src_pool and self._cfg_pool
            and (self._dev_pool or self._guest_dev_slot is not None)
        ):
            victim = int(np.random.randint(0, self.num_registered_users))
            victim_machines = [
                m for m in range(self._admitted) if victim in self.machine_users[m]
                and self.machine_tiers[m] < 2  # a TPM-bound identity cannot be replayed
            ]
            replay_m = None
            if victim_machines and random.random() < self.p_theft_session_replay:
                # Session hijack (infostealer -> pass-the-cookie): the attacker replays
                # the victim's device cookie, so device and device->user bindings are the
                # victim's own; only the source/config around them are foreign.
                replay_m = int(random.choice(victim_machines))
                dev_slot = self.machine_slot[replay_m]
            elif self.guest_device_fallback and self._guest_dev_slot is not None:
                # Attacker device is TPM-less too, so under the guest-fallback policy it
                # collapses onto the shared guest node (the device-identity tell is lost).
                dev_slot = self._guest_dev_slot
            else:
                # A fresh cookie, like any new browser (a busy pool falls back to replay
                # of an arbitrary fleet device rather than re-keying a live one).
                # Infostealer kits export software (non-TPM) device certificates with
                # the cookies, so the attacker can present the victim's tier-1 posture;
                # only a TPM-bound identity is out of reach (``victim_machines`` above).
                cert = any(self.machine_tiers[m] == 1 for m in victim_machines)
                dev_slot = self._alloc_dev(
                    tier=1 if cert and random.random() < self.p_theft_mimic_config else 0
                )
                if dev_slot is None:
                    dev_slot = self.machine_slot[int(np.random.randint(0, max_m))]
            # Mimicry: a real attacker runs a common client (stock Chrome, curl) and
            # egresses through residential / mobile address space the fleet also uses.
            # Only the remainder brings a never-seen JA3 / IP — drawn from the SAME fresh
            # pools that benign releases and roaming draw from.
            if random.random() < self.p_theft_known_source:
                src_slot = self.src_lo + int(np.random.randint(min(30, self.num_sources), self.num_sources))
            else:
                src_slot = self._alloc_src()
            if random.random() < self.p_theft_mimic_config:
                # A replayed session comes with the victim's fingerprint (anti-detect
                # browser kits sell both together); otherwise a popular fleet client.
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

        # --- pick the physical machine (progressively admitted), its user and IP ---
        if self.p_compromise is not None and random.random() < self.p_compromise:
            # A new intrusion lands on a random admitted, currently clean machine. Unlike
            # the v4 per-visit hazard, the global rate keeps prevalence independent of
            # fleet size, and remediation (below) returns machines to the clean pool.
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
            # Hot-desking: a registered user signs in on a machine that is not theirs
            # (meeting room, colleague's desk) — a BENIGN new device->user binding.
            user = int(random.choice(self._humans))
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
                # Mobile / CGNAT / hotel Wi-Fi: an address nobody in the fleet used before.
                source = self._alloc_src()
            else:
                src_local = int(np.random.randint(0, self.num_sources))
                while src_local in home:
                    src_local = int(np.random.randint(0, self.num_sources))
                source = self.src_lo + src_local
        else:
            source = self.src_lo + random.choice(sorted(home))

        # Client config (TLS/JA3): the machine's habitual fingerprint by default. It is a
        # software identity, so roaming (a network change) does NOT change it; lateral
        # movement may swap in a new tool (see below).
        config = self._habitual_config(machine)

        # --- APT kill chain on the physical machine (recon -> lateral -> exfil) ---
        if (
            self.p_compromise is None  # v4 hazard: per visit, never remediated
            and np.random.rand() < 0.005 and machine not in self.compromised_state
        ):
            self._compromise(machine)
        is_anomalous = (
            machine in self.compromised_state and np.random.rand() < 0.3
        )  # compromised machines blend in 70% of the time

        is_anonymous = not is_anomalous and random.random() < 0.15

        if not is_anomalous:
            # Benign Service Account (cronjob/bot) 5% of the time: very predictable pattern
            if random.random() < 0.05:
                user = 0  # Dedicate user 0 as a service account
                u_role, u_clearance = self.user_roles[user], self.user_clearances[user]
                config = self.cfg_lo + 1  # Fixed JA3 for script
                if self.service_machines is not None:
                    # The cronjob lives on its server, not on whichever desk stepped.
                    sm = int(random.choice(self.service_machines))
                    machine, dev_slot = sm, self.machine_slot[sm]
                    source = self.src_lo + random.choice(sorted(self.machine_home_ips[sm]))
                    scenario = SCEN_SHARED if len(self.machine_users[sm]) > 1 else 0
                valid = self._policy_valid_actions(u_role)
                if valid:
                    res_idx, method = valid[0]  # Deterministic access
                    # Tight but continuous byte volumes: a cronjob is predictable, not
                    # bit-identical. Constant values here were a guaranteed-benign
                    # fingerprint the model could memorise.
                    feat = [*self._benign_signals(machine), float(method),
                            ROLES.index(u_role) / (len(ROLES) - 1), u_clearance / 4.0,
                            float(abs(np.random.normal(0.2, 0.02))),
                            float(abs(np.random.normal(0.5, 0.03)))]
                    return self._event(source=source, config=config, device=dev_slot, user=user,
                                       res_idx=res_idx, feat=feat, label=0, etype=0, scenario=scenario)

            # Benign human error (OPA Deny) 2% of the time for valid users
            if not is_anonymous and random.random() < 0.02:
                invalid = self._policy_violations(u_role)
                if invalid:
                    res_idx, method = self._zipf_choice(invalid, ("viol", u_role))
                    feat = [*self._benign_signals(machine), float(method),
                            ROLES.index(u_role) / (len(ROLES) - 1), u_clearance / 4.0,
                            float(abs(np.random.normal(0.1, 0.05))),
                            float(abs(np.random.normal(0.2, 0.1)))]
                    # An OPA denial triggered by a benign user mistake. label=1 because OPA
                    # does deny it, but etype=6 keeps it separable from genuine attacks:
                    # folding it into etype=1 mixed honest mistakes into the policy class.
                    return self._event(source=source, config=config, device=dev_slot, user=user,
                                       res_idx=res_idx, feat=feat, label=1, etype=6, scenario=scenario)

            if is_anonymous:
                user = self.num_registered_users + int(np.random.randint(0, self.num_guests))
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
                # Benign exploration: an authorised-but-non-habitual access (policy-clean,
                # signal-clean, label=0) — novelty alone is not an anomaly cue.
                if non_habit and random.random() < self.benign_explore_prob:
                    res_idx, method = self._zipf_choice(non_habit, ("nonhabit", user))
                elif habit:
                    res_idx, method = self._zipf_choice(habit, ("habit", user))
                elif valid:
                    res_idx, method = self._zipf_choice(valid, ("valid", u_role))
                else:
                    res_idx, method = 0, 0  # public-path fallback

            ja3, s1, s2, s3 = self._benign_signals(machine)
            bytes_in = abs(np.random.normal(0.1, 0.05))
            bytes_out = abs(np.random.normal(0.2, 0.1))
            label, etype = 0, 0
        else:
            state = self.compromised_state[machine]
            # v5 (p_compromise set): recon and exfiltration last 1-3 events each instead
            # of exactly one, so every class is measurable at a realistic base rate.
            multi = self.p_compromise is not None
            if state == 1:
                anomaly_type = "context"  # Recon phase (often triggers Snort)
                recon_left = self.compromised_chain_remaining.get(machine)
                if multi and recon_left is None:
                    recon_left = int(np.random.randint(1, 4))
                if not multi or recon_left <= 1:
                    self.compromised_state[machine] = 2
                    self.compromised_chain_remaining[machine] = int(np.random.randint(5, 12))
                else:
                    self.compromised_chain_remaining[machine] = recon_left - 1
            elif state == 2:
                anomaly_type = "lateral"  # Lateral movement phase chain
                self.compromised_chain_remaining[machine] -= 1
                if self.compromised_chain_remaining[machine] <= 0:
                    self.compromised_state[machine] = 3
                    self.compromised_chain_remaining[machine] = int(np.random.randint(1, 4)) if multi else 1
            elif state == 3:
                anomaly_type = "exfil"    # Data Exfiltration
                self.compromised_chain_remaining[machine] -= 1
                if self.compromised_chain_remaining[machine] <= 0:
                    self.compromised_state[machine] = 4 # Done
                    if multi:
                        # Post-exploitation dwell, then detection + clean-up by the SOC.
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


            bytes_in = abs(np.random.normal(0.1, 0.05))
            bytes_out = abs(np.random.normal(0.2, 0.1))

            if anomaly_type == "exfil":
                sensitive = self._sensitive_actions(u_role)
                if sensitive:
                    res_idx, method = self._zipf_choice(sensitive, ("sens", u_role))
                else:
                    res_idx, method = self._zipf_choice(self._all_actions, ("all",))
                ja3, s1, s2, s3 = self._benign_signals(machine)
                # A massive transfer is a legitimate, genuinely easy signal for this class
                # — drawn continuously rather than as an exact constant. Exfiltration gets
                # its OWN etype: folding it into etype=3 put a single-feature-separable
                # sub-population inside the lateral-movement class, whose whole premise is
                # that its edge features are indistinguishable from a benign access.
                bytes_in = abs(np.random.normal(1.0, 0.2))
                bytes_out = abs(np.random.normal(15.0, 3.0))
                etype = 5
            elif anomaly_type == "lateral":
                creds = self.harvested_creds.get(machine)
                pivot = bool(creds) and random.random() < self.p_lateral_foreign_cred
                if pivot:
                    # Pivot with a harvested credential (Euler / LANL sense of lateral
                    # movement): the compromised machine now acts as ANOTHER user, so
                    # the tell is a new device->user binding plus its timing. The role
                    # in the message is that user's real role — the IdP issued a valid
                    # token — and the destination follows that user's own policy space.
                    user = int(random.choice(creds))
                    u_role, u_clearance = self.user_roles[user], self.user_clearances[user]
                    valid = self._policy_valid_actions(u_role)
                    target = (
                        self._zipf_choice(valid, ("valid", u_role)) if valid else None
                    )
                else:
                    # Own-identity lateral: authorised but non-habitual access.
                    _habit, non_habit = self._user_actions(user, u_role)
                    target = (
                        self._zipf_choice(non_habit, ("nonhabit", user)) if non_habit else None
                    )
                if target is not None:
                    res_idx, method = target
                    ja3 = 0.0 if self.machine_legacy[machine] else 1.0
                    # Lateral movement: stealth — legitimate credentials and protocols,
                    # rarely triggers the IDS (the network must study the graph).
                    s1 = 0.0
                    s2 = 1.0 if np.random.rand() > 0.98 else 0.0  # 2%
                    s3 = 1.0 if np.random.rand() > 0.90 else 0.0  # 10%
                    etype = 3
                    if not pivot and random.random() < self.p_lateral_role_spoof:
                        # v4 only: a role claim that disagrees with the identity. Since a
                        # user's role never changes, this is a zero-false-positive rule
                        # tell — set p_lateral_role_spoof=0 for publication streams.
                        allowed_roles = [
                            r for r in ROLES
                            if r != u_role and self.policy_allows(r, method, self.resource_uris[res_idx])
                        ]
                        if allowed_roles:
                            u_role = random.choice(allowed_roles)
                            u_clearance = ROLE_CLEARANCE[u_role]  # spoofed role's clearance
                    if random.random() < self.p_lateral_new_config:
                        # a new tool on this device: a config it has never presented
                        config = self._new_tool_config(machine)
                else:
                    anomaly_type = "policy"

            if anomaly_type == "policy":
                # A genuine policy denial for this role: read-up, write-down, or a
                # missing compartment on a protected route.
                invalid = self._policy_violations(u_role)
                if not invalid:
                    # No deniable action exists for this role: labelling a random
                    # (allowed) action as etype=1 would contradict policy_allows, so
                    # the event degrades to a contextual anomaly instead.
                    anomaly_type = "context"
            if anomaly_type == "policy":
                res_idx, method = self._zipf_choice(invalid, ("viol", u_role))
                ja3, s1, s2, s3 = self._benign_signals(machine)
                etype = 1
            elif anomaly_type == "context":
                # The tell of a contextual anomaly is the compromised TLS trust and the
                # sensor alarms below — NOT the destination. So the destination is drawn
                # from exactly the same action space, under exactly the same popularity
                # law, as this role's benign traffic. Drawing from the full catalogue
                # instead made the class ~93% protected routes against ~43% for benign,
                # and the resource RISK alone then separated it at AUC 0.83.
                valid = self._policy_valid_actions(u_role)
                if valid:
                    res_idx, method = self._zipf_choice(valid, ("valid", u_role))
                else:
                    res_idx, method = self._zipf_choice(self._all_actions, ("all",))
                ja3 = 0.0 if np.random.rand() > 0.5 else 1.0
                # Recon: external attack — high probability on Edge (80%), medium on
                # Mid (50%), low on Internal (20%).
                s1 = 1.0 if np.random.rand() > 0.2 else 0.0
                s2 = 1.0 if np.random.rand() > 0.5 else 0.0
                s3 = 1.0 if np.random.rand() > 0.8 else 0.0
                etype = 2

            label = 1

        feat = [ja3, float(s1), float(s2), float(s3), float(method),
                ROLES.index(u_role) / (len(ROLES) - 1), u_clearance / 4.0,
                float(bytes_in), float(bytes_out)]
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
        self._sensitive_cache.clear()
        self._user_action_cache.clear()
        self._zipf_probs.clear()
        self._all_actions = [
            (r, m) for r, u in enumerate(self.resource_uris) for m in self.route_methods[u]
        ]
        if hasattr(self, "_anon_actions"):
            del self._anon_actions


@dataclass
class SyntheticStream:
    """Tensorised v4 stream plus the node-space layout the training pipeline needs."""

    source: torch.Tensor        # [N] global source (IP) node ids
    config: torch.Tensor        # [N] global config (JA3) node ids
    device: torch.Tensor        # [N] global device node ids
    user: torch.Tensor          # [N] global user node ids
    dst: torch.Tensor           # [N] global resource node ids
    t: torch.Tensor             # [N] timestamps
    msg: torch.Tensor           # [N, msg_dim=10] edge messages (access edge)
    y: torch.Tensor             # [N] binary labels
    types: torch.Tensor         # [N] 0=benign, 1=policy, 2=contextual, 3=lateral,
                                #     4=cred-theft, 5=exfil, 6=benign human error (denied)
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
    """Generate the offline v4 training stream (see the module docstring).

    ``seed`` seeds both ``numpy`` and the stdlib ``random`` module so the stream is
    fully reproducible — ``random.choice`` is used alongside ``np.random``. ``realism``
    forwards the v5 open-world / difficulty knobs of :class:`ZTAStreamSimulator`.
    """
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
