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
  * **credential theft** (``p_cred_theft``, etype 4): a never-seen attacker IP +
    never-seen attacker device + never-seen attacker config suddenly issue requests as
    an existing victim user. Policy-clean and signal-clean — only the broken
    ``ip -> config -> device -> user`` binding pattern exposes it.

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
An earlier revision also carried ``http_status``, which is a **response** field: using it
to decide whether to allow the request is a causality violation, and since it took a
class-specific constant it was also a near-deterministic label channel. It was removed.

Two invariants this generator must preserve, both regression-tested in
``tests/test_leakage_audit.py``:

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

ROLES = ["guest", "operator", "manager", "admin"]
CLEARANCES = ["PUBLIC", "INTERNAL", "CONFIDENTIAL", "SECRET", "TOP_SECRET"]
WRITE_METHODS = {1, 2, 3, 4}  # POST/PUT/DELETE/PATCH (0=GET is the only read)

# --- Bell-LaPadula model: a 1:1 mirror of infra/opa/policy.rego ---------------------
# These three maps are kept in lockstep with the rego `livelli`, `ruoli_to_blp` and
# `matrice_sicurezza`. The generator's notion of "policy-compliant" MUST equal OPA's
# allow decision so the model's benign manifold is exactly the set of accesses OPA
# permits — and the `etype=1` violations are genuine OPA denials.
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
# methods declared in matrice_sicurezza; public/auth routes are GET or POST. Public routes
# (everything outside SECURITY_MATRIX) are allowed for every role — OPA gates them on
# ai_score only (rego `public_paths`). NB: /api/v1/auth/register/{begin,finish} are PUBLIC
# in the current policy (they sit in public_paths), no longer authenticated-only.
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

# Inherent RISK per classification (node_feat index 4). Single source of truth: this
# MIRRORS getResourceSensitivity() in services/security-orchestrator/internal/handler/
# handler.go (and the impact encoded in policy.rego matrice_sicurezza), collapsed to a
# per-resource scalar (the max over methods, since the resource node is method-agnostic).
# Routes the orchestrator never scores as sensitive (public/auth/static) are 0.0.
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
    """OPA-equivalent allow decision for ``(role, method, route)`` — a direct port of
    ``infra/opa/policy.rego`` (Bell-LaPadula + compartments + trusted-guard exception).

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
# Per-(lateral)-event chance the compromised machine presents a config it has never used
# (a globally-known fingerprint, but a new *tool* on this device) — the lateral-movement
# config tell. Otherwise the lateral event keeps the machine's habitual config (stealth).
_P_LATERAL_NEW_CONFIG = 0.5


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

        # --- resource popularity, DECOUPLED from the resource index --------------------
        # Access frequency follows a Zipf law, but the rank a resource gets is a random
        # permutation of the index space rather than the index itself. Otherwise the
        # resource id would encode popularity, and since benign traffic concentrates on
        # popular resources while attacks do not, the id alone would separate the classes
        # — a pure dataset artifact. See tests/test_leakage_audit.py.
        pop_rank = np.random.permutation(num_resources).astype(np.float64)
        self._res_pop_weight = 1.0 / ((pop_rank + 1.0) ** 1.2)

        # --- node index layout: [users][device slots][source slots][config slots][resources] ----
        self.user_lo = 0
        self.dev_lo = self.num_users
        self.dev_slots = num_devices + num_wipe_slots + num_theft_slots
        self.src_lo = self.dev_lo + self.dev_slots
        self.src_slots = num_sources + num_theft_slots
        self.cfg_lo = self.src_lo + self.src_slots
        # Config slot 0 is the generic ``conf:guest``; [1, num_configs) are habitual
        # fingerprints; the trailing num_theft_slots hold never-seen attacker configs.
        self.cfg_slots = num_configs + num_theft_slots
        self.res_lo = self.cfg_lo + self.cfg_slots
        self.num_nodes = self.res_lo + num_resources

        # --- users -----------------------------------------------------------------
        # Clearance is NOT independent: like policy.rego it derives from the role
        # (ruoli_to_blp), so the role/clearance pair in every benign message is exactly
        # what the JWT would carry in production.
        self.user_roles = [str(np.random.choice(ROLES)) for _ in range(self.num_registered_users)]
        self.user_roles.extend(["guest"] * self.num_guests)
        self.user_clearances = [ROLE_CLEARANCE[r] for r in self.user_roles]

        # --- physical machines (stable across cookie wipes) -------------------------
        # Tier: 0=no cert/tpm, 1=cert, 2=cert+tpm. tier-2 machines are TPM-keyed;
        # the rest are cookie-keyed (re-keyed on wipe).
        self.machine_tiers = [int(np.random.choice([0, 1, 2], p=[0.2, 0.5, 0.3]))
                              for _ in range(num_devices)]
        # Owner(s): base round-robin owner + extra users on shared machines.
        self.machine_users: list[list[int]] = []
        for m in range(num_devices):
            users = [m % self.num_registered_users]
            if np.random.rand() < p_shared_device:
                extra = np.random.randint(1, 4)
                pool = [u for u in range(self.num_registered_users) if u not in users]
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

        # --- habitual client configs (TLS/JA3) per machine --------------------------
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

        # --- external keys per node slot ---------------------------------------------
        self.keys: list[str | None] = [None] * self.num_nodes
        for u in range(self.num_registered_users):
            self.keys[self.user_lo + u] = f"user_{u:04d}"
        for g in range(self.num_guests):
            self.keys[self.user_lo + self.num_registered_users + g] = f"guest_{g:04d}"
        for m in range(num_devices):
            tier = self.machine_tiers[m]
            self.keys[self.dev_lo + m] = (
                f"tpm:{m:04d}" if tier == 2 else f"ck:{m:04d}-g0"
            )
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
        for k in range(num_theft_slots):
            self.keys[self.src_lo + num_sources + k] = f"src:203.0.113.{k}"
        # Config keys: slot 0 = generic guest, [1, num_configs) = habitual JA3
        # fingerprints, trailing slots = never-seen attacker configs (credential theft).
        self.keys[self.cfg_lo] = "conf:guest"
        for c in range(1, num_configs):
            self.keys[self.cfg_lo + c] = f"conf:{c:04d}"
        for k in range(num_theft_slots):
            self.keys[self.cfg_lo + num_configs + k] = f"_spare_cfg_{k}"
        for r in range(num_resources):
            self.keys[self.res_lo + r] = self.resource_uris[r]

        # --- static node features (16-dim) -------------------------------------------
        # Index map: [2]=device tier, [3]=UNUSED, [4]=resource RISK,
        # [5]=source network internal(1)/external(0), [14]=trust score.
        #
        # [3] used to carry ``r / (num_resources - 1)``, i.e. the raw resource index
        # dressed up as a "priority". It was redundant with the RISK in [4] and, because
        # the index correlated with access frequency, it leaked the label: that single
        # column reached AUC ~0.92 on every anomaly class. It is deliberately left at 0.
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

        # --- behaviour model -----------------------------------------------------------
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

        # --- mutable state ---------------------------------------------------------------
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
        self._next_wipe_slot = 0
        self._next_theft_slot = 0
        self._slot_age: dict[int, int] = {}      # events seen by a re-keyed (wiped) slot
        self.last_user_t: dict[int, int] = {}
        self.compromised_state: dict[int, int] = {}  # machine -> kill-chain phase
        self.compromised_chain_remaining: dict[int, int] = {} # machine -> steps left in lateral chain
        self._active_thefts: list[dict] = []

    # --- helpers ------------------------------------------------------------------------
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

    def _maybe_wipe_cookie(self, machine: int) -> None:
        """Re-key a cookie-identified machine onto a fresh (cold) device slot."""
        if (
            not self.guest_device_fallback  # guest devices have no per-machine cookie
            and self.machine_tiers[machine] < 2
            and self._next_wipe_slot < self.num_wipe_slots
            and random.random() < self.p_cookie_wipe
        ):
            slot = self.dev_lo + self.num_devices + self._next_wipe_slot
            gen = self._next_wipe_slot + 1
            self._next_wipe_slot += 1
            self.keys[slot] = f"ck:{machine:04d}-g{gen}"
            self.node_features[slot, 2] = self.machine_tiers[machine] / 2.0
            self.machine_slot[machine] = slot
            self._slot_age[slot] = 0

    def _habitual_config(self, machine: int) -> int:
        """Global slot of a benign client config for ``machine`` (its habitual JA3, or
        occasionally the generic ``conf:guest``)."""
        if random.random() < _P_GUEST_CONFIG:
            return self.cfg_lo  # conf:guest
        return self.cfg_lo + int(random.choice(self.machine_configs[machine]))

    def _new_tool_config(self, machine: int) -> int:
        """Global slot of a config ``machine`` has never used (a new tool on a known
        device — the lateral-movement config tell). Falls back to the habitual config
        when the pool offers no alternative."""
        habit = set(self.machine_configs[machine])
        others = [c for c in range(1, self.num_configs) if c not in habit]
        if not others:
            return self._habitual_config(machine)
        return self.cfg_lo + int(random.choice(others))

    def _emit_theft_event(self, incident: dict) -> dict:
        """One credential-theft request: attacker IP + config + device, victim identity."""
        u = incident["victim"]
        role, clr = self.user_roles[u], self.user_clearances[u]
        # The attacker holds the victim's credentials: OPA's role/clearance/category
        # checks all pass. Prefer protected routes (the loot) over public ones.
        sensitive = self._sensitive_actions(role)
        if sensitive:
            res_idx, method = self._zipf_choice(sensitive, ("sens", role))
        else:
            valid = self._policy_valid_actions(role)
            res_idx, method = (self._zipf_choice(valid, ("valid", role)) if valid else (0, 0))
        # Byte volumes are drawn from the SAME laws as benign traffic. Credential theft is
        # policy-clean and signal-clean by construction — only the broken
        # ip -> config -> device -> user binding exposes it. Constant byte values here used
        # to identify the class with 100% precision and recall, i.e. pure label leakage.
        feat = [1.0, 0.0, 0.0, 0.0, float(method),
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

    # --- one event -------------------------------------------------------------------
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
        self.t += int(np.random.exponential(scale=self._current_interarrival_scale()))
        self.step_count += 1

        # Credential-theft incidents: requests from a never-seen attacker IP + device as
        # an existing victim user (signal-clean, policy-clean). They INTERLEAVE with normal
        # traffic rather than arriving back-to-back: a high emission rate made the victim's
        # inter-request gap collapse, and that gap is an edge feature — the class became
        # identifiable from ``log1p(Δt)`` alone, with no need for the binding structure that
        # is supposed to be the only thing exposing it.
        if self._active_thefts and random.random() < 0.15:
            return self._emit_theft_event(random.choice(self._active_thefts))
        if (
            random.random() < self.p_cred_theft
            and self._next_theft_slot < self.num_theft_slots
        ):
            k = self._next_theft_slot
            self._next_theft_slot += 1
            if self.guest_device_fallback and self._guest_dev_slot is not None:
                # Attacker device is TPM-less too, so under the guest-fallback policy it
                # collapses onto the shared guest node; the never-seen IP / JA3 remain the
                # only novelty tells (the device-identity tell is intentionally lost).
                dev_slot = self._guest_dev_slot
            else:
                dev_slot = self.dev_lo + self.num_devices + self.num_wipe_slots + k
                self.keys[dev_slot] = f"ck:atk-{k:03d}"  # attacker gets a fresh cookie
            cfg_slot = self.cfg_lo + self.num_configs + k
            self.keys[cfg_slot] = f"conf:atk-{k:03d}"  # ...and a never-seen JA3
            incident = {
                "victim": int(np.random.randint(0, self.num_registered_users)),
                "dev_slot": dev_slot,
                "src_slot": self.src_lo + self.num_sources + k,
                "cfg_slot": cfg_slot,
                "remaining": int(np.random.randint(3, 7)),
            }
            self._active_thefts.append(incident)
            return self._emit_theft_event(incident)

        # --- pick the physical machine (progressively admitted), its user and IP ---
        if self.admission_horizon:
            max_m = min(
                self.num_devices,
                int(self.step_count / self.admission_horizon * self.num_devices) + 1,
            )
        else:
            max_m = self.num_devices
        machine = int(np.random.randint(0, max_m))
        self._maybe_wipe_cookie(machine)
        dev_slot = self.machine_slot[machine]
        user = int(random.choice(self.machine_users[machine]))
        u_role, u_clearance = self.user_roles[user], self.user_clearances[user]

        scenario = 0
        if len(self.machine_users[machine]) > 1:
            scenario |= SCEN_SHARED
        if self._slot_age.get(dev_slot, _WIPE_COLD_EVENTS) < _WIPE_COLD_EVENTS:
            scenario |= SCEN_WIPED

        home = self.machine_home_ips[machine]
        if random.random() < self.p_roam:
            scenario |= SCEN_ROAMING
            src_local = int(np.random.randint(0, self.num_sources))
            while src_local in home:
                src_local = int(np.random.randint(0, self.num_sources))
        else:
            src_local = random.choice(sorted(home))
        source = self.src_lo + src_local

        # Client config (TLS/JA3): the machine's habitual fingerprint by default. It is a
        # software identity, so roaming (a network change) does NOT change it; lateral
        # movement may swap in a new tool (see below).
        config = self._habitual_config(machine)

        # --- APT kill chain on the physical machine (recon -> lateral -> exfil) ----
        if np.random.rand() < 0.005 and machine not in self.compromised_state:
            self.compromised_state[machine] = 1
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
                valid = self._policy_valid_actions(u_role)
                if valid:
                    res_idx, method = valid[0]  # Deterministic access
                    # Tight but continuous byte volumes: a cronjob is predictable, not
                    # bit-identical. Constant values here were a guaranteed-benign
                    # fingerprint the model could memorise.
                    feat = [1.0, 0.0, 0.0, 0.0, float(method),
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
                    feat = [1.0, 0.0, 0.0, 0.0, float(method),
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
                dev_slot = self._guest_dev_slot if self._guest_dev_slot is not None else self.dev_lo + machine
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

            ja3, s1, s2, s3 = 1.0, 0.0, 0.0, 0.0
            bytes_in = abs(np.random.normal(0.1, 0.05))
            bytes_out = abs(np.random.normal(0.2, 0.1))
            label, etype = 0, 0
        else:
            state = self.compromised_state[machine]
            if state == 1:
                anomaly_type = "context"  # Recon phase (often triggers Snort)
                self.compromised_state[machine] = 2
                self.compromised_chain_remaining[machine] = int(np.random.randint(5, 12))
            elif state == 2:
                anomaly_type = "lateral"  # Lateral movement phase chain
                self.compromised_chain_remaining[machine] -= 1
                if self.compromised_chain_remaining[machine] <= 0:
                    self.compromised_state[machine] = 3
            elif state == 3:
                anomaly_type = "exfil"    # Data Exfiltration
                self.compromised_state[machine] = 4 # Done
            else:
                anomaly_type = np.random.choice(["policy", "context", "lateral"])

            bytes_in = abs(np.random.normal(0.1, 0.05))
            bytes_out = abs(np.random.normal(0.2, 0.1))

            if anomaly_type == "exfil":
                sensitive = self._sensitive_actions(u_role)
                if sensitive:
                    res_idx, method = self._zipf_choice(sensitive, ("sens", u_role))
                else:
                    res_idx, method = self._zipf_choice(self._all_actions, ("all",))
                ja3, s1, s2, s3 = 1.0, 0.0, 0.0, 0.0
                # A massive transfer is a legitimate, genuinely easy signal for this class
                # — drawn continuously rather than as an exact constant. Exfiltration gets
                # its OWN etype: folding it into etype=3 put a single-feature-separable
                # sub-population inside the lateral-movement class, whose whole premise is
                # that its edge features are indistinguishable from a benign access.
                bytes_in = abs(np.random.normal(1.0, 0.2))
                bytes_out = abs(np.random.normal(15.0, 3.0))
                etype = 5
            elif anomaly_type == "lateral":
                _habit, non_habit = self._user_actions(user, u_role)
                if non_habit:
                    res_idx, method = self._zipf_choice(non_habit, ("nonhabit", user))
                    ja3 = 1.0
                    # Movimento laterale: stealth — credenziali e protocolli legittimi,
                    # raramente fa scattare l'IDS (la rete deve studiare il grafo).
                    s1 = 0.0
                    s2 = 1.0 if np.random.rand() > 0.98 else 0.0  # 2%
                    s3 = 1.0 if np.random.rand() > 0.90 else 0.0  # 10%
                    etype = 3
                    if random.random() < 0.5:  # stolen identity inside the message
                        u_role = random.choice([r for r in ROLES if r != u_role])
                        u_clearance = ROLE_CLEARANCE[u_role]  # spoofed role's clearance
                    if random.random() < _P_LATERAL_NEW_CONFIG:
                        # a new tool on this device: a config it has never presented
                        config = self._new_tool_config(machine)
                else:
                    anomaly_type = "policy"

            if anomaly_type == "policy":
                # A genuine OPA denial for this role: read-up, write-down, or a missing
                # compartment on a protected route.
                invalid = self._policy_violations(u_role)
                if invalid:
                    res_idx, method = self._zipf_choice(invalid, ("viol", u_role))
                else:
                    res_idx, method = self._zipf_choice(self._all_actions, ("all",))
                ja3, s1, s2, s3 = 1.0, 0.0, 0.0, 0.0
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
                # Recon: attacco esterno — alta probabilità su Edge (80%), media su
                # Mid (50%), bassa su Internal (20%).
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
        new_feat[0, 3] = self.num_resources / max(1.0, float(self.num_resources))
        if self.use_resource_risk:
            new_feat[0, 4] = risk
            
        self.node_features = torch.cat([self.node_features, new_feat], dim=0)
        
        self.num_resources += 1
        self.num_nodes += 1
        
        # Invalidate caches so the new resource is picked up
        self._valid_cache.clear()
        self._violation_cache.clear()
        self._zipf_probs.clear()


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
) -> SyntheticStream:
    """Generate the offline v4 training stream (see the module docstring).

    ``seed`` seeds both ``numpy`` and the stdlib ``random`` module so the stream is
    fully reproducible — ``random.choice`` is used alongside ``np.random``.
    """
    sim = ZTAStreamSimulator(
        num_users=num_users, num_guests=num_guests, num_devices=num_devices, num_sources=num_sources,
        num_configs=num_configs, num_resources=num_resources, num_wipe_slots=num_wipe_slots,
        num_theft_slots=num_theft_slots, benign_explore_prob=benign_explore_prob,
        p_roam=p_roam, p_shared_device=p_shared_device, p_cookie_wipe=p_cookie_wipe,
        p_cred_theft=p_cred_theft, admission_horizon=num_events, seed=seed,
        use_resource_risk=use_resource_risk, use_source_internal=use_source_internal,
        guest_device_fallback=guest_device_fallback,
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
