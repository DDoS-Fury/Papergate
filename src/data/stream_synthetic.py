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
3=lateral movement, 4=credential theft. ``scenario`` is a per-event bitmask of
benign-context flags (an event can be several at once): 1=roaming, 2=recently wiped
cookie device, 4=shared device.

Edge message layout (7-dim): ``[ja3, s1, s2, s3, method, role, clearance]`` —
TLS-fingerprint trust, the three Snort/sensor probes, the HTTP method code and the
requesting identity's role/clearance (possibly stolen).
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
# Resource node keys MUST be the exact URIs the orchestrator sends as key_dst
# (after its normalizeAIPath): no synthetic suffixes, one node per real route.
RESOURCE_URIS = list(ROUTE_METHODS)


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

# Per-resource inherent RISK (node_feat index 4), in [0, 1]. Single source of truth:
# this MIRRORS getResourceSensitivity() in
# services/security-orchestrator/internal/handler/handler.go (and the impact encoded
# in policy.rego matrice_sicurezza), collapsed to a per-resource scalar (the max over
# methods, since the resource node is method-agnostic). Routes the orchestrator never
# scores as sensitive (public/auth/static) are 0.0. Keep aligned if the Go map changes.
RESOURCE_RISK = {uri: 0.0 for uri in RESOURCE_URIS}
RESOURCE_RISK.update({
    "/api/v1/personnel": 0.6,                                  # max(GET/POST 0.4, DELETE 0.6)
    "/api/v1/documents": 0.7,                                  # max(GET/POST 0.5, DELETE 0.7)
    "/api/v1/nuclear-materials": 0.7,                          # max(0.5, DELETE 0.7)
    "/api/v1/reactor-parameters": 1.0,                         # max(GET/POST 0.8, DELETE 1.0)
    "/api/v1/trusted-guard/sanitized-delete-personnel": 1.0,  # sanitized delete = highest
})

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
        num_users: int = 50,
        num_devices: int = 80,
        num_sources: int = 150,
        num_configs: int = 40,
        num_resources: int = 19,
        num_wipe_slots: int = 16,
        num_theft_slots: int = 64,
        benign_explore_prob: float = 0.15,
        p_roam: float = 0.10,
        p_shared_device: float = 0.20,
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

        assert num_resources == len(RESOURCE_URIS), (
            f"num_resources ({num_resources}) must equal len(RESOURCE_URIS) "
            f"({len(RESOURCE_URIS)}): set TGNConfig.num_resources accordingly"
        )
        self.num_users = num_users
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

        # --- node index layout: [users][device slots][source slots][config slots][resources] ----
        self.user_lo = 0
        self.dev_lo = num_users
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
        self.user_roles = [str(np.random.choice(ROLES)) for _ in range(num_users)]
        self.user_clearances = [ROLE_CLEARANCE[r] for r in self.user_roles]

        # --- physical machines (stable across cookie wipes) -------------------------
        # Tier: 0=no cert/tpm, 1=cert, 2=cert+tpm. tier-2 machines are TPM-keyed;
        # the rest are cookie-keyed (re-keyed on wipe).
        self.machine_tiers = [int(np.random.choice([0, 1, 2], p=[0.2, 0.5, 0.3]))
                              for _ in range(num_devices)]
        # Owner(s): base round-robin owner + extra users on shared machines.
        self.machine_users: list[list[int]] = []
        for m in range(num_devices):
            users = [m % num_users]
            if np.random.rand() < p_shared_device:
                extra = np.random.randint(1, 3)
                pool = [u for u in range(num_users) if u not in users]
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
        self.keys: list = [None] * self.num_nodes
        for u in range(num_users):
            self.keys[u] = u
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
            self.keys[self.res_lo + r] = RESOURCE_URIS[r]

        # --- static node features (16-dim) -------------------------------------------
        # Index map: [2]=device tier, [3]=resource priority, [4]=resource RISK,
        # [5]=source network internal(1)/external(0), [14]=trust score.
        nf = torch.zeros(self.num_nodes, 16)
        nf[:, 14] = 1.0  # trust score default
        for m in range(num_devices):
            nf[self.dev_lo + m, 2] = self.machine_tiers[m] / 2.0
        for r in range(num_resources):
            nf[self.res_lo + r, 3] = r / float(num_resources - 1)
            if use_resource_risk:
                nf[self.res_lo + r, 4] = RESOURCE_RISK[RESOURCE_URIS[r]]
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
        self.user_habitual: list[set[tuple[int, int]]] = []
        for u in range(num_users):
            valid = self._policy_valid_actions(self.user_roles[u])
            if valid:
                k = max(1, len(valid) // 2)
                hab_idx = np.random.choice(len(valid), size=k, replace=False)
                self.user_habitual.append({valid[j] for j in hab_idx})
            else:
                self.user_habitual.append(set())

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
        self.compromised_state: dict[int, int] = {}  # machine -> kill-chain phase
        self._active_thefts: list[dict] = []

    # --- helpers ------------------------------------------------------------------------
    def _policy_valid_actions(self, role: str):
        """``(resource_idx, method)`` actions OPA would ALLOW for this role (cached)."""
        if role not in self._valid_cache:
            self._valid_cache[role] = [
                (r, m)
                for r, uri in enumerate(RESOURCE_URIS)
                for m in ROUTE_METHODS[uri]
                if policy_allows(role, m, uri)
            ]
        return self._valid_cache[role]

    def _policy_violations(self, role: str):
        """``(resource_idx, method)`` actions on PROTECTED routes that OPA would DENY for
        this role — genuine policy.rego violations (read-up / write-down / missing
        compartment). Public routes can never be a policy violation."""
        if role not in self._violation_cache:
            self._violation_cache[role] = [
                (r, m)
                for r, uri in enumerate(RESOURCE_URIS)
                for m in ROUTE_METHODS[uri]
                if uri in SECURITY_MATRIX and not policy_allows(role, m, uri)
            ]
        return self._violation_cache[role]

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
        valid = self._policy_valid_actions(role)
        sensitive = [(r, m) for r, m in valid if RESOURCE_URIS[r] in SECURITY_MATRIX]
        res_idx, method = random.choice(sensitive or valid or [(0, 0)])
        feat = [1.0, 0.0, 0.0, 0.0, float(method),
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
    def step(self) -> dict:
        self.t += int(np.random.exponential(scale=300.0))
        self.step_count += 1

        # Credential-theft incidents: a burst of requests from a never-seen attacker
        # IP + device as an existing victim user (signal-clean, policy-clean).
        if self._active_thefts and random.random() < 0.6:
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
                "victim": int(np.random.randint(0, self.num_users)),
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

        if not is_anomalous:
            valid = self._policy_valid_actions(u_role)
            habit = [a for a in valid if a in self.user_habitual[user]]
            non_habit = [a for a in valid if a not in self.user_habitual[user]]
            # Benign exploration: an authorised-but-non-habitual access (policy-clean,
            # signal-clean, label=0) — novelty alone is not an anomaly cue.
            if non_habit and random.random() < self.benign_explore_prob:
                res_idx, method = random.choice(non_habit)
            elif habit:
                res_idx, method = random.choice(habit)
            elif valid:
                res_idx, method = random.choice(valid)
            else:
                res_idx, method = 0, 0  # public-path fallback
            ja3, s1, s2, s3 = 1.0, 0.0, 0.0, 0.0
            label, etype = 0, 0
        else:
            state = self.compromised_state[machine]
            if state == 1:
                anomaly_type = "context"  # Recon phase (often triggers Snort)
                self.compromised_state[machine] = 2
            elif state == 2:
                anomaly_type = "lateral"  # Lateral movement phase
                self.compromised_state[machine] = 3
            else:
                anomaly_type = np.random.choice(["policy", "context", "lateral"])

            if anomaly_type == "lateral":
                valid = self._policy_valid_actions(u_role)
                non_habit = [a for a in valid if a not in self.user_habitual[user]]
                if non_habit:
                    res_idx, method = random.choice(non_habit)
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
                    res_idx, method = random.choice(invalid)
                else:
                    res_idx = int(np.random.randint(0, self.num_resources))
                    method = int(np.random.randint(0, 4))
                ja3, s1, s2, s3 = 1.0, 0.0, 0.0, 0.0
                etype = 1
            elif anomaly_type == "context":
                res_idx = int(np.random.randint(0, self.num_resources))
                method = int(np.random.randint(0, 4))
                ja3 = 0.0 if np.random.rand() > 0.5 else 1.0
                # Recon: attacco esterno — alta probabilità su Edge (80%), media su
                # Mid (50%), bassa su Internal (20%).
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


@dataclass
class SyntheticStream:
    """Tensorised v4 stream plus the node-space layout the training pipeline needs."""

    source: torch.Tensor        # [N] global source (IP) node ids
    config: torch.Tensor        # [N] global config (JA3) node ids
    device: torch.Tensor        # [N] global device node ids
    user: torch.Tensor          # [N] global user node ids
    dst: torch.Tensor           # [N] global resource node ids
    t: torch.Tensor             # [N] timestamps
    msg: torch.Tensor           # [N, 7] edge messages (access edge)
    y: torch.Tensor             # [N] binary labels
    types: torch.Tensor         # [N] anomaly types (0..4)
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
    num_users=50,
    num_devices=80,
    num_sources=150,
    num_configs=40,
    num_resources=19,
    num_events=5000,
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
        num_users=num_users, num_devices=num_devices, num_sources=num_sources,
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
