"""Reference authorization model (Bell-LaPadula + compartments) and the resource catalogue.

Ground truth for the synthetic generator: decides which ``(role, method, uri)`` requests are
allowed, and therefore which events are benign accesses and which are policy violations.
Deployed OPA policies (``docs/opa-policies``) should mirror this model.

Rules:
  * Simple Security Property: no read-up on GET (clearance >= classification).
  * *-Property: no write-down on writes (clearance <= classification).
  * Compartments: the route's categories must be a subset of the role's categories.
  * Trusted guard: sanitized write-down exception, admin POST only.
"""

from __future__ import annotations

import random

# HTTP method codes carried in the edge message (feature 4).
GET, POST, PUT, DELETE, PATCH = range(5)

ROLES = ["guest", "operator", "manager", "admin"]
SECURITY_LEVELS = {
    "PUBLIC": 0, "INTERNAL": 1, "CONFIDENTIAL": 2, "SECRET": 3, "TOP_SECRET": 4,
}
ROLE_CLEARANCE = {  # clearance derives from role
    "guest": 0, "operator": 1, "manager": 2, "admin": 4,
}
ROLE_CATEGORIES = {  # compartments granted per role
    "guest": set(),
    "operator": {"hr", "ops"},
    "manager": {"hr", "ops", "finance"},
    "admin": {"hr", "ops", "finance", "nuclear", "security"},
}

TRUSTED_GUARD = "/api/v1/trusted-guard/sanitized-delete-personnel"

# Orchestrator routes -> served methods. Insertion order fixes the resource index.
# Public/auth routes are open to every role and gated only by the PDP risk score.
_BASE_ROUTE_METHODS = {
    "/": {GET},
    "/materials": {GET},
    "/reserved": {GET},
    "/login": {GET},
    "/register": {GET},
    "/static": {GET},
    "/favicon.ico": {GET},
    "/api/v1/auth/register": {POST},
    "/api/v1/auth/login": {POST},
    "/api/v1/auth/verify-otp": {POST},
    "/api/v1/auth/register/begin": {POST},
    "/api/v1/auth/register/finish": {POST},
    "/api/v1/auth/login/begin": {POST},
    "/api/v1/auth/login/finish": {POST},
    "/api/v1/personnel": {GET, POST},
    "/api/v1/documents": {GET, POST, DELETE},
    "/api/v1/nuclear-materials": {GET, POST, DELETE},
    "/api/v1/reactor-parameters": {GET, POST, DELETE},
    TRUSTED_GUARD: {POST},
}

# Protected orchestrator routes: uri -> (classification, required categories).
_BASE_SECURITY_MATRIX = {
    "/api/v1/personnel":          ("INTERNAL",     {"hr"}),
    "/api/v1/documents":          ("CONFIDENTIAL", {"finance"}),
    "/api/v1/nuclear-materials":  ("TOP_SECRET",   {"nuclear"}),
    "/api/v1/reactor-parameters": ("TOP_SECRET",   {"nuclear", "security"}),
    TRUSTED_GUARD:                ("SECRET",       {"security"}),
}

# Synthetic estate: category -> (classification, required categories); "public" is unprotected.
_GENERATED_CATEGORIES = ("public", "hr", "finance", "nuclear", "ops", "security")
_GENERATED_CLASSIFICATION = {
    "hr": ("INTERNAL", {"hr"}),
    "finance": ("CONFIDENTIAL", {"finance"}),
    "ops": ("INTERNAL", {"ops"}),
    "nuclear": ("TOP_SECRET", {"nuclear"}),
    "security": ("SECRET", {"security"}),
}

# Inherent resource risk (node_features[:, 4]): per classification, then per-route overrides.
_CLASSIFICATION_RISK = {
    "INTERNAL": 0.5, "CONFIDENTIAL": 0.6, "SECRET": 0.8, "TOP_SECRET": 0.9,
}
_RISK_OVERRIDES = {
    "/api/v1/personnel": 0.6,
    "/api/v1/documents": 0.7,
    "/api/v1/nuclear-materials": 0.7,
    "/api/v1/reactor-parameters": 1.0,
    TRUSTED_GUARD: 1.0,
}

NUM_BASE_ROUTES = len(_BASE_ROUTE_METHODS)
_DEFAULT_NUM_GENERATED = 1000 - NUM_BASE_ROUTES  # default catalogue = TGNConfig.num_resources


def build_resource_universe(num_generated: int = _DEFAULT_NUM_GENERATED, seed: int = 42):
    """Base routes plus ``num_generated`` synthetic endpoints drawn with ``random.Random(seed)``.

    Returns ``(route_methods, security_matrix, resource_uris, resource_risk)``. Keys are the
    normalized URI paths the orchestrator sends.
    """
    route_methods = dict(_BASE_ROUTE_METHODS)
    security_matrix = dict(_BASE_SECURITY_MATRIX)

    rng = random.Random(seed)
    for i in range(num_generated):
        cat = rng.choice(_GENERATED_CATEGORIES)
        if cat == "public":
            uri = f"/api/v2/public/resource_{i}"
            route_methods[uri] = {GET, POST}
        else:
            uri = f"/internal/{cat}/doc_{i}"
            route_methods[uri] = {GET, POST, DELETE}
            security_matrix[uri] = _GENERATED_CLASSIFICATION[cat]

    resource_uris = list(route_methods)
    resource_risk = {uri: 0.0 for uri in resource_uris}
    for uri, (cls, _cats) in security_matrix.items():
        resource_risk[uri] = _CLASSIFICATION_RISK[cls]
    resource_risk.update(_RISK_OVERRIDES)
    return route_methods, security_matrix, resource_uris, resource_risk


# Default catalogue (seed 42) for tests and tools that need no simulator.
ROUTE_METHODS, SECURITY_MATRIX, RESOURCE_URIS, RESOURCE_RISK = build_resource_universe()


def policy_allows(role: str, method: int, uri: str,
                  route_methods: dict = ROUTE_METHODS,
                  security_matrix: dict = SECURITY_MATRIX) -> bool:
    """True iff ``role`` may call ``method`` on ``uri`` under the given catalogue.

    Checks, in order: the route serves the method; public routes allow everyone; the role
    holds every required compartment; Bell-LaPadula (trusted guard: admin POST only).
    """
    if method not in route_methods.get(uri, set()):
        return False
    if uri not in security_matrix:
        return True  # public / auth / static route
    classification, categories = security_matrix[uri]
    if not categories.issubset(ROLE_CATEGORIES[role]):
        return False
    if uri == TRUSTED_GUARD:
        return role == "admin" and method == POST
    clearance = ROLE_CLEARANCE[role]
    level = SECURITY_LEVELS[classification]
    if method == GET:
        return clearance >= level  # no read-up
    return clearance <= level      # no write-down
