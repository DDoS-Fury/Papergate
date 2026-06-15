"""Bell-LaPadula compliance tests — the synthetic generator's authorization model
MUST mirror infra/opa/policy.rego, so the model's benign manifold equals the set of
accesses OPA allows and the ``etype=1`` events are genuine OPA denials.

    pytest tests/test_policy_blp.py

Reference (kept in lockstep): ``matrice_sicurezza`` / ``ruoli_to_blp`` in
``infra/opa/policy.rego``. Method codes: 0=GET, 1=POST, 2=PUT, 3=DELETE, 4=PATCH.
"""

from graphagate.data.stream_synthetic import (
    RESOURCE_URIS,
    ROLE_CLEARANCE,
    SECURITY_MATRIX,
    ZTAStreamSimulator,
    policy_allows,
)

GET, POST, DELETE = 0, 1, 3
DOCS = "/api/v1/documents"
PERS = "/api/v1/personnel"
NUKE = "/api/v1/nuclear-materials"
REACTOR = "/api/v1/reactor-parameters"
GUARD = "/api/v1/trusted-guard/sanitized-delete-personnel"


def test_no_read_up_on_get():
    # CONFIDENTIAL document GET needs clearance >= CONFIDENTIAL and the finance category.
    assert not policy_allows("operator", GET, DOCS)   # clearance INTERNAL < CONFIDENTIAL
    assert policy_allows("manager", GET, DOCS)        # clearance == CONFIDENTIAL
    assert policy_allows("admin", GET, DOCS)          # clearance TOP_SECRET >= CONFIDENTIAL


def test_no_write_down_on_writes():
    # *-Property: a write requires clearance <= classification.
    assert policy_allows("manager", POST, DOCS)       # CONFIDENTIAL == CONFIDENTIAL
    assert not policy_allows("admin", POST, DOCS)     # TOP_SECRET would write down
    assert not policy_allows("admin", DELETE, DOCS)
    # personnel is INTERNAL: only the operator (clearance INTERNAL) may write it.
    assert policy_allows("operator", POST, PERS)
    assert not policy_allows("manager", POST, PERS)   # CONFIDENTIAL > INTERNAL: write-down
    assert not policy_allows("admin", POST, PERS)


def test_compartments_required():
    # nuclear-materials carries the {"nuclear"} compartment — only admin holds it.
    assert not policy_allows("manager", GET, NUKE)    # manager lacks the nuclear category
    assert policy_allows("admin", GET, NUKE)
    assert policy_allows("admin", POST, NUKE)         # TOP_SECRET == TOP_SECRET (no write-down)
    # reactor-parameters needs both nuclear AND security.
    assert not policy_allows("manager", GET, REACTOR)
    assert policy_allows("admin", GET, REACTOR)


def test_trusted_guard_sanitized_write_down():
    # SECRET route: only admin POST is allowed (the sanitized write-down exception).
    assert policy_allows("admin", POST, GUARD)
    assert not policy_allows("manager", POST, GUARD)
    assert not policy_allows("operator", POST, GUARD)
    assert not policy_allows("admin", GET, GUARD)     # route serves POST only


def test_guest_denied_everywhere_protected_allowed_public():
    for uri in SECURITY_MATRIX:
        for m in (GET, POST, DELETE):
            assert not policy_allows("guest", m, uri), (uri, m)
    # Public routes: allowed for every role (OPA gates them on ai_score only).
    assert policy_allows("guest", GET, "/")
    assert policy_allows("guest", POST, "/api/v1/auth/login")
    assert policy_allows("guest", POST, "/api/v1/auth/register/begin")  # now public


def test_unserved_method_denied():
    # personnel does not serve DELETE; trusted-guard does not serve GET.
    assert not policy_allows("operator", DELETE, PERS)
    assert not policy_allows("admin", GET, GUARD)


def test_generated_stream_is_policy_compliant():
    """Every benign event is an OPA-allowed access; every etype=1 event is an OPA denial."""
    sim = ZTAStreamSimulator(
        num_users=60, num_devices=40, num_sources=80,
        num_resources=len(RESOURCE_URIS), num_wipe_slots=8, num_theft_slots=16, seed=0,
    )
    roles = sim.user_roles
    res_lo = sim.res_lo
    events = [sim.step() for _ in range(8000)]

    n_benign = n_viol = 0
    for e in events:
        role = roles[e["user"]]
        uri = RESOURCE_URIS[e["dst"] - res_lo]
        method = int(round(e["features"][4]))
        if e["label"] == 0 and e["etype"] == 0:
            n_benign += 1
            assert policy_allows(role, method, uri), ("benign denied", role, method, uri)
        if e["etype"] == 1:
            n_viol += 1
            assert not policy_allows(role, method, uri), ("fake violation", role, method, uri)

    # The stream must actually exercise both classes (guards against a vacuous pass).
    assert n_benign > 1000
    assert n_viol > 0


def test_clearance_derived_from_role():
    sim = ZTAStreamSimulator(num_users=40, num_devices=20, num_sources=40,
                             num_resources=len(RESOURCE_URIS), seed=1)
    for u in range(40):
        assert sim.user_clearances[u] == ROLE_CLEARANCE[sim.user_roles[u]]
