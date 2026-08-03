"""Unit tests for the v3 refinements: key namespacing (anti-aliasing), the source
network internal/external classifier, and the resource-risk node feature.

    pytest tests/test_netclass.py
"""

from pytest import approx

from graphagate.netclass import ip_is_internal, strip_source_prefix
from graphagate.data.stream_synthetic import (
    RESOURCE_RISK,
    RESOURCE_URIS,
    ZTAStreamSimulator,
)


def test_ip_is_internal_rfc1918_only():
    # RFC1918 private == internal.
    for ip in ("10.0.0.7", "10.255.1.1", "172.16.0.1", "172.31.255.254", "192.168.1.20"):
        assert ip_is_internal(ip), ip
    # Just outside RFC1918, CGNAT, public, TEST-NET, loopback, link-local == external.
    for ip in ("172.15.0.1", "172.32.0.1", "100.64.0.9", "203.0.113.4",
               "8.8.8.8", "127.0.0.1", "169.254.1.1"):
        assert not ip_is_internal(ip), ip
    # Accepts a namespaced key and tolerates garbage (fail-safe external).
    assert ip_is_internal("src:10.0.0.7")
    assert not ip_is_internal("src:203.0.113.4")
    assert not ip_is_internal("not-an-ip")
    assert not ip_is_internal(None)


def test_strip_source_prefix():
    assert strip_source_prefix("src:10.0.0.7") == "10.0.0.7"
    assert strip_source_prefix("10.0.0.7") == "10.0.0.7"  # idempotent on bare IP


def test_resource_risk_covers_every_uri():
    # Every resource node has a defined risk; the high-impact routes match the
    # orchestrator's getResourceSensitivity (collapsed per-resource max).
    assert set(RESOURCE_RISK) == set(RESOURCE_URIS)
    assert RESOURCE_RISK["/api/v1/reactor-parameters"] == 1.0
    assert RESOURCE_RISK["/api/v1/trusted-guard/sanitized-delete-personnel"] == 1.0
    assert RESOURCE_RISK["/api/v1/documents"] == 0.7
    assert RESOURCE_RISK["/api/v1/personnel"] == 0.6
    assert RESOURCE_RISK["/"] == 0.0  # public route, no risk


def _sim():
    return ZTAStreamSimulator(num_users=10, num_devices=12, num_sources=40,
                              num_resources=len(RESOURCE_URIS), num_wipe_slots=4,
                              num_theft_slots=8, seed=0,
                              use_resource_risk=True, use_source_internal=True)


def test_source_keys_are_namespaced_and_disjoint_from_devices():
    sim = _sim()
    src_keys = [sim.keys[sim.src_lo + s] for s in range(sim.src_slots)]
    dev_keys = [sim.keys[sim.dev_lo + m] for m in range(sim.dev_slots)]
    # Sources carry the src: namespace; devices use tpm:/ck: (or spare slots).
    assert all(isinstance(k, str) and k.startswith("src:") for k in src_keys)
    # No bare IP can alias a device slot: the keyspaces are disjoint.
    assert set(src_keys).isdisjoint(dev_keys)


def test_resource_risk_and_source_internal_baked_into_node_features():
    sim = _sim()
    nf = sim.node_features
    # node_feat[res, 4] mirrors the risk map for every preregistered resource. Resolved
    # against the simulator's own per-seed catalogue (see build_resource_universe).
    for r in range(len(sim.resource_uris)):
        assert nf[sim.res_lo + r, 4].item() == approx(sim.resource_risk[sim.resource_uris[r]])
    # node_feat[src, 5] is the internal bit derived from the (namespaced) source IP.
    for s in range(sim.src_slots):
        key = sim.keys[sim.src_lo + s]
        assert nf[sim.src_lo + s, 5].item() == (1.0 if ip_is_internal(key) else 0.0)
