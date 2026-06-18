"""Unit tests for the v4 serving path (5 nodes, up to 5 edges) — no trained checkpoint needed.

Covers:
  * the schema_version gate (v1/v2/v3 checkpoints are rejected with a clear error);
  * the full source→config→device→user→resource chain plus the config→user binding:
    key admission, per-edge commits and the auxiliary (device, resource) habituality
    counter;
  * the ``key_config`` default (``conf:guest`` when omitted);
  * the ``key_source=None`` fallback (no client IP): the source→config edge is skipped
    and no source node is admitted.

Run inside the project's Docker image (torch required):

    pytest tests/test_serve_v2.py
"""

import pytest
import torch

from graphagate.model.registry import NodeRegistry
from graphagate.serve_tgn import build_model, commit_event, score_event

DEVICE = torch.device("cpu")
FEAT = [1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5]


def _hp(**overrides):
    hp = {
        "schema_version": 4,
        "capacity": 64,
        "node_feat_dim": 16,
        "msg_dim": 7,
        "memory_dim": 32,
        "time_dim": 8,
        "num_hops": 2,
        "hash_buckets": 100,
        "hash_dim": 8,
        "hist_feat_dim": 6,
        "neighbor_size": 5,
        "use_source_internal": True,  # exercise the source-internal feature path
    }
    hp.update(overrides)
    return hp


def _fresh():
    torch.manual_seed(0)
    model = build_model(_hp(), DEVICE)
    return model, NodeRegistry(capacity=64)


def test_schema_version_gate():
    # Older schemas (v1 2-edge, v2 bare-IP keys, v3 4-node) are all incompatible.
    for stale in (1, 2, 3):
        with pytest.raises(RuntimeError, match="schema_version"):
            build_model(_hp(schema_version=stale), DEVICE)
    # A checkpoint without the field at all must also be rejected.
    hp = _hp()
    del hp["schema_version"]
    with pytest.raises(RuntimeError, match="schema_version"):
        build_model(hp, DEVICE)


def test_five_edge_chain_commits():
    model, reg = _fresh()
    score, is_anom = score_event(
        model, reg, 2.0, "alice", "tpm:0001", "/api/v1/documents", 100, FEAT, DEVICE,
        key_source="src:10.0.0.7", key_config="conf:0001", update=True,
    )
    assert 0.0 <= score <= 1.0 and not is_anom
    keys = ["alice", "tpm:0001", "/api/v1/documents", "src:10.0.0.7", "conf:0001"]
    ui, di, ri, si, ci = (reg.get(k) for k in keys)
    assert None not in (ui, di, ri, si, ci)
    assert model.pair_count[(si, ci)] == 1   # source→config binding
    assert model.pair_count[(ci, di)] == 1   # config→device binding
    assert model.pair_count[(ci, ui)] == 1   # config→user binding
    assert model.pair_count[(di, ui)] == 1   # device→user binding
    assert model.pair_count[(ui, ri)] == 1   # user→resource access
    assert model.pair_count[(di, ri)] == 1   # aux (device, resource) counter — no edge
    # The aux counter must not inflate the device's activity count (one bump per
    # event, from its binding-edge commit).
    assert model.src_count[di] == 1
    # config is the src of two edges (config→device, config→user).
    assert model.src_count[ci] == 2
    assert int(model.memory.last_update[si]) == 100
    # src:10.0.0.7 is RFC1918 → the source node's internal bit (node_feat[*,5]) is set.
    assert model.node_feat[si, 5].item() == 1.0


def test_key_config_defaults_to_guest():
    model, reg = _fresh()
    score_event(model, reg, 2.0, "alice", "tpm:0001", "/api/v1/documents", 100, FEAT,
                DEVICE, key_source="src:10.0.0.7", update=True)  # no key_config
    assert reg.get("conf:guest") is not None
    ci = reg.get("conf:guest")
    ui = reg.get("alice")
    assert model.pair_count[(ci, ui)] == 1   # the guest config still binds to the user


def test_guest_device_fallback_collapses_non_tpm():
    # Experiment: with guest_device_fallback, every non-TPM device (ck:/ipdev:/None)
    # collapses onto the single shared dev:guest node, mirroring conf:guest. A TPM key
    # keeps its own identity.
    model, reg = _fresh()
    score_event(model, reg, 2.0, "alice", "ck:0001-g0", "/api/v1/documents", 100, FEAT,
                DEVICE, key_source="src:10.0.0.7", update=True,
                guest_device_fallback=True)
    score_event(model, reg, 2.0, "bob", "ck:0002-g0", "/api/v1/documents", 110, FEAT,
                DEVICE, key_source="src:10.0.0.8", update=True,
                guest_device_fallback=True)
    score_event(model, reg, 2.0, "carol", None, "/api/v1/documents", 120, FEAT,
                DEVICE, key_source="src:10.0.0.9", update=True,
                guest_device_fallback=True)
    # Distinct cookies and the missing-device case all map to the one guest node.
    assert reg.get("dev:guest") is not None
    assert reg.get("ck:0001-g0") is None and reg.get("ck:0002-g0") is None
    gi = reg.get("dev:guest")
    # device→user binding fired for each of the three users through the same guest node.
    assert model.src_count[gi] == 3
    # A TPM device is never collapsed: it keeps a separate slot.
    score_event(model, reg, 2.0, "dave", "tpm:0009", "/api/v1/documents", 130, FEAT,
                DEVICE, key_source="src:10.0.0.10", update=True,
                guest_device_fallback=True)
    assert reg.get("tpm:0009") is not None and reg.get("tpm:0009") != gi


def test_source_internal_external_feature():
    model, reg = _fresh()
    # Internal RFC1918 source.
    score_event(model, reg, 2.0, "alice", "tpm:0001", "/api/v1/documents", 100, FEAT,
                DEVICE, key_source="src:192.168.1.20", update=True)
    assert model.node_feat[reg.get("src:192.168.1.20"), 5].item() == 1.0
    # CGNAT 100.64/10 (roaming / 5G) and public TEST-NET are external.
    score_event(model, reg, 2.0, "bob", "ck:0001-g0", "/api/v1/personnel", 110, FEAT,
                DEVICE, key_source="src:100.64.0.9", update=True)
    assert model.node_feat[reg.get("src:100.64.0.9"), 5].item() == 0.0
    score_event(model, reg, 2.0, "carol", "ck:0002-g0", "/api/v1/personnel", 120, FEAT,
                DEVICE, key_source="src:203.0.113.4", update=True)
    assert model.node_feat[reg.get("src:203.0.113.4"), 5].item() == 0.0


def test_key_source_none_skips_source_edge():
    model, reg = _fresh()
    score_event(model, reg, 2.0, "alice", "tpm:0001", "/api/v1/documents", 100, FEAT,
                DEVICE, key_source="src:10.0.0.7", key_config="conf:0001", update=True)
    si = reg.get("src:10.0.0.7")
    ci = reg.get("conf:0001")
    n_before = len(reg)

    score, _ = score_event(
        model, reg, 2.0, "alice", "tpm:0001", "/api/v1/documents", 200, FEAT, DEVICE,
        key_config="conf:0001", update=True,  # no key_source
    )
    assert 0.0 <= score <= 1.0
    assert len(reg) == n_before                       # nothing new admitted
    assert model.pair_count[(si, ci)] == 1            # source→config edge NOT advanced
    assert model.pair_count[(reg.get("alice"), reg.get("/api/v1/documents"))] == 2


def test_commit_event_full_chain():
    model, reg = _fresh()
    commit_event(model, reg, "bob", "ck:0042-g0", "/api/v1/personnel", 50, FEAT, DEVICE,
                 key_source="src:100.64.0.9", key_config="conf:0007")
    ui, di, ri, si, ci = (reg.get(k) for k in
                          ["bob", "ck:0042-g0", "/api/v1/personnel",
                           "src:100.64.0.9", "conf:0007"])
    assert model.pair_count[(si, ci)] == 1
    assert model.pair_count[(ci, di)] == 1
    assert model.pair_count[(ci, ui)] == 1
    assert model.pair_count[(di, ui)] == 1
    assert model.pair_count[(ui, ri)] == 1
    assert model.pair_count[(di, ri)] == 1
    # Without the IP the chain still commits its remaining edges (source→config skipped).
    commit_event(model, reg, "bob", "ck:0042-g0", "/api/v1/personnel", 60, FEAT, DEVICE,
                 key_config="conf:0007")
    assert model.pair_count[(si, ci)] == 1
    assert model.pair_count[(ci, di)] == 2
    assert model.pair_count[(di, ui)] == 2
    assert model.pair_count[(ui, ri)] == 2
