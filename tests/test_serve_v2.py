"""Unit tests for the v2 serving path (4 nodes, 3 edges) — no trained checkpoint needed.

Covers:
  * the schema_version gate (v1 checkpoints are rejected with a clear error);
  * the full source→device→user→resource chain: key admission, per-edge commits and
    the auxiliary (device, resource) habituality counter;
  * the ``key_source=None`` fallback (no client IP): the source→device edge is
    skipped and no source node is admitted.

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
        "schema_version": 2,
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
    }
    hp.update(overrides)
    return hp


def _fresh():
    torch.manual_seed(0)
    model = build_model(_hp(), DEVICE)
    return model, NodeRegistry(capacity=64)


def test_schema_version_gate():
    with pytest.raises(RuntimeError, match="schema_version"):
        build_model(_hp(schema_version=1), DEVICE)
    # A v1 checkpoint without the field at all must also be rejected.
    hp = _hp()
    del hp["schema_version"]
    with pytest.raises(RuntimeError, match="schema_version"):
        build_model(hp, DEVICE)


def test_three_edge_chain_commits():
    model, reg = _fresh()
    score, is_anom = score_event(
        model, reg, 2.0, "alice", "tpm:0001", "/api/v1/documents", 100, FEAT, DEVICE,
        key_source="10.0.0.7", update=True,
    )
    assert 0.0 <= score <= 1.0 and not is_anom
    keys = ["alice", "tpm:0001", "/api/v1/documents", "10.0.0.7"]
    ui, di, ri, si = (reg.get(k) for k in keys)
    assert None not in (ui, di, ri, si)
    assert model.pair_count[(si, di)] == 1   # source→device binding
    assert model.pair_count[(di, ui)] == 1   # device→user binding
    assert model.pair_count[(ui, ri)] == 1   # user→resource access
    assert model.pair_count[(di, ri)] == 1   # aux (device, resource) counter — no edge
    # The aux counter must not inflate the device's activity count (one bump per
    # event, from its binding-edge commit).
    assert model.src_count[di] == 1
    assert int(model.memory.last_update[si]) == 100


def test_key_source_none_skips_source_edge():
    model, reg = _fresh()
    score_event(model, reg, 2.0, "alice", "tpm:0001", "/api/v1/documents", 100, FEAT,
                DEVICE, key_source="10.0.0.7", update=True)
    si = reg.get("10.0.0.7")
    di = reg.get("tpm:0001")
    n_before = len(reg)

    score, _ = score_event(
        model, reg, 2.0, "alice", "tpm:0001", "/api/v1/documents", 200, FEAT, DEVICE,
        update=True,  # no key_source
    )
    assert 0.0 <= score <= 1.0
    assert len(reg) == n_before                       # nothing new admitted
    assert model.pair_count[(si, di)] == 1            # source edge NOT advanced
    assert model.pair_count[(reg.get("alice"), reg.get("/api/v1/documents"))] == 2


def test_commit_event_full_chain():
    model, reg = _fresh()
    commit_event(model, reg, "bob", "ck:0042-g0", "/api/v1/personnel", 50, FEAT, DEVICE,
                 key_source="100.64.0.9")
    ui, di, ri, si = (reg.get(k) for k in
                      ["bob", "ck:0042-g0", "/api/v1/personnel", "100.64.0.9"])
    assert model.pair_count[(si, di)] == 1
    assert model.pair_count[(di, ui)] == 1
    assert model.pair_count[(ui, ri)] == 1
    assert model.pair_count[(di, ri)] == 1
    # Without the IP the chain still commits its two remaining edges.
    commit_event(model, reg, "bob", "ck:0042-g0", "/api/v1/personnel", 60, FEAT, DEVICE)
    assert model.pair_count[(si, di)] == 1
    assert model.pair_count[(di, ui)] == 2
    assert model.pair_count[(ui, ri)] == 2
