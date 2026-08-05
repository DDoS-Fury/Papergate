"""Contract tests for the PicoDomain → StreamData mapping.

The training pipeline cannot run on a MacBook (PyTorch's ``scatter_reduce_`` rejects
int64 on MPS/CPU, so even the synthetic path fails locally — see ``docs/docker.md``),
which means a broken adapter would only surface on the GPU box. These tests check the
part that can be checked anywhere: that the tensors satisfy the invariants
``train_tgn`` relies on. In particular the per-group index ranges, because the
binding-edge negative samplers draw from ``[*_lo, *_lo + *_num)`` and an off-by-one
there silently trains against negatives from the wrong entity type.

Skipped unless the dataset is present. To enable:

    git clone --depth 1 https://github.com/iHeartGraph/PicoDomain.git data/pico
    7zz x -odata/logs data/pico/Zeek_Logs.7z
    PICO_LOGS=data/logs PICO_REDLOG="data/pico/Red Log.xlsx" pytest tests/test_picodomain_mapping.py
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LOGS = os.environ.get("PICO_LOGS", "data/logs")
REDLOG = os.environ.get("PICO_REDLOG", "data/pico/Red Log.xlsx")

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(LOGS) and os.path.isfile(REDLOG)),
    reason="PicoDomain not present; set PICO_LOGS / PICO_REDLOG (see module docstring)",
)


@pytest.fixture(scope="module")
def stream():
    from datasets.picodomain import load_picodomain_stream
    return load_picodomain_stream(LOGS, REDLOG)


def test_shapes_agree(stream):
    n = stream.t.numel()
    for name in ("user", "dst", "msg", "y", "types", "source_nodes", "config_nodes",
                 "device_nodes"):
        assert getattr(stream, name).shape[0] == n, f"{name} has the wrong length"
    assert stream.msg.shape[1] == 10, "message width must match TGNConfig.msg_dim"
    assert stream.node_features.shape == (stream.num_nodes, 16)
    assert len(stream.keys) == stream.num_nodes


def test_every_index_is_in_range(stream):
    for name in ("user", "dst", "source_nodes", "config_nodes", "device_nodes"):
        idx = getattr(stream, name)
        assert int(idx.min()) >= 0 and int(idx.max()) < stream.num_nodes, name


def test_entity_groups_are_contiguous_and_disjoint(stream):
    """The negative samplers assume one interval per entity type."""
    groups = {
        "user": (stream.user, stream.usr_lo, stream.usr_num),
        "device": (stream.device_nodes, stream.dev_lo, stream.dev_num),
        "config": (stream.config_nodes, stream.cfg_lo, stream.cfg_num),
        "resource": (stream.dst, stream.neg_lo, stream.neg_num),
    }
    for name, (idx, lo, num) in groups.items():
        assert num > 0, f"{name} range is empty"
        assert int(idx.min()) >= lo, f"{name} index below its range"
        assert int(idx.max()) < lo + num, f"{name} index above its range"

    spans = sorted((lo, lo + num, name) for name, (_, lo, num) in groups.items())
    for (_, end_a, name_a), (start_b, _, name_b) in zip(spans, spans[1:]):
        assert end_a <= start_b, f"{name_a} and {name_b} ranges overlap"


def test_stream_is_chronological(stream):
    """Streaming evaluation replays in arrival order; unsorted input would silently
    score events against a memory that has already seen them."""
    assert bool((stream.t[1:] >= stream.t[:-1]).all())


def test_alarm_columns_are_held_clean(stream):
    """PicoDomain ships no IDS stream. Columns 1-3 must stay zero so a rule baseline
    is blind to the red-team activity and the temporal pattern is the discriminator.
    Column 8 (response bytes) is zero for causality: it is not known at decision time.
    """
    for col in (1, 2, 3, 8):
        assert float(stream.msg[:, col].abs().max()) == 0.0, f"msg column {col} is not clean"
    assert float(stream.msg[:, 0].min()) == 1.0, "ja3-valid column must be constant"


def test_labels_cover_the_classes_the_paper_relies_on(stream):
    """Lateral movement and credential reuse must both be present, or the corpus does
    not support the claims it is cited for."""
    counts = {int(c): int((stream.types == c).sum()) for c in stream.types.unique()}
    assert counts.get(3, 0) > 0, "no lateral-movement events labelled"
    assert counts.get(4, 0) > 0, "no credential-reuse events labelled"
    assert (stream.y == (stream.types != 0).long()).all(), "y disagrees with types"
    frac = float(stream.y.float().mean())
    assert 0.0 < frac < 0.10, f"implausible anomalous fraction {frac:.4f}"


def test_unattributed_events_do_not_share_a_node(stream):
    """A shared 'unknown' node would let an unattributed event inherit another
    entity's memory. Fallbacks must be per source address."""
    sentinels = [k for k in stream.keys if ":none:" in k]
    assert sentinels, "expected per-IP sentinel nodes"
    assert len(set(sentinels)) == len(sentinels)
    for prefix in ("usr:", "dev:", "cfg:"):
        assert sum(k.startswith(prefix) for k in sentinels) > 1, (
            f"{prefix} fallback collapsed to a single shared node"
        )


def test_trust_slot_defaults_to_one(stream):
    """Slot 14 at 0.0 would make every entity maximally suspicious on first sight."""
    assert bool((stream.node_features[:, 14] == 1.0).all())


def test_source_internal_flag_is_derived_from_the_address(stream):
    for i, key in enumerate(stream.keys):
        if key.startswith("src:") and key[4:].startswith("10."):
            assert float(stream.node_features[i, 5]) == 1.0, f"{key} not marked internal"
