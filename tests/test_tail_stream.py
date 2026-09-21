"""``tail_stream`` cuts the training budget without moving the validation / test windows.

    pytest tests/test_tail_stream.py

The data-budget curve is only meaningful if every budget is scored on the very same
validation and test events; these tests pin that, including the ``int(n * frac)``
rounding that decides where the split lands.
"""

import dataclasses

import numpy as np
import pytest
import torch

from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg
from graphagate.eval_common import _PER_EVENT, tail_stream

# Written out on purpose: the tests must not derive their expectation from the module
# under test (dropping a field from ``_PER_EVENT`` would otherwise silence its own check).
PER_EVENT = ("source", "config", "device", "user", "dst", "t", "msg", "y", "types", "scenario")


def _split(n, cfg):
    train_end = int(n * cfg.train_frac)
    return train_end, train_end + int(n * cfg.val_frac)


@pytest.fixture(scope="module")
def full():
    cfg = dataclasses.replace(TGNConfig(), num_events=30000, seed=2000)
    return generate_streaming_data(**stream_kwargs_from_cfg(cfg)), cfg


@pytest.mark.parametrize("n_train", [1, 500, 3000, 12345, 21000])
def test_validation_and_test_windows_are_bit_identical(full, n_train):
    s, cfg = full
    s2, cfg2 = tail_stream(s, cfg, n_train)

    train_end, val_end = _split(len(s.y), cfg)
    tr2, val2 = _split(len(s2.y), cfg2)

    assert tr2 == n_train                                    # the budget lands exactly
    assert val2 - tr2 == val_end - train_end                 # same validation length
    assert len(s2.y) - val2 == len(s.y) - val_end            # same test length
    for name in PER_EVENT:
        assert torch.equal(getattr(s2, name)[tr2:], getattr(s, name)[train_end:]), name
    # ... and the kept training events are the ones right before the validation window.
    assert torch.equal(s2.y[:tr2], s.y[train_end - n_train:train_end])
    assert cfg2.num_events == len(s2.y)


def test_full_budget_is_the_identity(full):
    s, cfg = full
    train_end, _ = _split(len(s.y), cfg)
    s2, cfg2 = tail_stream(s, cfg, train_end)
    assert _split(len(s2.y), cfg2) == _split(len(s.y), cfg)
    for name in PER_EVENT:
        assert torch.equal(getattr(s2, name), getattr(s, name)), name


def test_every_per_event_tensor_of_the_stream_is_cut(full):
    # A per-event tensor added to SyntheticStream but forgotten in _PER_EVENT would keep
    # its full length and silently misalign with the truncated events.
    s, cfg = full
    n = len(s.y)
    per_event = {f.name for f in dataclasses.fields(s)
                 if torch.is_tensor(getattr(s, f.name)) and getattr(s, f.name).shape[:1] == (n,)}
    assert per_event == set(PER_EVENT) == set(_PER_EVENT)
    s2, _ = tail_stream(s, cfg, 3000)
    n2 = len(s2.y)
    assert all(getattr(s2, name).shape[0] == n2 for name in PER_EVENT)


def test_entity_space_is_untouched(full):
    s, cfg = full
    s2, _ = tail_stream(s, cfg, 3000)
    assert s2.node_features is s.node_features
    assert s2.keys is s.keys
    assert s2.num_nodes == s.num_nodes


@pytest.mark.parametrize("bad", [0, -5, 10**9])
def test_out_of_range_budget_is_rejected(full, bad):
    s, cfg = full
    with pytest.raises(ValueError):
        tail_stream(s, cfg, bad)


def test_rounding_never_moves_the_split():
    # Property: for any stream length and budget, int(n' * frac) hits the target exactly.
    @dataclasses.dataclass
    class Stub:
        source: np.ndarray
        config: np.ndarray
        device: np.ndarray
        user: np.ndarray
        dst: np.ndarray
        t: np.ndarray
        msg: np.ndarray
        y: np.ndarray
        types: np.ndarray
        scenario: np.ndarray

    rng = np.random.default_rng(7)
    for _ in range(300):
        n = int(rng.integers(50, 4000))
        cfg = dataclasses.replace(TGNConfig(), train_frac=float(rng.choice([0.6, 0.7, 0.75])),
                                  val_frac=float(rng.choice([0.05, 0.1, 0.15])))
        train_end = int(n * cfg.train_frac)
        n_train = int(rng.integers(1, train_end + 1))
        z = np.zeros(n)
        s = Stub(*(z.copy() for _ in _PER_EVENT))
        s2, cfg2 = tail_stream(s, cfg, n_train)
        tr2, val2 = _split(len(s2.y), cfg2)
        assert tr2 == n_train
        assert val2 - tr2 == int(n * cfg.val_frac)
