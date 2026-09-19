"""Unit tests for the causal baseline signals in ``graphagate.eval_common``.

    pytest tests/test_eval_common.py
"""

import numpy as np
from pytest import approx

from graphagate.eval_common import causal_hist_features


def _stream():
    # One (src, dst) pair repeated 6 times; events 2..5 are attacks.
    src = np.zeros(6, dtype=int)
    dst = np.zeros(6, dtype=int)
    y = np.array([0, 0, 1, 1, 1, 1])
    return src, dst, y


def test_default_gate_walks_every_label():
    # Legacy behaviour (synthetic baselines): attacks never advance the counters.
    src, dst, y = _stream()
    pair_count = np.expm1(causal_hist_features(src, dst, y)[:, 0])
    assert pair_count == approx([0, 1, 2, 2, 2, 2])


def test_label_horizon_commits_everything_after_it():
    # Labels exist only up to index 3; from there every event is committed, so a repeated
    # attack pair becomes familiar instead of staying "never seen".
    src, dst, y = _stream()
    pair_count = np.expm1(causal_hist_features(src, dst, y, label_horizon=3)[:, 0])
    assert pair_count == approx([0, 1, 2, 2, 3, 4])


def test_features_past_horizon_ignore_test_labels():
    # The invariant that matters: past the horizon no label reaches the features.
    src, dst, y = _stream()
    y_flipped = y.copy()
    y_flipped[3:] = 0
    a = causal_hist_features(src, dst, y, label_horizon=3)
    b = causal_hist_features(src, dst, y_flipped, label_horizon=3)
    assert np.array_equal(a, b)
