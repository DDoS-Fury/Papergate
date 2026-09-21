"""The Isolation Forest baseline must not see test-window ground-truth labels.

    pytest tests/test_isolation_forest_baseline.py

Its history counters are benign-gated by ground truth; without a ``label_horizon`` the
counters of a test event depend on the labels of earlier test events (an oracle that
favours the baseline). The primitive is covered in ``test_eval_common.py``; this file
pins the *wiring* in the baseline, which is where the leak lived.
"""

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import torch

from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg

_PATH = Path(__file__).resolve().parent / "baselines" / "isolation_forest" / "isolation_forest_baseline.py"


def _load():
    spec = importlib.util.spec_from_file_location("isolation_forest_baseline", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


iforest = _load()


def _tensors(n=60, seed=0):
    rng = np.random.default_rng(seed)
    src = torch.as_tensor(rng.integers(0, 4, n))
    dst = torch.as_tensor(rng.integers(4, 8, n))
    msg = torch.as_tensor(rng.normal(size=(n, 10)), dtype=torch.float32)
    node_features = torch.as_tensor(rng.normal(size=(8, 16)), dtype=torch.float32)
    y = torch.as_tensor((rng.random(n) < 0.3).astype(np.int64))
    return msg, src, dst, node_features, y


def test_features_do_not_depend_on_labels_past_the_horizon():
    msg, src, dst, nf, y = _tensors()
    horizon = 40
    y_other = y.clone()
    y_other[horizon:] = 1 - y_other[horizon:]  # any labelling of the "test" window
    a = iforest._build_features(msg, src, dst, nf, y, label_horizon=horizon)
    b = iforest._build_features(msg, src, dst, nf, y_other, label_horizon=horizon)
    assert np.array_equal(a, b)


def test_features_do_depend_on_labels_before_the_horizon():
    # Guard against a vacuous invariance test: train/val labels must still gate the counters.
    msg, src, dst, nf, y = _tensors()
    horizon = 40
    y_other = y.clone()
    y_other[:horizon] = 1 - y_other[:horizon]
    a = iforest._build_features(msg, src, dst, nf, y, label_horizon=horizon)
    b = iforest._build_features(msg, src, dst, nf, y_other, label_horizon=horizon)
    assert not np.array_equal(a, b)


def test_pipeline_passes_val_end_and_uses_the_injected_stream(monkeypatch):
    cfg = dataclasses.replace(TGNConfig(), num_events=20000, seed=2000)
    stream = generate_streaming_data(**stream_kwargs_from_cfg(cfg))

    def _no_regeneration(**_):
        raise AssertionError("an injected stream must not be regenerated")

    monkeypatch.setattr(iforest, "generate_streaming_data", _no_regeneration)

    horizons = []
    real = iforest.causal_hist_features

    def spy(src, dst, y, **kw):
        horizons.append(kw.get("label_horizon"))
        return real(src, dst, y, **kw)

    monkeypatch.setattr(iforest, "causal_hist_features", spy)

    out = iforest.isolation_forest_baseline(cfg, stream=stream)

    n = len(stream.y)
    assert horizons == [int(n * cfg.train_frac) + int(n * cfg.val_frac)]
    assert 0.0 <= out["agg_auc"] <= 1.0
    assert "lateral" in out["per_type"]
