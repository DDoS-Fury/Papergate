"""The stateful-rules baseline scores like the TGN: same window, no test-label oracle.

    pytest tests/test_lookup_rules_baseline.py
"""

import dataclasses
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from graphagate.config import TGNConfig
from graphagate.data.lookup_rules import lookup_flags
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg

_PATH = Path(__file__).resolve().parent / "baselines" / "lookup_rules" / "lookup_rules_baseline.py"


def _load():
    spec = importlib.util.spec_from_file_location("lookup_rules_baseline", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rules = _load()

CFG = dataclasses.replace(TGNConfig(), num_events=60000, seed=2000)


@pytest.fixture(scope="module")
def stream():
    return generate_streaming_data(**stream_kwargs_from_cfg(CFG))


def _val_end(cfg, n):
    return int(n * cfg.train_frac) + int(n * cfg.val_frac)


def test_operating_point_is_calibrated_on_benign_validation(monkeypatch):
    # n=1000 -> train 0..699, val 700..799 (all benign), test 800..999.
    n, cfg = 1000, TGNConfig()
    types = np.zeros(n, dtype=np.int64)
    types[800:810] = 3                                   # 10 lateral in the test window
    score = np.zeros(n)
    score[700:702] = 2                                   # val benign: 2 events at 2 ...
    score[702:707] = 1                                   # ... 5 at 1  -> FPR(k=1)=7%, k=2: 2%, k=3: 0
    score[800:810] = [3, 3, 3, 2, 2, 1, 0, 0, 0, 0]      # lateral scores
    score[810] = 3                                       # one benign test event fires at k=3
    fake = SimpleNamespace(y=torch.as_tensor((types > 0).astype(np.int64)),
                           types=torch.as_tensor(types))
    monkeypatch.setattr(rules, "lookup_flags", lambda s, gate, test_start: {"stateful": score})

    out = rules.lookup_rules_baseline(cfg, stream=fake)["per_type"]

    assert set(out) == {"lateral"}                       # no cred-theft events in the window
    lat = out["lateral"]
    assert lat["threshold"] == 3                         # k=1 and k=2 exceed the 1% validation FPR
    assert lat["recall"] == pytest.approx(3 / 10)        # scores >= 3
    assert lat["fpr"] == pytest.approx(1 / 190)          # one of the 190 benign test events
    assert lat["n"] == 10


@pytest.mark.parametrize("fn, horizon", [("lookup_rules_baseline", "train"),
                                         ("lookup_rules_val_baseline", "val")])
def test_auc_matches_an_independent_recomputation(stream, fn, horizon):
    out = getattr(rules, fn)(CFG, stream=stream)["per_type"]
    n = len(stream.y)
    train_end = int(n * CFG.train_frac)
    val_end = _val_end(CFG, n)
    flags = lookup_flags(stream, "proto-self", train_end if horizon == "train" else val_end)["stateful"]
    flags = flags.astype(float)
    ty = stream.types.numpy()
    te = np.arange(n) >= val_end                         # the test window is the same for both
    for type_id, name in ((3, "lateral"), (4, "cred-theft")):
        sel = te & ((ty == 0) | (ty == type_id))
        expected = roc_auc_score((ty[sel] == type_id).astype(int), flags[sel])
        assert out[name]["auc"] == pytest.approx(expected)
        assert out[name]["n"] == int((te & (ty == type_id)).sum())


@pytest.mark.parametrize("fn, expected", [("lookup_rules_baseline", "train"),
                                          ("lookup_rules_val_baseline", "val")])
def test_label_horizon_of_each_variant(stream, monkeypatch, fn, expected):
    # The primary rules row must not see validation labels (the TGN's val/test replays are
    # self-gated); the sensitivity row does. Pin the horizon actually handed to the rules.
    n = len(stream.y)
    want = int(n * CFG.train_frac) if expected == "train" else _val_end(CFG, n)
    seen = []
    real = rules.lookup_flags
    monkeypatch.setattr(rules, "lookup_flags",
                        lambda s, gate, test_start: seen.append((gate, test_start)) or real(s, gate, test_start))
    getattr(rules, fn)(CFG, stream=stream)
    assert seen == [("proto-self", want)]


def test_unknown_label_horizon_is_rejected(stream):
    with pytest.raises(ValueError):
        rules.lookup_rules_baseline(CFG, stream=stream, labels_through="test")


def test_random_scores_are_chance(stream, monkeypatch):
    # The metric path must depend on the scores only: random integers -> AUC near 0.5.
    rng = np.random.default_rng(0)
    rnd = rng.integers(0, 6, len(stream.y)).astype(float)
    monkeypatch.setattr(rules, "lookup_flags", lambda s, gate, test_start: {"stateful": rnd})
    out = rules.lookup_rules_baseline(CFG, stream=stream)["per_type"]
    for name in ("lateral", "cred-theft"):
        assert abs(out[name]["auc"] - 0.5) < 0.12


@pytest.mark.parametrize("gate", ["proto-self", "all"])
def test_scores_ignore_test_window_labels(stream, gate):
    # Past the test start no ground-truth label may reach the rule state.
    n = len(stream.y)
    val_end = _val_end(CFG, n)
    y2 = stream.y.clone()
    y2[val_end:] = 1 - y2[val_end:]
    a = lookup_flags(stream, gate, val_end)["stateful"]
    b = lookup_flags(dataclasses.replace(stream, y=y2), gate, val_end)["stateful"]
    assert np.array_equal(a, b)


def test_all_gate_uses_no_label_at_all(stream):
    n = len(stream.y)
    val_end = _val_end(CFG, n)
    a = lookup_flags(stream, "all", val_end)["stateful"]
    b = lookup_flags(dataclasses.replace(stream, y=1 - stream.y), "all", val_end)["stateful"]
    assert np.array_equal(a, b)


def test_schema_is_what_the_report_consumes(stream):
    out = rules.lookup_rules_all_baseline(CFG, stream=stream)
    assert set(out) == {"per_type"}                      # no aggregate for the rules
    for entry in out["per_type"].values():
        assert set(entry) == {"auc", "ap", "recall", "fpr", "threshold", "n"}
