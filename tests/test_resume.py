"""Unit tests for the per-epoch resume state — no training run needed.

A full training run is hours long and the deployable artifact is only written at the
end, so an interruption (thermal sleep, OOM kill, power loss) must cost at most one
epoch rather than the whole run. These tests cover the state file that makes that true:

  * the round trip: weights, optimizer moments and RNG states all come back;
  * the fingerprint gate: a resume file from a *different* configuration is refused,
    because restoring it would silently blend two runs into one set of numbers;
  * corruption tolerance: a file truncated by the very crash it exists to survive
    costs a restart, never the ability to run at all.

    pytest tests/test_resume.py
"""

import numpy as np
import pytest
import torch
from torch.optim import AdamW

from graphagate.config import TGNConfig
from graphagate.train_tgn import _load_resume, _resume_fingerprint, _save_resume

DEVICE = torch.device("cpu")
FLAGS = {"use_struct_head": True, "dataset": "synthetic"}


def _model_and_optimizer(seed=0):
    torch.manual_seed(seed)
    model = torch.nn.Linear(4, 2)
    return model, AdamW(model.parameters(), lr=1e-3)


def _step(model, optimizer):
    """One optimizer step, so the saved moments are non-trivial."""
    optimizer.zero_grad()
    model(torch.ones(1, 4)).sum().backward()
    optimizer.step()


def test_round_trip_restores_weights_optimizer_and_rng(tmp_path):
    path = tmp_path / "resume.pt"
    cfg = TGNConfig(num_events=1000, epochs=4)
    fp = _resume_fingerprint(cfg, FLAGS)

    model, optimizer = _model_and_optimizer()
    _step(model, optimizer)
    torch.manual_seed(1234)
    np.random.seed(1234)
    _save_resume(path, model, optimizer, epoch=2, fingerprint=fp)
    # The next random draws of the interrupted run, to compare against.
    expected_torch = torch.randn(3)
    expected_numpy = np.random.rand(3)

    restored, restored_opt = _model_and_optimizer(seed=99)  # deliberately different
    assert _load_resume(path, restored, restored_opt, fp, DEVICE) == 2

    for a, b in zip(model.parameters(), restored.parameters()):
        assert torch.equal(a, b)
    # AdamW's moments must survive too: resuming with a fresh optimizer would restart
    # the momentum estimates and change the trajectory.
    saved_state = optimizer.state_dict()["state"]
    assert restored_opt.state_dict()["state"].keys() == saved_state.keys()
    for k, v in saved_state.items():
        assert torch.equal(restored_opt.state_dict()["state"][k]["exp_avg"], v["exp_avg"])

    # A resumed run must continue the same random trajectory as an uninterrupted one.
    assert torch.equal(torch.randn(3), expected_torch)
    assert np.allclose(np.random.rand(3), expected_numpy)


def test_missing_file_starts_from_scratch(tmp_path):
    model, optimizer = _model_and_optimizer()
    fp = _resume_fingerprint(TGNConfig(), FLAGS)
    assert _load_resume(tmp_path / "absent.pt", model, optimizer, fp, DEVICE) == 0


@pytest.mark.parametrize("other", [
    {"cfg": TGNConfig(num_events=1000, epochs=4, learning_rate=5e-4), "flags": FLAGS},
    {"cfg": TGNConfig(num_events=1000, epochs=4), "flags": {**FLAGS, "use_struct_head": False}},
    {"cfg": TGNConfig(num_events=1000, epochs=4), "flags": {**FLAGS, "dataset": "injected"}},
])
def test_foreign_configuration_is_refused(tmp_path, other):
    """Weights from another configuration must never be silently adopted."""
    path = tmp_path / "resume.pt"
    model, optimizer = _model_and_optimizer()
    _save_resume(path, model, optimizer, epoch=3,
                 fingerprint=_resume_fingerprint(TGNConfig(num_events=1000, epochs=4), FLAGS))

    guard, guard_opt = _model_and_optimizer(seed=99)
    before = [p.clone() for p in guard.parameters()]
    assert _load_resume(path, guard, guard_opt,
                        _resume_fingerprint(other["cfg"], other["flags"]), DEVICE) == 0
    for p, b in zip(guard.parameters(), before):
        assert torch.equal(p, b), "refused resume must leave the model untouched"


def test_truncated_file_starts_from_scratch(tmp_path):
    """A crash mid-write leaves a partial file; it must not abort the next run."""
    path = tmp_path / "resume.pt"
    model, optimizer = _model_and_optimizer()
    fp = _resume_fingerprint(TGNConfig(), FLAGS)
    _save_resume(path, model, optimizer, epoch=1, fingerprint=fp)
    data = path.read_bytes()
    path.write_bytes(data[:len(data) // 2])

    assert _load_resume(path, model, optimizer, fp, DEVICE) == 0


def test_save_is_atomic(tmp_path):
    """The real file appears whole or not at all, and no .tmp is left behind."""
    path = tmp_path / "resume.pt"
    model, optimizer = _model_and_optimizer()
    _save_resume(path, model, optimizer, epoch=1,
                 fingerprint=_resume_fingerprint(TGNConfig(), FLAGS))
    assert path.exists()
    assert list(tmp_path.glob("*.tmp")) == []
