"""The offline replay's commit gate and the state it starts calibration from.

* ``SyntheticStream.policy_allow`` is the reference-PDP verdict of every event: it denies the
  policy-violation and benign-denial classes by construction and allows benign traffic, and
  it is derived from the message and the destination only (no label).
* ``train_tgn`` leaves train mode before it snapshots the pre-calibration state: PyG's
  ``TGNMemory.train(False)`` flushes the message store, and doing it inside the first replay
  would give the calibration pass A a different memory than pass B and the test.

    pytest tests/test_replay_gates.py
"""

import dataclasses

import numpy as np
import pytest
import torch

from graphagate import train_tgn as T
from graphagate.config import TGNConfig
from graphagate.data.access_policy import NUM_BASE_ROUTES, build_resource_universe, policy_allows
from graphagate.data.stream_synthetic import ROLES, generate_streaming_data, stream_kwargs_from_cfg

SMALL = dict(num_events=6000, epochs=1, ft_epochs=1, seed=5)
BENIGN, POLICY, BENIGN_DENIED = 0, 1, 6


def test_stream_policy_allow_is_the_reference_policy():
    cfg = TGNConfig(**SMALL)
    s = generate_streaming_data(**stream_kwargs_from_cfg(cfg))
    routes, matrix, uris, _ = build_resource_universe(cfg.num_resources - NUM_BASE_ROUTES, cfg.seed)
    msg, dst = s.msg.numpy(), s.dst.numpy()
    expect = np.array([
        policy_allows(ROLES[int(round(msg[i, 5] * (len(ROLES) - 1)))], int(msg[i, 4]),
                      uris[dst[i] - s.res_lo], routes, matrix)
        for i in range(len(dst))
    ])
    allow, types = s.policy_allow.numpy(), s.types.numpy()
    assert (allow == expect).all()
    assert allow[types == BENIGN].all()
    assert not allow[types == POLICY].any() and (types == POLICY).any()
    assert not allow[types == BENIGN_DENIED].any() and (types == BENIGN_DENIED).any()


class _Stop(Exception):
    pass


def test_calibration_starts_from_the_flushed_memory(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)  # pdist: no MPS kernel

    def first_replay(model, *args, **kwargs):
        mem = model.memory
        seen["training"] = model.training
        seen["pending"] = sum(int(v[0].numel()) for v in (*mem.msg_s_store.values(), *mem.msg_d_store.values()))
        raise _Stop

    monkeypatch.setattr(T, "_replay", first_replay)
    with pytest.raises(_Stop):
        T.train_tgn(TGNConfig(**SMALL), save=False, log_dir=str(tmp_path))
    assert seen["training"] is False
    assert seen["pending"] == 0   # nothing left in the store: the last training batch is applied
