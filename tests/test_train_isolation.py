"""The training loop reads nothing that belongs to validation or test.

The stream holds, in one node-id space, the entities of the whole run: also the ones that first
appear after the training window (re-keyed cookies, new JA3s, a thief's device, new hires), with
their final key and static features. A training step that touches one of them (as a negative,
for instance) trains its identity embedding before the entity exists, which no deployment can do.
"""

import numpy as np
import pytest
import torch

from graphagate import train_tgn as T
from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg
from graphagate.model.tgn import ZTATemporalGraphNetwork

SMALL = dict(num_events=6000, epochs=1, ft_epochs=1, seed=5)


class _TrainingDone(Exception):
    pass


def _nodes_read_in_training(monkeypatch, tmp_path, **overrides):
    """Run the gradient loop of ``train_tgn`` (both stages) and return every node id it embedded
    plus the stream; the run is cut at the first calibration replay."""
    read = []
    embed = ZTATemporalGraphNetwork.embed

    def spy(self, n_id, *args, **kwargs):
        if self.training:
            read.append(n_id.detach().cpu())
        return embed(self, n_id, *args, **kwargs)

    def stop(*args, **kwargs):
        raise _TrainingDone

    monkeypatch.setattr(ZTATemporalGraphNetwork, "embed", spy)
    monkeypatch.setattr(T, "_replay", stop)
    cfg = TGNConfig(**SMALL, **overrides)
    with pytest.raises(_TrainingDone):
        T.train_tgn(cfg, save=False, log_dir=str(tmp_path))
    stream = generate_streaming_data(**stream_kwargs_from_cfg(cfg))
    return cfg, stream, torch.cat(read).unique().numpy()


def _train_benign_nodes(cfg, s):
    train_end = int(len(s.dst) * cfg.train_frac)
    benign = (s.y[:train_end] == 0).numpy()
    return np.unique(np.concatenate([
        role[:train_end].numpy()[benign] for role in (s.source, s.config, s.device, s.user, s.dst)
    ]))


def test_training_reads_only_entities_of_the_benign_training_traffic(monkeypatch, tmp_path):
    cfg, s, read = _nodes_read_in_training(monkeypatch, tmp_path)
    allowed = _train_benign_nodes(cfg, s)
    outside = np.setdiff1d(read, allowed)
    assert outside.size == 0, (
        f"{outside.size} nodes read by the training loop are not in the benign training traffic "
        f"(e.g. {[s.keys[i] for i in outside[:5]]})"
    )


def test_the_check_sees_range_wide_negatives(monkeypatch, tmp_path):
    """Guard on the test above: with the pools off, the loop does read later entities."""
    cfg, s, read = _nodes_read_in_training(monkeypatch, tmp_path, negatives_from_train_only=False)
    later = np.setdiff1d(
        np.concatenate([r[int(len(s.dst) * cfg.train_frac):].numpy()
                        for r in (s.config, s.device, s.user)]),
        _train_benign_nodes(cfg, s),
    )
    assert np.intersect1d(read, later).size > 0


def test_negative_pool_holds_benign_training_entities_only():
    nodes = torch.tensor([3, 4, 5, 6, 7, 8])
    y = torch.tensor([0, 1, 0, 0, 0, 0])
    assert T.train_negative_pool(nodes, y, train_end=4).tolist() == [3, 5, 6]
    assert T.train_negative_pool(nodes, y, train_end=4, enabled=False) is None
    assert T.train_negative_pool(None, y, train_end=4) is None


def test_sampler_draws_inside_the_pool():
    torch.manual_seed(0)
    pool = torch.tensor([10, 12, 19])
    avoid = torch.full((600,), 12)
    hard = torch.tensor([10, 19])
    neg = T._sample_structural_negatives(600, 50, 0, "cpu", avoid=avoid, hard_pool=hard,
                                         hard_ratio=0.25, pool=pool)
    assert set(neg.tolist()) <= set(pool.tolist())


def test_recycled_fresh_pool_is_an_error():
    cfg = TGNConfig(num_events=6000, seed=5, num_new_sources=8)
    with pytest.raises(RuntimeError, match="num_new_sources"):
        generate_streaming_data(**stream_kwargs_from_cfg(cfg))
