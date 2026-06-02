"""Central configuration: hyper-parameters and paths for the streaming TGN.

The streaming Temporal Graph Network (v2) consumes synthetic ZTA access events
and scores each interaction online. Its memory is sized for ``capacity`` slots so
previously unseen entities can be admitted at inference time via the
:class:`~graphagate.model.registry.NodeRegistry`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Repository root = parent of ``src/`` (this file lives at src/config.py).
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR: Path = REPO_ROOT / "public"

# Streaming TGN deployment artifacts.
TGN_CHECKPOINT_PATH: Path = ARTIFACTS_DIR / "tgn_checkpoint.pt"
TGN_STATS_PATH: Path = ARTIFACTS_DIR / "tgn_stats.json"


@dataclass(frozen=True)
class TGNConfig:
    """Streaming Temporal Graph Network hyper-parameters.

    The model memory is sized for ``capacity`` slots (= training entities +
    ``capacity_headroom``) so previously unseen ZTA entities can be admitted at
    inference time through the :class:`~graphagate.model.registry.NodeRegistry`.
    """

    # Synthetic stream shape (entity counts + number of events).
    num_users: int = 50
    num_ips: int = 100
    num_resources: int = 20
    num_events: int = 50000

    msg_dim: int = 6
    memory_dim: int = 256
    time_dim: int = 32
    node_feat_dim: int = 16
    hash_buckets: int = 10000
    hash_dim: int = 16
    num_hops: int = 3
    # Temporal neighbours kept per node in the (bounded, in-memory) neighbour loader.
    # Enables message passing over each entity's recent interaction history → the
    # structural signal for lateral-movement detection. No graph DB required.
    neighbor_size: int = 30

    # Optimisation.
    batch_size: int = 200
    epochs: int = 15
    learning_rate: float = 1e-3

    # Chronological split fractions (test = 1 - train - val).
    train_frac: float = 0.7
    val_frac: float = 0.1

    # Spare memory slots reserved for entities first seen at inference time.
    capacity_headroom: int = 50000
    # Decision-threshold calibration target (benign false-positive rate).
    target_fpr: float = 0.01

    seed: int = 42

    @property
    def total_nodes(self) -> int:
        return self.num_users + self.num_ips + self.num_resources

    @property
    def capacity(self) -> int:
        return self.total_nodes + self.capacity_headroom
