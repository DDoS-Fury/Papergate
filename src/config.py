"""Central configuration: hyper-parameters, static buckets, paths.

Values follow the architecture document: a shallow GNN (2-3 message-passing layers),
DropEdge regularisation, a structure/attribute reconstruction trade-off ``alpha``,
fixed-size **static buckets** for ONNX/TensorRT, and an example decision threshold of
0.15 (the value used in the document's Rego snippet).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .data.schema import FEATURE_DIM

# Repository root = parent of ``src/`` (this file lives at src/config.py).
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR: Path = REPO_ROOT / "public"
CHECKPOINT_PATH: Path = ARTIFACTS_DIR / "checkpoint.pt"
NORM_STATS_PATH: Path = ARTIFACTS_DIR / "norm_stats.json"

# Streaming TGN (v2) deployment artifacts.
TGN_CHECKPOINT_PATH: Path = ARTIFACTS_DIR / "tgn_checkpoint.pt"
TGN_STATS_PATH: Path = ARTIFACTS_DIR / "tgn_stats.json"


def onnx_path(bucket: str) -> Path:
    """Path of the exported ONNX model for a given bucket name (S/M/L)."""
    return ARTIFACTS_DIR / f"model_{bucket}.onnx"


@dataclass(frozen=True)
class ModelConfig:
    """Graph Autoencoder architecture / training hyper-parameters."""

    feature_dim: int = FEATURE_DIM
    hidden_dim: int = 64
    embed_dim: int = 32
    num_layers: int = 2          # message-passing layers (document: 2-3)
    dropout: float = 0.1         # feature dropout inside the encoder
    dropedge_p: float = 0.2      # DropEdge probability (training only)
    alpha: float = 0.6           # score = alpha*structure + (1-alpha)*attribute

    # --- v2 levers -----------------------------------------------------------
    # Lever 1 — balanced structural loss (DOMINANT pos-weighting). Edge entries
    # (A=1) are weighted by this factor in the structure MSE so the sparse real
    # edges are not drowned by the abundant non-edges. ``None`` = auto, i.e. the
    # per-graph #neg/#pos ratio (clamped); a float pins a fixed weight.
    struct_pos_weight: float | None = None
    # Lever 3 — z-score the numeric feature block (not the type one-hot) using
    # benign training statistics; stats are stored in norm_stats and re-applied
    # at inference. Keeps per-feature attribute error comparable.
    standardize_features: bool = True
    # Lever 2 — inductive training: learn the benign *distribution* from a pool
    # of independently sampled benign graphs instead of a single one.
    num_train_graphs: int = 8
    num_val_graphs: int = 2
    # Lever 4 — data-driven decision threshold targeting this benign
    # false-positive rate; the calibrated value is stored in norm_stats.
    target_fpr: float = 0.05

    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    epochs: int = 300
    patience: int = 30           # early-stopping patience on validation loss

    seed: int = 42


@dataclass(frozen=True)
class SyntheticConfig:
    """Synthetic benign IAM graph generation parameters."""

    num_users: int = 60
    num_devices: int = 40
    num_ip_subnets: int = 15
    num_resources: int = 25
    num_roles: int = 6
    sessions_per_user: tuple[int, int] = (2, 8)   # uniform range
    devices_per_user: tuple[int, int] = (1, 3)
    resources_per_role: tuple[int, int] = (2, 6)
    anomaly_fraction: float = 0.05   # fraction of nodes perturbed (eval only)
    seed: int = 42


@dataclass(frozen=True)
class TGNConfig:
    """Streaming Temporal Graph Network (v2) hyper-parameters.

    The model memory is sized for ``capacity`` slots (= training entities +
    ``capacity_headroom``) so previously unseen ZTA entities can be admitted at
    inference time through the :class:`~graphagate.model.registry.NodeRegistry`.
    """

    # Synthetic stream shape (entity counts + number of events).
    num_users: int = 50
    num_ips: int = 100
    num_resources: int = 20
    num_events: int = 50000

    # Architecture.
    msg_dim: int = 6
    memory_dim: int = 64
    time_dim: int = 32
    node_feat_dim: int = 16

    # Optimisation.
    batch_size: int = 200
    epochs: int = 3
    learning_rate: float = 1e-3

    # Chronological split fractions (test = 1 - train - val).
    train_frac: float = 0.7
    val_frac: float = 0.1

    # Spare memory slots reserved for entities first seen at inference time.
    capacity_headroom: int = 512
    # Decision-threshold calibration target (benign false-positive rate).
    target_fpr: float = 0.05

    seed: int = 42

    @property
    def total_nodes(self) -> int:
        return self.num_users + self.num_ips + self.num_resources

    @property
    def capacity(self) -> int:
        return self.total_nodes + self.capacity_headroom


@dataclass(frozen=True)
class Config:
    """Top-level configuration bundle."""

    model: ModelConfig = field(default_factory=ModelConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)

    # Static buckets: max #nodes of a (padded) subgraph / ego-network. Padding nodes
    # are masked out of the loss and the anomaly score. Inputs to the exported ONNX
    # model are therefore always one of these fixed sizes.
    buckets: dict[str, int] = field(
        default_factory=lambda: {"S": 64, "M": 128, "L": 256}
    )

    # Example OPA decision threshold on the normalised [0,1] anomaly score.
    score_threshold: float = 0.15

    def bucket_for(self, n_nodes: int) -> tuple[str, int]:
        """Return the smallest bucket (name, size) that fits ``n_nodes`` nodes."""
        for name, size in sorted(self.buckets.items(), key=lambda kv: kv[1]):
            if n_nodes <= size:
                return name, size
        # Larger than the biggest bucket: fall back to the largest.
        name, size = max(self.buckets.items(), key=lambda kv: kv[1])
        return name, size


DEFAULT_CONFIG = Config()
