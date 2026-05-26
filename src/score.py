"""Inference & evaluation for the Graphagate GAE.

Loads the trained checkpoint and turns the GAE reconstruction error into the
calibrated ``[0, 1]`` "anomaly / legality" score consumed by OPA. The raw
per-node reconstruction error is mapped through the benign calibration anchors
(``lo``→0, ``hi``→1) stored at training time.

Two modes:
    - default: score a fresh benign graph and report the score distribution and
      how many nodes exceed the decision threshold (false-positive sanity check).
    - ``--eval``: inject anomalies into a benign graph and report ROC-AUC /
      PR-AUC plus precision/recall at the configured threshold — validating that
      the unsupervised model separates anomalous from benign nodes.

Usage::

    python -m graphagate.score [--eval] [--seed S]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch_geometric.data import Data

from graphagate.config import (
    CHECKPOINT_PATH,
    DEFAULT_CONFIG,
    NORM_STATS_PATH,
    Config,
    ModelConfig,
)
from graphagate import features as feat_utils
from graphagate.data.synthetic import generate_benign_graph, inject_anomalies, to_dense
from graphagate.model.gae import DominantGAE
from graphagate.model.losses import node_anomaly_score


def load_model(checkpoint_path=CHECKPOINT_PATH) -> tuple[DominantGAE, ModelConfig]:
    """Load a trained GAE and its config from a checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mcfg = ModelConfig(**ckpt["config"])
    model = DominantGAE(mcfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, mcfg


def load_norm_stats(path=NORM_STATS_PATH) -> dict:
    """Load benign calibration statistics."""
    return json.loads(path.read_text())


def normalize_scores(raw: np.ndarray, stats: dict) -> np.ndarray:
    """Map raw reconstruction errors to a calibrated ``[0, 1]`` anomaly score."""
    lo, hi = stats["lo"], stats["hi"]
    denom = max(hi - lo, 1e-8)
    return np.clip((raw - lo) / denom, 0.0, 1.0)


def _build_feat_tensors(
    stats: dict,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Extract feature mean/std tensors from norm_stats if standardization was on."""
    if stats.get("standardize_features"):
        feat_mean = torch.tensor(stats["feat_mean"], dtype=torch.float32)
        feat_std = torch.tensor(stats["feat_std"], dtype=torch.float32)
        return feat_mean, feat_std
    return None, None


@torch.no_grad()
def raw_scores(
    model: DominantGAE,
    data: Data,
    alpha: float,
    feat_mean: torch.Tensor | None = None,
    feat_std: torch.Tensor | None = None,
) -> np.ndarray:
    """Per-node raw reconstruction error for every real node in ``data``.

    Pads the graph to the smallest static bucket that fits it (the ego-network /
    bucketing path used at inference), optionally standardizes features using the
    training-pool statistics, runs the model, and returns scores for the ``N``
    real nodes only.

    Parameters
    ----------
    model:
        Trained :class:`~graphagate.model.gae.DominantGAE`.
    data:
        PyG :class:`~torch_geometric.data.Data` graph to score.
    alpha:
        Structure/attribute trade-off coefficient.
    feat_mean:
        Per-numeric-feature mean ``[NUM_NUMERIC_FEATURES]`` from training, or
        ``None`` to skip standardization.
    feat_std:
        Per-numeric-feature std ``[NUM_NUMERIC_FEATURES]`` from training, or
        ``None`` to skip standardization.
    """
    n = data.num_nodes
    _, size = DEFAULT_CONFIG.bucket_for(n)
    size = max(size, n)  # full-graph eval may exceed the largest static bucket
    x, adj, mask = to_dense(data, size)

    # Lever 3: apply feature standardization before model inference.
    if feat_mean is not None and feat_std is not None:
        x = feat_utils.standardize(x, feat_mean, feat_std, mask)

    a_hat, x_hat, _ = model(x, adj)
    scores = node_anomaly_score(adj, a_hat, x, x_hat, alpha, mask).numpy()
    return scores[:n]


def _print_distribution(norm: np.ndarray, threshold: float) -> None:
    flagged = int((norm >= threshold).sum())
    print(
        f"  normalised score: min={norm.min():.3f} mean={norm.mean():.3f} "
        f"p95={np.percentile(norm, 95):.3f} max={norm.max():.3f}"
    )
    print(
        f"  nodes >= threshold ({threshold:.3f}): {flagged}/{len(norm)} "
        f"({100 * flagged / len(norm):.1f}%)"
    )


def evaluate(config: Config = DEFAULT_CONFIG, seed: int | None = None) -> dict[str, float]:
    """Inject anomalies and report ROC-AUC / PR-AUC and threshold metrics."""
    model, mcfg = load_model()
    stats = load_norm_stats()
    base_seed = seed if seed is not None else mcfg.seed + 7

    # Lever 4: use calibrated threshold from norm_stats; fall back to config value.
    thr = float(stats.get("threshold", config.score_threshold))
    target_fpr = stats.get("target_fpr", config.score_threshold)

    # Lever 3: load feature stats if standardization was applied at training.
    feat_mean, feat_std = _build_feat_tensors(stats)

    benign = generate_benign_graph(config.synthetic, seed=base_seed)
    graph = inject_anomalies(benign, config.synthetic, seed=base_seed)
    y = graph.y.numpy()

    raw = raw_scores(model, graph, mcfg.alpha, feat_mean=feat_mean, feat_std=feat_std)
    norm = normalize_scores(raw, stats)

    roc = roc_auc_score(y, raw)
    pr = average_precision_score(y, raw)
    base_rate = float(y.mean())

    pred = (norm >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )

    print("=" * 60)
    print("  Graphagate GAE — unsupervised anomaly detection (eval)")
    print("=" * 60)
    print(f"  graph: {graph.num_nodes} nodes | anomalies: {int(y.sum())} "
          f"(base rate {base_rate:.3f})")
    print(f"  ROC-AUC : {roc:.4f}")
    print(f"  PR-AUC  : {pr:.4f}  (random baseline = {base_rate:.3f})")
    print(f"  @threshold {thr:.3f} (calibrated, target FPR {target_fpr}): "
          f"precision={prec:.3f} recall={rec:.3f} f1={f1:.3f}")
    print("-" * 60)
    print("  benign node scores:")
    _print_distribution(norm[y == 0], thr)
    print("  anomalous node scores:")
    _print_distribution(norm[y == 1], thr)
    print("=" * 60)

    return {
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "threshold": thr,
    }


def score_benign(config: Config = DEFAULT_CONFIG, seed: int | None = None) -> None:
    """Score a fresh benign graph (false-positive sanity check)."""
    model, mcfg = load_model()
    stats = load_norm_stats()
    base_seed = seed if seed is not None else mcfg.seed + 13

    # Lever 4: use calibrated threshold from norm_stats; fall back to config value.
    thr = float(stats.get("threshold", config.score_threshold))
    target_fpr = stats.get("target_fpr", config.score_threshold)

    # Lever 3: load feature stats if standardization was applied at training.
    feat_mean, feat_std = _build_feat_tensors(stats)

    graph = generate_benign_graph(config.synthetic, seed=base_seed)
    norm = normalize_scores(
        raw_scores(model, graph, mcfg.alpha, feat_mean=feat_mean, feat_std=feat_std),
        stats,
    )
    print("=" * 60)
    print("  Graphagate GAE — benign scoring (sanity check)")
    print("=" * 60)
    print(f"  graph: {graph.num_nodes} nodes (all benign)")
    _print_distribution(norm, thr)
    print(f"  (calibrated threshold={thr:.3f}, target FPR={target_fpr})")
    print("=" * 60)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score / evaluate the Graphagate GAE.")
    parser.add_argument("--eval", action="store_true",
                        help="Inject anomalies and report ROC/PR-AUC.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    args = parser.parse_args(argv)

    if args.eval:
        evaluate(DEFAULT_CONFIG, seed=args.seed)
    else:
        score_benign(DEFAULT_CONFIG, seed=args.seed)


if __name__ == "__main__":
    main()
