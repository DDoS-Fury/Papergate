"""Unsupervised training of the DOMINANT Graph Autoencoder on benign IAM graphs.

The model is trained **only on benign traffic** (the document's "log benigni e
consolidati"): it learns to reconstruct the relational topology + node features
of a normal Zero-Trust access graph. At inference, deviating behaviour yields a
high reconstruction error → anomaly score.

Training runs on a *pool* of independently sampled benign graphs (full dense
adjacency, sizes vary per graph). The static buckets in :mod:`graphagate.config`
are an *inference-time* concern (padded ego-networks for ONNX/TensorRT), not a
training one.

Outputs (written to ``public/``):
    - ``checkpoint.pt``   — model weights + serialised ModelConfig.
    - ``norm_stats.json`` — benign score statistics used to map raw
      reconstruction errors to a calibrated ``[0, 1]`` "legality / anomaly"
      score, plus feature normalisation statistics and the calibrated threshold.

Usage::

    python -m graphagate.train [--epochs N] [--seed S] [--quiet]
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict

import numpy as np
import torch

from graphagate.config import (
    CHECKPOINT_PATH,
    DEFAULT_CONFIG,
    NORM_STATS_PATH,
    Config,
)
from graphagate import features as feat_utils
from graphagate.data.synthetic import generate_benign_graph, to_dense
from graphagate.model.gae import DominantGAE
from graphagate.model.losses import node_anomaly_score, reconstruction_loss


def _benign_score_stats(scores: np.ndarray) -> dict[str, float]:
    """Summarise the benign per-node score distribution for [0,1] calibration."""
    return {
        "score_min": float(scores.min()),
        "score_p50": float(np.percentile(scores, 50)),
        "score_p95": float(np.percentile(scores, 95)),
        "score_p99": float(np.percentile(scores, 99)),
        "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        # Calibration anchors: benign min -> 0, benign p99 -> 1 (clamped).
        "lo": float(scores.min()),
        "hi": float(np.percentile(scores, 99)),
    }


def train(
    config: Config = DEFAULT_CONFIG,
    epochs: int | None = None,
    seed: int | None = None,
    verbose: bool = True,
) -> tuple[DominantGAE, dict]:
    """Train the GAE on a pool of benign graphs; return the model and score stats.

    Lever 2 (inductive training): learns from a pool of ``mcfg.num_train_graphs``
    independently seeded benign graphs, with a disjoint validation pool of
    ``mcfg.num_val_graphs`` graphs.

    Lever 3 (feature standardization): when ``mcfg.standardize_features`` is
    ``True``, z-score normalises the numeric feature block using statistics
    computed over the training pool and stores them in the returned stats dict.

    Lever 4 (threshold calibration): computes a decision threshold that targets
    ``mcfg.target_fpr`` false-positive rate on the validation pool's benign
    scores; stored in stats as ``"threshold"``.
    """
    mcfg = config.model
    torch.manual_seed(seed if seed is not None else mcfg.seed)

    base_seed = seed if seed is not None else mcfg.seed

    # -------------------------------------------------------------------------
    # Lever 2: build train and validation pools
    # -------------------------------------------------------------------------
    # Train pool: seeds base_seed + i for i in range(num_train_graphs).
    train_graphs = [
        generate_benign_graph(config.synthetic, seed=base_seed + i)
        for i in range(mcfg.num_train_graphs)
    ]
    # Validation pool: seeds base_seed + 1000 + i (disjoint from train).
    val_graphs = [
        generate_benign_graph(config.synthetic, seed=base_seed + 1000 + i)
        for i in range(mcfg.num_val_graphs)
    ]

    # Convert to dense tensors (full size per graph; sizes may differ).
    train_dense = [
        to_dense(g, g.num_nodes) for g in train_graphs
    ]  # list of (x, adj, mask)
    val_dense = [
        to_dense(g, g.num_nodes) for g in val_graphs
    ]

    if verbose:
        train_nodes = [g.num_nodes for g in train_graphs]
        val_nodes = [g.num_nodes for g in val_graphs]
        print(
            f"[data] train pool: {mcfg.num_train_graphs} graphs, "
            f"nodes={train_nodes} | "
            f"val pool: {mcfg.num_val_graphs} graphs, nodes={val_nodes}"
        )

    # -------------------------------------------------------------------------
    # Lever 3: feature standardization
    # -------------------------------------------------------------------------
    feat_mean: torch.Tensor | None = None
    feat_std: torch.Tensor | None = None

    if mcfg.standardize_features:
        x_train_list = [xd[0] for xd in train_dense]
        mask_train_list = [xd[2] for xd in train_dense]
        feat_mean, feat_std = feat_utils.compute_feature_stats(x_train_list, mask_train_list)

        # Apply standardization to all train graphs.
        train_dense = [
            (feat_utils.standardize(x, feat_mean, feat_std, mask), adj, mask)
            for (x, adj, mask) in train_dense
        ]
        # Apply the same standardization to all val graphs.
        val_dense = [
            (feat_utils.standardize(x, feat_mean, feat_std, mask), adj, mask)
            for (x, adj, mask) in val_dense
        ]

        if verbose:
            print(
                f"[features] standardize_features=True; "
                f"numeric mean=[{feat_mean.numpy().round(4)}] "
                f"std=[{feat_std.numpy().round(4)}]"
            )

    # -------------------------------------------------------------------------
    # Model + optimiser
    # -------------------------------------------------------------------------
    model = DominantGAE(mcfg)
    optim = torch.optim.Adam(
        model.parameters(), lr=mcfg.learning_rate, weight_decay=mcfg.weight_decay
    )

    n_epochs = epochs if epochs is not None else mcfg.epochs
    best_val = float("inf")
    best_state: dict | None = None
    patience = 0
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        # ------------------------------------------------------------------
        # Train step: mean loss over the train pool.
        # DropEdge active via model.train().
        # ------------------------------------------------------------------
        model.train()
        optim.zero_grad()

        train_loss_accum = torch.tensor(0.0)
        for x_tr, adj_tr, mask_tr in train_dense:
            a_hat, x_hat, _ = model(x_tr, adj_tr)
            g_loss = reconstruction_loss(
                adj_tr, a_hat, x_tr, x_hat, mcfg.alpha, mask_tr,
                struct_pos_weight=mcfg.struct_pos_weight,
            )
            train_loss_accum = train_loss_accum + g_loss

        # Mean over pool, single backward.
        loss = train_loss_accum / max(len(train_dense), 1)
        loss.backward()
        optim.step()

        # ------------------------------------------------------------------
        # Validation: mean loss over the val pool; DropEdge off.
        # ------------------------------------------------------------------
        model.eval()
        with torch.no_grad():
            val_loss_accum = 0.0
            for x_va, adj_va, mask_va in val_dense:
                va_hat, vx_hat, _ = model(x_va, adj_va)
                val_loss_accum += reconstruction_loss(
                    adj_va, va_hat, x_va, vx_hat, mcfg.alpha, mask_va,
                    struct_pos_weight=mcfg.struct_pos_weight,
                ).item()
            val_loss = val_loss_accum / max(len(val_dense), 1)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1

        if verbose and (epoch == 1 or epoch % 20 == 0 or patience == 0):
            print(
                f"[epoch {epoch:4d}] train_loss={loss.item():.5f} "
                f"val_loss={val_loss:.5f} best={best_val:.5f} patience={patience}"
            )

        if patience >= mcfg.patience:
            if verbose:
                print(f"[early-stop] no val improvement for {mcfg.patience} epochs.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if verbose:
        print(f"[train] done in {time.time() - t0:.1f}s | best val_loss={best_val:.5f}")

    # -------------------------------------------------------------------------
    # Benign score statistics for [0,1] calibration — computed over TRAIN pool.
    # -------------------------------------------------------------------------
    model.eval()
    all_train_scores: list[np.ndarray] = []
    with torch.no_grad():
        for x_tr, adj_tr, mask_tr in train_dense:
            a_hat, x_hat, _ = model(x_tr, adj_tr)
            n_real = int(mask_tr.sum().item())
            scores = node_anomaly_score(
                adj_tr, a_hat, x_tr, x_hat, mcfg.alpha, mask_tr
            ).numpy()
            all_train_scores.append(scores[:n_real])

    train_scores_all = np.concatenate(all_train_scores)
    stats = _benign_score_stats(train_scores_all)
    stats["alpha"] = mcfg.alpha

    # -------------------------------------------------------------------------
    # Lever 4: calibrated threshold targeting target_fpr on the val pool.
    # -------------------------------------------------------------------------
    lo = stats["lo"]
    hi = stats["hi"]
    denom = max(hi - lo, 1e-8)

    def _normalize(raw: np.ndarray) -> np.ndarray:
        return np.clip((raw - lo) / denom, 0.0, 1.0)

    # Compute normalised benign scores on the val pool (or fall back to train).
    if val_dense:
        norm_pool_scores: list[np.ndarray] = []
        with torch.no_grad():
            for x_va, adj_va, mask_va in val_dense:
                a_hat, x_hat, _ = model(x_va, adj_va)
                n_real = int(mask_va.sum().item())
                scores_va = node_anomaly_score(
                    adj_va, a_hat, x_va, x_hat, mcfg.alpha, mask_va
                ).numpy()
                norm_pool_scores.append(_normalize(scores_va[:n_real]))
        norm_val = np.concatenate(norm_pool_scores)
    else:
        # No val graphs: fall back to normalised train scores.
        norm_val = _normalize(train_scores_all)

    # threshold = quantile(norm_val, 1 - target_fpr)
    threshold = float(np.quantile(norm_val, 1.0 - mcfg.target_fpr))
    stats["threshold"] = threshold
    stats["target_fpr"] = mcfg.target_fpr

    if verbose:
        print(
            f"[calib] target_fpr={mcfg.target_fpr} → "
            f"threshold={threshold:.4f} "
            f"(computed on {'val' if val_dense else 'train'} pool)"
        )

    # -------------------------------------------------------------------------
    # Lever 3: store feature stats in norm_stats.
    # -------------------------------------------------------------------------
    if mcfg.standardize_features and feat_mean is not None and feat_std is not None:
        stats["standardize_features"] = True
        stats["feat_mean"] = feat_mean.tolist()
        stats["feat_std"] = feat_std.tolist()
    else:
        stats["standardize_features"] = False

    return model, stats


def save_artifacts(model: DominantGAE, stats: dict) -> None:
    """Persist the checkpoint and normalisation stats under ``public/``."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "config": asdict(model.cfg)},
        CHECKPOINT_PATH,
    )
    NORM_STATS_PATH.write_text(json.dumps(stats, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the Graphagate GAE on benign IAM graphs.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-epoch logging.")
    args = parser.parse_args(argv)

    model, stats = train(
        DEFAULT_CONFIG, epochs=args.epochs, seed=args.seed, verbose=not args.quiet
    )
    save_artifacts(model, stats)
    print(f"[save] checkpoint  -> {CHECKPOINT_PATH}")
    print(f"[save] norm_stats  -> {NORM_STATS_PATH}")
    print(
        f"[calib] benign score: min={stats['score_min']:.4f} "
        f"p50={stats['score_p50']:.4f} p99={stats['score_p99']:.4f} "
        f"max={stats['score_max']:.4f}"
    )
    print(
        f"[calib] threshold={stats['threshold']:.4f} "
        f"(target_fpr={stats['target_fpr']})"
    )


if __name__ == "__main__":
    main()
