"""Isolated score-level fidelity of the batched replay on real LANL data.

The thresholded metrics (AUC / FPR) on a small LANL subset are dominated by tiny-positive
noise and a possibly-degenerate validation calibration. To measure the *pure* batching
effect we instead compare the raw per-event anomaly scores at several ``batch_size`` values
against the per-event reference, on one identically-warmed model:

  * warm a model once (via a batched gate-by-label replay over train+val), then deepcopy it;
  * run the TEST replay at each batch size with ``gate_by_label=True`` so the SET of
    memory-update events is identical across runs — the only difference is that batching
    defers those updates to the end of each block (the within-batch staleness);
  * report mean/max |Δscore|, Spearman ρ, and the benign FP-flip rate at the ref's q99,
    all vs the reference. Two independent ``batch_size=1`` runs give the GPU-nondeterminism
    noise floor to compare the batched drift against.

Run (GPU; reuses the drift cache so no gz re-scan):
    docker run --rm --gpus all -v "$PWD/src:/app/src" -v "$PWD/tests:/app/tests" \
        -v "$PWD/data:/data" --entrypoint python graphagate \
        /app/tests/verify_lanl_scores.py --cache /data/_lanl_cache_m300000_s300_w0.pt
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graphagate.config import TGNConfig  # noqa: E402
from graphagate.train_tgn import _replay  # noqa: E402
from verify_replay_batching import _build_model  # noqa: E402


def _spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--train-frac", type=float, default=0.3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--batch-sizes", default="1,1,128,256,512,1024,2048")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(args.cache, weights_only=False)
    n = len(data.dst)
    train_end = int(n * args.train_frac)
    val_end = int(n * (args.train_frac + args.val_frac))
    print(f"events={n} test=[{val_end},{n}) test_lat={int((data.types[val_end:] == 3).sum())}")

    cfg = TGNConfig()
    base = _build_model(data, cfg, device)
    # Warm once over train+val (batched, gate-by-label) — identical starting state for all runs.
    _replay(base, None, None, data.user[:val_end], data.dst[:val_end], data.t[:val_end],
            data.msg[:val_end], data.y[:val_end], device, gate_by_label=True,
            batch_size=2048, desc="warm")

    def test_replay(bs, tag):
        m = copy.deepcopy(base)
        s, _ = _replay(m, None, None, data.user[val_end:], data.dst[val_end:], data.t[val_end:],
                       data.msg[val_end:], data.y[val_end:], device, gate_by_label=True,
                       batch_size=bs, desc=tag)
        return s

    sizes = [int(x) for x in args.batch_sizes.split(",")]
    scores, tags = [], []
    seen1 = 0
    for bs in sizes:
        tag = f"bs={bs}" + ("(b)" if (bs == 1 and seen1) else "")
        seen1 += int(bs == 1)
        scores.append(test_replay(bs, tag))
        tags.append(tag)

    ref = scores[0]
    benign = (data.y[val_end:].numpy() == 0)
    q99 = np.quantile(ref[benign], 0.99)
    ref_flag = ref[benign] >= q99

    print("\n==================== SCORE-LEVEL DRIFT vs ref (bs=1) ====================")
    print(f"{'run':12s} {'mean|Δ|':>10s} {'max|Δ|':>10s} {'spearman':>10s} {'FPflip@q99':>11s}")
    for tag, s in zip(tags, scores):
        dabs = np.abs(s - ref)
        flip = float(np.mean((s[benign] >= q99) != ref_flag))
        print(f"{tag:12s} {dabs.mean():10.3e} {dabs.max():10.3e} {_spearman(s, ref):10.5f} {flip:11.4f}")
    print("\n(the second bs=1 row is the GPU-nondeterminism noise floor; compare batched rows to it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
