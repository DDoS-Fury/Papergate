"""Measure batched-replay metric drift + speedup on a real LANL subset.

Loads the LANL stream ONCE (cached to disk so repeated invocations skip the multi-GB gz
scan), then runs the *full* train_tgn pipeline at several ``eval_batch_size`` values on the
identical dataset. Training is seeded and identical across runs, so the wall-time delta is
attributable to the streaming replay, and the metric delta to the batched within-batch
staleness. ``eval_batch_size=1`` is the per-event reference.

Run (in the project image, GPU):
    docker run --rm --gpus all -v "$PWD/src:/app/src" -v "$PWD/tests:/app/tests" \
        -v "$PWD/data:/data" --entrypoint python graphagate \
        /app/tests/verify_lanl_drift.py --max-events 200000 --window-pad 0 \
        --benign-stride 10 --epochs 2 --train-frac 0.3 --batch-sizes 1,1024
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.lanl_auth import load_lanl_stream  # noqa: E402

from graphagate.config import TGNConfig  # noqa: E402
from graphagate.train_tgn import train_tgn  # noqa: E402

KEYS = ("agg_auc", "agg_ap", "lateral_recall_before", "lateral_recall_after",
        "fpr_before", "fpr_after")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--auth-path", default="/data/auth.txt.gz")
    p.add_argument("--redteam-path", default="/data/redteam.txt")
    p.add_argument("--max-events", type=int, default=200_000)
    p.add_argument("--benign-stride", type=int, default=10)
    p.add_argument("--window-pad", type=int, default=0)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--train-frac", type=float, default=0.3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--batch-sizes", default="1,1024")
    p.add_argument("--cache", default="")
    args = p.parse_args()

    # Param-aware cache name so changing the subset never reuses a stale StreamData.
    if not args.cache:
        args.cache = (f"/data/_lanl_cache_m{args.max_events}_s{args.benign_stride}"
                      f"_w{args.window_pad}.pt")

    if os.path.exists(args.cache):
        print(f"[cache] loading StreamData from {args.cache}")
        data = torch.load(args.cache, weights_only=False)
    else:
        t0 = time.time()
        data = load_lanl_stream(
            args.auth_path, args.redteam_path, max_events=args.max_events,
            benign_stride=args.benign_stride, window_pad=args.window_pad,
        )
        print(f"[load] {time.time() - t0:.1f}s")
        try:
            torch.save(data, args.cache)
            print(f"[cache] saved -> {args.cache}")
        except Exception as e:  # caching is best-effort
            print(f"[cache] save skipped: {e}")

    sizes = [int(x) for x in args.batch_sizes.split(",")]
    results = {}
    for bs in sizes:
        cfg = dataclasses.replace(
            TGNConfig(), epochs=args.epochs, train_frac=args.train_frac,
            val_frac=args.val_frac, eval_batch_size=bs,
        )
        print(f"\n================ eval_batch_size={bs} ================")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        m = train_tgn(cfg, dataset=data, save=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        results[bs] = (m, time.time() - t0)

    ref_bs = sizes[0]
    ref_m, ref_t = results[ref_bs]
    print("\n\n==================== DRIFT / SPEEDUP SUMMARY ====================")
    print(f"reference eval_batch_size={ref_bs}  total_time={ref_t:.1f}s")
    header = f"{'metric':24s} " + " ".join(f"bs={bs:<10d}" for bs in sizes)
    print(header)
    for k in KEYS:
        row = f"{k:24s} "
        for bs in sizes:
            row += f"{results[bs][0].get(k, float('nan')):<13.4f}"
        print(row)
    print(f"{'total_time_s':24s} " + "".join(f"{results[bs][1]:<13.1f}" for bs in sizes))
    print(f"{'speedup_vs_ref':24s} " + "".join(f"{ref_t / results[bs][1]:<13.2f}" for bs in sizes))

    print("\nmax |Δmetric| vs reference:")
    for bs in sizes[1:]:
        worst = max(abs(results[bs][0].get(k, 0.0) - ref_m.get(k, 0.0)) for k in KEYS)
        print(f"  bs={bs:5d}: {worst:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
