"""External-validity evaluation of the lateral-movement detector on LANL auth.

Runs the *exact* training / cost-sensitive-calibration / streaming-evaluation pipeline used
on the synthetic data (``graphagate.train_tgn``), but fed a real public dataset — the LANL
'Comprehensive Multi-Source' authentication logs with red-team lateral-movement labels —
mapped to the ZTA stream schema by ``tests/datasets/lanl_auth.py``.

Reusing ``train_tgn(dataset=...)`` means the headline "global-FPR threshold vs cost-sensitive
routing" lateral-recall comparison, the per-type breakdown and the cold-start split are all
reported for LANL with no duplicated logic.

Usage (files downloaded from https://csr.lanl.gov/data/cyber1/, not committed):

    docker run --rm --gpus all \
        -v "$PWD/src:/app/src" -v "$PWD/tests:/app/tests" -v /path/to/lanl:/data \
        --entrypoint python graphagate /app/tests/eval_lanl.py \
        --auth-path /data/auth.txt.gz --redteam-path /data/redteam.txt \
        --max-events 200000 --epochs 8

or via the docker-compose ``eval-lanl`` profile (mounts ./data → /data).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

# Make the sibling ``datasets`` package importable when run as a bare script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.lanl_auth import load_lanl_stream  # noqa: E402

from graphagate.config import TGNConfig  # noqa: E402
from graphagate.train_tgn import train_tgn  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate the TGN lateral detector on LANL auth.")
    p.add_argument("--auth-path", default=os.environ.get("LANL_AUTH", "/data/auth.txt.gz"))
    p.add_argument("--redteam-path", default=os.environ.get("LANL_REDTEAM", "/data/redteam.txt"))
    p.add_argument("--max-events", type=int, default=200_000)
    p.add_argument("--benign-stride", type=int, default=16)
    p.add_argument("--window-pad", type=int, default=3600)
    p.add_argument("--window-start", type=int, default=None)
    p.add_argument("--window-end", type=int, default=None)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--cost-ratio", type=float, default=TGNConfig().cost_ratio)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--val-frac", type=float, default=0.1)
    # Offline streaming-eval batch size. >1 scores a block against the start-of-batch memory
    # snapshot then commits in batch (standard batched-TGN regime) to saturate the GPU; the
    # online serving path is unaffected. Lower it if you hit CUDA OOM. See train_tgn._replay.
    p.add_argument("--eval-batch-size", type=int, default=1024)
    args = p.parse_args()

    window = None
    if args.window_start is not None and args.window_end is not None:
        window = (args.window_start, args.window_end)

    print("--- LOADING LANL AUTH STREAM ---")
    data = load_lanl_stream(
        args.auth_path, args.redteam_path,
        max_events=args.max_events, benign_stride=args.benign_stride,
        window_pad=args.window_pad, window=window,
    )

    # Override only the knobs that matter for an injected stream; entity-count / num_events
    # fields are unused when ``dataset`` is supplied.
    cfg = dataclasses.replace(TGNConfig(), epochs=args.epochs, cost_ratio=args.cost_ratio, train_frac=args.train_frac, val_frac=args.val_frac, eval_batch_size=args.eval_batch_size)

    print("\n--- TRAIN + EVALUATE (LANL external validity) ---")
    metrics = train_tgn(cfg, dataset=data, save=False)

    print("\n--- LANL SUMMARY ---")
    print(f"  aggregate AUC={metrics['agg_auc']:.4f} AP={metrics['agg_ap']:.4f}")
    print(f"  lateral recall  before(global-FPR)={metrics['lateral_recall_before']:.4f}"
          f"  after(cost-sensitive)={metrics['lateral_recall_after']:.4f}")
    print(f"  benign FPR      before={metrics['fpr_before']:.4f}"
          f"  after={metrics['fpr_after']:.4f}")
    lat = metrics["per_type"].get("lateral")
    if lat:
        print(f"  lateral AUC={lat['auc']:.4f} AP={lat['ap']:.4f} "
              f"recall@routed={lat['recall']:.4f} n={lat['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
