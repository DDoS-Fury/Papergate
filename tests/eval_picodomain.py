"""External-validity evaluation of the detector on the PicoDomain Zeek logs.

Runs the *exact* training / cost-sensitive-calibration / streaming-evaluation pipeline used
on the synthetic data (``graphagate.train_tgn``), but fed real network telemetry — the
PicoDomain capture of a Windows-domain intrusion — mapped to the v4 ZTA chain by
``tests/datasets/picodomain.py``.

Unlike LANL (host→host authentication only), PicoDomain populates **all five nodes** of the
chain from real fields, including the ``config`` node: its ``ssl.log`` carries ``ja3``. It
is therefore the one public dataset on which the config-node contribution — and with it the
credential-theft class — is measurable at all. It is small (2.67 days, 8 source IPs), so it
is a case study, not a headline benchmark. See ``docs/datasets.md``.

Reusing ``train_tgn(dataset=...)`` means the "global-FPR threshold vs cost-sensitive
routing" lateral-recall comparison, the per-type breakdown and the cold-start split are all
reported here with no duplicated logic.

Usage (dataset downloaded from https://github.com/iHeartGraph/PicoDomain, not committed;
extract ``Zeek_Logs.7z`` first):

    docker run --rm --gpus all \
        -v "$PWD/src:/app/src" -v "$PWD/tests:/app/tests" -v /path/to/pico:/data \
        --entrypoint python graphagate /app/tests/eval_picodomain.py \
        --log-dir /data/logs --red-log "/data/Red Log.xlsx"

or via the docker-compose ``eval-picodomain`` profile (mounts ./data → /data).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

# Make the sibling ``datasets`` package importable when run as a bare script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.picodomain import load_picodomain_stream  # noqa: E402

from graphagate.config import TGNConfig  # noqa: E402
from graphagate.train_tgn import train_tgn  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate the detector on PicoDomain Zeek logs.")
    p.add_argument("--log-dir", default=os.environ.get("PICO_LOGS", "/data/logs"))
    p.add_argument("--red-log", default=os.environ.get("PICO_REDLOG", "/data/Red Log.xlsx"))
    p.add_argument("--max-events", type=int, default=200_000)
    p.add_argument("--bind-ttl", type=float, default=36_000.0)
    p.add_argument("--label-window", type=float, default=90.0)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--cost-ratio", type=float, default=TGNConfig().cost_ratio)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    args = p.parse_args()

    print("--- LOADING PICODOMAIN STREAM ---")
    data = load_picodomain_stream(
        args.log_dir, args.red_log,
        max_events=args.max_events, bind_ttl=args.bind_ttl, label_window=args.label_window,
    )

    # Override only the knobs that matter for an injected stream; entity-count / num_events
    # fields are unused when ``dataset`` is supplied.
    cfg = dataclasses.replace(
        TGNConfig(), epochs=args.epochs, cost_ratio=args.cost_ratio,
        train_frac=args.train_frac, val_frac=args.val_frac,
        eval_batch_size=args.eval_batch_size,
    )

    print("\n--- TRAIN + EVALUATE (PicoDomain external validity) ---")
    metrics = train_tgn(cfg, dataset=data, save=False)

    print("\n--- PICODOMAIN SUMMARY ---")
    print(f"  aggregate AUC={metrics['agg_auc']:.4f} AP={metrics['agg_ap']:.4f}")
    print(f"  lateral recall  before(global-FPR)={metrics['lateral_recall_before']:.4f}"
          f"  after(cost-sensitive)={metrics['lateral_recall_after']:.4f}")
    print(f"  benign FPR      before={metrics['fpr_before']:.4f}"
          f"  after={metrics['fpr_after']:.4f}")
    for name in ("lateral", "theft", "contextual"):
        m = metrics["per_type"].get(name)
        if m:
            print(f"  {name:<11s} AUC={m['auc']:.4f} AP={m['ap']:.4f} "
                  f"recall@routed={m['recall']:.4f} n={m['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
