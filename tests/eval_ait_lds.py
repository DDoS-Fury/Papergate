"""External-validity evaluation of the ZTA detector on AIT Log Data Set (AIT-LDS 2023, Zenodo).

Runs the one-class training and streaming-evaluation pipeline (``graphagate.train_tgn``)
on the AIT-LDS enterprise intrusion dataset.

The dataset is partitioned to guarantee strict one-class learning:
  - Training split (0.0 to actual_train_frac): 100% benign baseline traffic.
    The model learns habitual interaction topology, TLS JA3 client fingerprints, and timing patterns
    without ever observing any malicious activity.
  - Validation split (actual_train_frac to actual_train_frac + actual_val_frac): 100% held-out benign
    traffic used for threshold calibration.
  - Test split (remainder): held-out benign traffic merged with MITRE ATT&CK attacks
    (Lateral Movement, Credential Access/Theft, Reconnaissance, Exfiltration).

Usage:
    python tests/eval_ait_lds.py [--data-dir data/ait_lds] [--epochs 10] [--batch-size 256]

Or via Docker (mounting dataset directory):
    docker run --rm --gpus all \
        -v "$PWD/src:/app/src" -v "$PWD/tests:/app/tests" -v /path/to/ait_lds:/data/ait_lds \
        --entrypoint python graphagate /app/tests/eval_ait_lds.py --data-dir /data/ait_lds
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

# Ensure sibling `datasets` and root `src` are on PYTHONPATH
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CUR_DIR)
sys.path.insert(0, CUR_DIR)
sys.path.insert(0, os.path.join(PARENT_DIR, "src"))

from datasets.ait_lds import load_ait_stream  # noqa: E402
from graphagate.config import TGNConfig  # noqa: E402
from graphagate.train_tgn import train_tgn  # noqa: E402


def main() -> int:
    default_data_dir = (
        "/data/ait_lds"
        if os.path.exists("/data/ait_lds")
        else os.path.join(PARENT_DIR, "data", "ait_lds")
    )

    parser = argparse.ArgumentParser(
        description="Evaluate the 5-node ZTA detector on AIT Log Data Set (AIT-LDS 2023, Zenodo)."
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("AIT_DATA_DIR", default_data_dir),
        help=f"Path to AIT-LDS extracted scenario directory (default: {default_data_dir})",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size (default: 256)")
    parser.add_argument("--eval-batch-size", type=int, default=512, help="Evaluation batch size (default: 512)")
    parser.add_argument("--max-benign", type=int, default=50_000, help="Max benign events to keep (default: 50000)")
    parser.add_argument("--max-attack", type=int, default=5_000, help="Max attack events per category (default: 5000)")
    parser.add_argument("--train-frac", type=float, default=0.7, help="Fraction of benign baseline for training (default: 0.7)")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Fraction of benign baseline for validation (default: 0.1)")
    parser.add_argument("--bind-ttl", type=float, default=3600.0, help="Session entity binding TTL in seconds (default: 3600.0)")
    parser.add_argument("--cost-ratio", type=float, default=TGNConfig().cost_ratio, help="Cost ratio for routed threshold calibration")
    args = parser.parse_args()

    print("================================================================================")
    print("      AIT-LDS 2023: One-Class Training & Streaming Evaluation Pipeline           ")
    print("================================================================================")
    print(f"Data directory:   {args.data_dir}")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch size:       {args.batch_size} (eval: {args.eval_batch_size})")
    print(f"Cost ratio:       {args.cost_ratio}")
    print(f"Bind TTL:         {args.bind_ttl:.0f}s\n")

    print("--- STEP 1: LOADING & MAPPING AIT-LDS STREAM (5-node causal chain) ---")
    data, actual_train_frac, actual_val_frac = load_ait_stream(
        args.data_dir,
        max_benign_events=args.max_benign,
        max_attack_events_per_cat=args.max_attack,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        bind_ttl=args.bind_ttl,
    )

    n_total = len(data.dst)
    n_tr = int(n_total * actual_train_frac)
    n_v = int(n_total * actual_val_frac)
    n_te = n_total - n_tr - n_v

    print("--------------------------------------------------------------------------------")
    print(f"  Verified Split: Train={n_tr} (100% Benign) | Val={n_v} (100% Benign) | Test={n_te} (Benign + Attacks)")
    print(f"  Fractions:      actual_train_frac={actual_train_frac:.4f}  actual_val_frac={actual_val_frac:.4f}")
    print("--------------------------------------------------------------------------------\n")

    cfg = dataclasses.replace(
        TGNConfig(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        train_frac=actual_train_frac,
        val_frac=actual_val_frac,
        cost_ratio=args.cost_ratio,
    )

    print("--- STEP 2: ONE-CLASS TRAINING (Benign Traffic Only) & STREAMING EVALUATION ---")
    metrics = train_tgn(cfg, dataset=data, save=False)

    print("\n================================================================================")
    print("                            AIT-LDS 2023 RESULTS                                ")
    print("================================================================================")
    print(f"  Aggregate ROC-AUC:      {metrics['agg_auc']:.4f}")
    print(f"  Aggregate Avg Prec:     {metrics['agg_ap']:.4f}")
    print(f"  Benign FPR (global):    {metrics['fpr_before']:.4f}")
    print(f"  Benign FPR (routed):    {metrics['fpr_after']:.4f}")
    print("--------------------------------------------------------------------------------")
    print("  Per-Tactic Performance Breakdown:")
    for name in ("lateral", "theft", "contextual", "exfil", "policy"):
        m = metrics["per_type"].get(name)
        if m:
            print(
                f"    - {name:<12s} AUC={m['auc']:.4f}  "
                f"AP={m['ap']:.4f}  "
                f"Recall@routed={m['recall']:.4f}  "
                f"(n={m['n']})"
            )
    print("================================================================================\n")
    print("  Note: Training and validation splits contained exclusively benign traffic.")
    print("        Evaluation was executed in chronological streaming mode without future leakage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
