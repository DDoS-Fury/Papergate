"""External-validity evaluation of the ZTA detector on UWF-ZeekData24.

Runs the one-class training and streaming-evaluation pipeline (``graphagate.train_tgn``)
on the UWF-ZeekData24 cyber-range dataset.

Training is performed strictly one-class on the benign baseline slice (first 70% of benign traffic),
so the model learns normal network interactions and habitual topologies without ever seeing an attack.
Evaluation is performed on the test slice containing both unseen benign traffic and MITRE ATT&CK attacks.

Usage:
    python tests/eval_uwf_zeekdata.py [--data-dir data/uwf_zeekdata24] [--epochs 5]
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

# Ensure the sibling `datasets` package and root `src` are importable.
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CUR_DIR)
sys.path.insert(0, CUR_DIR)
sys.path.insert(0, os.path.join(PARENT_DIR, "src"))

from datasets.uwf_zeekdata import load_uwf_stream  # noqa: E402
from graphagate.config import TGNConfig  # noqa: E402
from graphagate.train_tgn import train_tgn  # noqa: E402


def main() -> int:
    default_data_dir = os.path.join(PARENT_DIR, "data", "uwf_zeekdata24")
    parser = argparse.ArgumentParser(description="Evaluate the ZTA detector on UWF-ZeekData24.")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("UWF_DATA_DIR", default_data_dir),
        help=f"Path to UWF-ZeekData24 directory (default: {default_data_dir})",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size (default: 256)")
    parser.add_argument("--eval-batch-size", type=int, default=512, help="Evaluation batch size (default: 512)")
    parser.add_argument("--max-benign", type=int, default=30_000, help="Max benign events to keep (default: 30000)")
    parser.add_argument("--max-attack", type=int, default=3_000, help="Max attack events per category (default: 3000)")
    parser.add_argument("--train-frac", type=float, default=0.7, help="Fraction of benign for train (default: 0.7)")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Fraction of benign for val (default: 0.1)")
    args = parser.parse_args()

    print("================================================================")
    print("      UWF-ZeekData24: One-Class Training & Evaluation Pipeline   ")
    print("================================================================")
    print(f"Data directory: {args.data_dir}")
    print(f"Epochs:         {args.epochs}")
    print(f"Batch size:     {args.batch_size} (eval: {args.eval_batch_size})\n")

    print("--- STEP 1: LOADING & MAPPING UWF-ZEEKDATA24 STREAM ---")
    data, actual_train_frac, actual_val_frac = load_uwf_stream(
        args.data_dir,
        max_benign_events=args.max_benign,
        max_attack_events_per_cat=args.max_attack,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )

    n_total = len(data.dst)
    n_tr = int(n_total * actual_train_frac)
    n_v = int(n_total * actual_val_frac)
    n_te = n_total - n_tr - n_v
    print(f"[uwf] Verified Split: Train={n_tr} (100% benign), Val={n_v} (100% benign), Test={n_te} (benign + attacks)\n")

    cfg = dataclasses.replace(
        TGNConfig(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        train_frac=actual_train_frac,
        val_frac=actual_val_frac,
    )

    print("\n--- STEP 2: ONE-CLASS TRAINING (Benign Traffic Only) & STREAMING EVALUATION ---")
    metrics = train_tgn(cfg, dataset=data, save=False)

    print("\n================================================================")
    print("                   UWF-ZeekData24 RESULTS                       ")
    print("================================================================")
    print(f"  Aggregate AUC:          {metrics['agg_auc']:.4f}")
    print(f"  Aggregate AP:           {metrics['agg_ap']:.4f}")
    print(f"  Benign FPR (before):    {metrics['fpr_before']:.4f}")
    print(f"  Benign FPR (after):     {metrics['fpr_after']:.4f}")
    print("----------------------------------------------------------------")
    print("  Per-Tactic Breakdown:")
    for name in ("theft", "contextual", "lateral", "exfil"):
        m = metrics["per_type"].get(name)
        if m:
            print(f"    - {name:<12s} AUC={m['auc']:.4f}  AP={m['ap']:.4f}  recall@routed={m['recall']:.4f}  (n={m['n']})")
    print("================================================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
