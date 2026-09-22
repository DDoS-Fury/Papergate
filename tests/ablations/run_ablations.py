"""Ablation study for the streaming TGN — isolates each component whose contribution
is the actual novelty claim, focusing on the only genuinely hard / model-owned class:
**lateral movement** (policy is OPA-owned; contextual is rule-trivial).

Components toggled (full pipeline = all ON):
  * **history features** (``use_hist_feats``) — explicit causal interaction-history counts;
  * **kill-chain precursor** (``use_precursor``) — serving-time recon→lateral prior;
  * **structural-compatibility head** (``use_struct_head``) — scaled cosine of embeddings;
  * **hashed-identity embedding** (``use_hash_identity``) — deterministic per-entity identity.

It runs the full training+evaluation pipeline (``graphagate.train_tgn.train_tgn``) with
``save=False`` (the deployable checkpoint in ``public/`` is never touched) for every variant
across several seeds, then prints a per-variant table with **mean ± std** so the lateral
lift is reported above the run-to-run (CUDA) noise. The temporal machinery itself
(recurrent memory + temporal neighbourhood) is ablated separately by the *Static GNN*
baseline (``tests/baselines/simple_gnn``), which now shares the exact same tabular signals
(history counts + precursor), so its gap to the TGN isolates the temporal-graph value.

Run inside the project's Docker image (torch required):

    docker compose --profile ablations up
    docker compose run --rm ablations /app/tests/ablations/run_ablations.py --seeds 2000 --events 200000 --epochs 15
"""

import argparse
import dataclasses

import numpy as np
from graphagate.report_metrics import mean_std

from graphagate.config import TGNConfig
from graphagate.train_tgn import train_tgn

# Seeds for the multi-seed report (mean ± std). Lateral AUC carries ~±0.03 single-run
# CUDA nondeterminism, so a single seed is not enough for an honest claim.
SEEDS = [42, 7, 123]
# Ablation runs use a slightly reduced stream/epochs so the 5×|SEEDS| runs finish in a
# reasonable time; the relative ordering is what matters here, not the absolute headline
# (the headline comes from the full-config run that saves the artifact).
ABLATION_EVENTS = 40000
ABLATION_EPOCHS = 12

VARIANTS = [
    ("full",            dict()),
    ("no hist feats",   dict(use_hist_feats=False)),
    ("no precursor",    dict(use_precursor=False)),
    ("no struct head",  dict(use_struct_head=False)),
    ("no hashed id",    dict(use_hash_identity=False)),
]


def _lat(metrics):
    p = metrics["per_type"].get("lateral", {})
    return p.get("auc", float("nan")), p.get("ap", float("nan")), p.get("recall", float("nan"))


def _theft_auc(metrics):
    return metrics["per_type"].get("cred-theft", {}).get("auc", float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--events", type=int, default=ABLATION_EVENTS)
    ap.add_argument("--epochs", type=int, default=ABLATION_EPOCHS)
    ap.add_argument("--variants", nargs="+", choices=[n for n, _ in VARIANTS],
                    default=[n for n, _ in VARIANTS])
    args = ap.parse_args()

    base = dataclasses.replace(TGNConfig(), num_events=args.events, epochs=args.epochs)
    # results[name] = list over seeds of (lat_auc, lat_ap, lat_rec, theft_auc, agg_auc)
    variants = [(n, f) for n, f in VARIANTS if n in args.variants]
    results = {name: [] for name, _ in variants}

    for seed in args.seeds:
        cfg = dataclasses.replace(base, seed=seed)
        for name, flags in variants:
            print("\n" + "=" * 78)
            print(f"=== ABLATION: {name}  (seed={seed}) ===")
            print("=" * 78)
            m = train_tgn(cfg, save=False, **flags)
            la, lp, lr = _lat(m)
            results[name].append((la, lp, lr, _theft_auc(m), m["agg_auc"]))

    # --- summary table (mean ± std over seeds) -------------------------------
    print("\n" + "=" * 90)
    print(f"ABLATION SUMMARY — {len(args.seeds)} seeds {args.seeds}, "
          f"{args.events} events / {args.epochs} epochs, FPR target {base.target_fpr:.0%}")
    print("=" * 90)
    header = (f"{'variant':16s} | {'lateral AUC':>15s} | {'lateral AP':>15s} | "
              f"{'lateral Rec@thr':>17s} | {'theft AUC':>15s} | {'agg AUC':>13s}")
    print(header)
    print("-" * len(header))

    def ms(vals):
        a = np.array(vals, dtype=float)
        m, sd = mean_std(a)
        return f"{m:.3f}±{sd:.3f}"

    for name, _ in variants:
        arr = np.array(results[name], dtype=float)  # [seeds, 5]
        print(f"{name:16s} | {ms(arr[:, 0]):>15s} | {ms(arr[:, 1]):>15s} | "
              f"{ms(arr[:, 2]):>17s} | {ms(arr[:, 3]):>15s} | {ms(arr[:, 4]):>13s}")

    print(
        "\nReading: 'lateral AUC' is the discriminating column (the only model-owned class). "
        "The drop from 'full' to each ablation is that component's contribution; report it "
        "relative to the across-seed std. Policy/contextual are omitted (OPA-owned / "
        "rule-trivial). The Static-GNN baseline (same history+precursor signals) isolates "
        "the temporal-graph contribution."
    )


if __name__ == "__main__":
    main()
