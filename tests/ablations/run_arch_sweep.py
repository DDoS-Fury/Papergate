"""Architecture sweep for the v4 model — does extra capacity help the (model-owned)
lateral-movement class?

Given the new configuration node, two capacity bumps are worth evaluating:
  * an extra hidden layer in the feature-head MLP (``link_pred_hidden_layers``);
  * a wider recurrent memory (``memory_dim``) and/or more GNN attention heads
    (``gnn_heads``).

Each variant runs the full train+eval pipeline with ``save=False`` (deployable in
``public/`` untouched), multi-seed (mean ± std over CUDA noise), reduced stream/epochs
(relative ordering is the point, not the absolute headline — the headline comes from the
full 200k/15ep run that saves the artifact). Adopt a variant only if its lateral metrics
improve beyond the across-seed std.

    docker compose --profile arch-sweep up
"""

import argparse
import dataclasses

import numpy as np
from graphagate.report_metrics import mean_std

from graphagate.config import TGNConfig
from graphagate.train_tgn import train_tgn

SEEDS = [42, 7, 123]
_CFG_DEFAULT = TGNConfig()
DEFAULT_EVENTS = _CFG_DEFAULT.num_events  # 200000
DEFAULT_EPOCHS = _CFG_DEFAULT.epochs      # 15

# Architecture sweep order: MLP depth, then memory dimension and attention heads.
VARIANTS = [
    ("baseline",           dict()),
    ("+1 MLP layer",       dict(link_pred_hidden_layers=3)),
    ("+memory (384)",      dict(memory_dim=384)),
    ("+heads (8)",         dict(gnn_heads=8)),
    ("+mem+heads+layer",   dict(memory_dim=384, gnn_heads=8, link_pred_hidden_layers=3)),
]


def _lat(m):
    p = m["per_type"].get("lateral", {})
    return p.get("auc", float("nan")), p.get("ap", float("nan")), p.get("recall", float("nan"))


def _ms(vals):
    a = np.array(vals, dtype=float)
    m, sd = mean_std(a)
    return f"{m:.3f}±{sd:.3f}"


def main():
    ap = argparse.ArgumentParser(description="Architecture sweep for the TGN model")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS, help="Seeds for multi-seed run")
    ap.add_argument("--events", type=int, default=DEFAULT_EVENTS, help="Number of stream events")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs")
    ap.add_argument("--variants", nargs="+", choices=[n for n, _ in VARIANTS],
                    default=[n for n, _ in VARIANTS], help="Variants to evaluate")
    args = ap.parse_args()

    base = dataclasses.replace(TGNConfig(), num_events=args.events, epochs=args.epochs)
    selected_variants = [(n, f) for n, f in VARIANTS if n in args.variants]
    results = {name: [] for name, _ in selected_variants}

    for seed in args.seeds:
        for name, over in selected_variants:
            cfg = dataclasses.replace(base, seed=seed, **over)
            print("\n" + "=" * 78)
            print(f"=== ARCH-SWEEP: {name}  (seed={seed}, events={args.events}, epochs={args.epochs}) ===")
            print("=" * 78)
            m = train_tgn(cfg, save=False)
            la, lp, lr = _lat(m)
            results[name].append((la, lp, lr, m["agg_auc"]))

    print("\n" + "=" * 92)
    print(f"ARCH SWEEP SUMMARY — {len(args.seeds)} seeds {args.seeds}, {args.events} events / {args.epochs} epochs")
    print("=" * 92)
    header = (f"{'variant':18s} | {'lateral AUC':>15s} | {'lateral AP':>15s} | "
              f"{'lateral Rec@thr':>17s} | {'agg AUC':>13s}")
    print(header)
    print("-" * len(header))
    base_lat = None
    for name, _ in selected_variants:
        arr = np.array(results[name], dtype=float)
        if base_lat is None:
            base_lat = np.nanmean(arr[:, 0])
        delta = np.nanmean(arr[:, 0]) - base_lat
        print(f"{name:18s} | {_ms(arr[:, 0]):>15s} | {_ms(arr[:, 1]):>15s} | "
              f"{_ms(arr[:, 2]):>17s} | {_ms(arr[:, 3]):>13s}  (ΔlatAUC {delta:+.3f})")

    print(
        "\nReading: adopt a variant only if its lateral AUC gain exceeds the across-seed std "
        "(otherwise it is run-to-run noise, not capacity). The chosen variant, if any, is then "
        "retrained at full 200k/15ep and saved to public/."
    )


if __name__ == "__main__":
    main()
