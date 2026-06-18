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

import dataclasses

import numpy as np

from graphagate.config import TGNConfig
from graphagate.train_tgn import train_tgn

SEEDS = [42, 7, 123]
EVENTS = 40000
EPOCHS = 12

# Ordered per the user's request: the MLP layer first, then memory + heads.
VARIANTS = [
    ("baseline v4",        dict()),
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
    return f"{np.nanmean(a):.3f}±{np.nanstd(a):.3f}"


def main():
    base = dataclasses.replace(TGNConfig(), num_events=EVENTS, epochs=EPOCHS)
    results = {name: [] for name, _ in VARIANTS}

    for seed in SEEDS:
        for name, over in VARIANTS:
            cfg = dataclasses.replace(base, seed=seed, **over)
            print("\n" + "=" * 78)
            print(f"=== ARCH-SWEEP: {name}  (seed={seed}) ===")
            print("=" * 78)
            m = train_tgn(cfg, save=False)
            la, lp, lr = _lat(m)
            results[name].append((la, lp, lr, m["agg_auc"]))

    print("\n" + "=" * 92)
    print(f"ARCH SWEEP SUMMARY — {len(SEEDS)} seeds {SEEDS}, {EVENTS} events / {EPOCHS} epochs")
    print("=" * 92)
    header = (f"{'variant':18s} | {'lateral AUC':>15s} | {'lateral AP':>15s} | "
              f"{'lateral Rec@thr':>17s} | {'agg AUC':>13s}")
    print(header)
    print("-" * len(header))
    base_lat = None
    for name, _ in VARIANTS:
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
