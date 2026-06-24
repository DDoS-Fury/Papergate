"""Fill the reporting gap: evaluate the **deployable** ``v4 + dev:guest`` on the
**standard** synthetic stream (the Panel B protocol), not just on theft-rich.

Why this exists
---------------
``run_guest_device_eval.py`` compares cookie-keying vs ``dev:guest`` on a *theft-rich*
stream (Panel C, ``tab:guestdev``) so credential-theft incidents land in the test split.
But the deployable artifact runs with ``guest_device_fallback=True`` (``dev:guest``), while
every **standard-stream** number in the report (Panels A/B, ``tab:baselines`` /
``tab:v3v4``) is **per-cookie** — produced before guest adoption. So the production config
has no row under the headline protocol.

This driver closes that gap: on the standard stream (default 200k ev / 15 ep, v4 config
node on), at parity of data and per seed, with ``save=False`` (the artifact in ``public/``
is never touched), it runs the **paired** comparison

    v4 per-cookie   (guest_device_fallback=False)   -- the published Panel B "v4" row
    v4 + dev:guest  (guest_device_fallback=True)     -- the deployable, the missing row

reporting the same metrics Panel B uses (routed lateral/aggregate recall, lateral AUC/AP,
routed benign FPR), multi-seed (mean ± std) to clear the ±0.01–0.03 CUDA noise band.

On the standard stream the theft slots exhaust inside the train split (cfr. report
sec:v4results), so theft is zero in test and not measured here — exactly as in Panel B;
the focus is lateral + aggregate + routed FPR.

    docker compose --profile guest-standard-eval up      # or:
    python tests/ablations/run_guest_standard_eval.py
"""

import dataclasses

import numpy as np

from graphagate.config import TGNConfig
from graphagate.train_tgn import train_tgn

SEEDS = [42, 7, 123]

# Standard stream = TGNConfig defaults (num_events=200000, epochs=15). v4 config node is
# train_tgn's default (use_config_node=True); we only flip the device-keying policy.
VARIANTS = [
    ("v4 per-cookie", dict(guest_device_fallback=False)),
    ("v4 + dev:guest", dict(guest_device_fallback=True)),
]


def _cfg(seed, guest_device_fallback):
    return dataclasses.replace(
        TGNConfig(), seed=seed, guest_device_fallback=guest_device_fallback,
    )


def _grab(m):
    """Panel-B metric vector (routed decision), in COLS order."""
    lat = m["per_type"].get("lateral", {})
    pol = m["per_type"].get("policy", {})
    ctx = m["per_type"].get("contextual", {})
    return (
        m.get("agg_auc", float("nan")), m.get("agg_ap", float("nan")),
        m.get("agg_recall", float("nan")),          # routed (operational)
        lat.get("auc", float("nan")), lat.get("ap", float("nan")),
        lat.get("recall", float("nan")),            # routed lateral recall
        m.get("lateral_recall_before", float("nan")),  # @1% FPR global threshold
        pol.get("auc", float("nan")), ctx.get("auc", float("nan")),
        m.get("fpr_after", float("nan")),           # routed benign FPR
        lat.get("n", 0),
    )


COLS = ["agg_auc", "agg_ap", "agg_rec", "lat_auc", "lat_ap", "lat_rec",
        "lat_rec_1fpr", "policy_auc", "ctx_auc", "fpr_routed", "lat_n"]


def _ms(vals):
    a = np.array(vals, dtype=float)
    return f"{np.nanmean(a):.3f}±{np.nanstd(a):.3f}"


def main():
    results = {name: [] for name, _ in VARIANTS}
    for seed in SEEDS:
        for name, flags in VARIANTS:
            print("\n" + "=" * 78)
            print(f"=== GUEST-STANDARD EVAL: {name}  (seed={seed}, standard stream) ===")
            print("=" * 78, flush=True)
            m = train_tgn(_cfg(seed, **flags), save=False)
            results[name].append(_grab(m))

    print("\n" + "=" * 120)
    cfg0 = TGNConfig()
    print(f"GUEST-STANDARD EXPERIMENT — {len(SEEDS)} seeds {SEEDS}, {cfg0.num_events} events / "
          f"{cfg0.epochs} epochs, standard stream, cost-sensitive routed decision (save=False)")
    print("=" * 120)
    header = (f"{'variant':16s} | {'agg AUC':>13s} | {'agg AP':>13s} | {'agg Rec':>13s} | "
              f"{'lat AUC':>13s} | {'lat AP':>13s} | {'lat Rec':>13s} | "
              f"{'lat R@1%FPR':>13s} | {'FPR benign':>13s}")
    print(header)
    print("-" * len(header))

    agg = {}
    for name, _ in VARIANTS:
        arr = np.array(results[name], dtype=float)  # [seeds, len(COLS)]
        agg[name] = arr
        lat_n = int(np.nansum(arr[:, 10]))
        print(f"{name:16s} | {_ms(arr[:, 0]):>13s} | {_ms(arr[:, 1]):>13s} | "
              f"{_ms(arr[:, 2]):>13s} | {_ms(arr[:, 3]):>13s} | {_ms(arr[:, 4]):>13s} | "
              f"{_ms(arr[:, 5]):>13s} | {_ms(arr[:, 6]):>13s} | {_ms(arr[:, 9]):>13s}  "
              f"[lat_n={lat_n}]")

    cookie, guest = agg[VARIANTS[0][0]], agg[VARIANTS[1][0]]

    def _d(col):
        return np.nanmean(guest[:, col]) - np.nanmean(cookie[:, col])

    print(
        f"\nΔ (dev:guest − cookie, standard stream):  agg AUC {_d(0):+.3f} | "
        f"lateral AUC {_d(3):+.3f} | lateral AP {_d(4):+.3f} | lateral recall {_d(5):+.3f} | "
        f"agg recall {_d(2):+.3f} | benign FPR {_d(9):+.3f}."
    )
    print(
        "Reading: this is the deployable (dev:guest) under the SAME protocol as the "
        "published Panel B 'v4' row (per-cookie). A flat/positive Δ on lateral with a "
        "lower benign FPR means adopting dev:guest cost nothing (or helped) on the "
        "standard stream too — closing the gap left by tab:guestdev being theft-rich only."
    )
    print("\nDONE_GUEST_STANDARD_EVAL")


if __name__ == "__main__":
    main()
