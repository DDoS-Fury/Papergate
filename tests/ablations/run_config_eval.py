"""Targeted validation of the v4 *configuration* (JA3) node — the credential-theft case.

In the default config the attacker slots exhaust early (``num_theft_slots=64`` /
``p_cred_theft=0.0012`` ⇒ all incidents start before ~event 53k, inside the 70% train
split), so cred-theft (and cookie-wipe) have **n=0** in the test window and the config
node's expected benefit on credential theft is unmeasured. This driver runs a
**theft-rich** stream (many attacker/wipe slots + higher incident rate) so incidents
spread across the WHOLE stream including the test split, then compares — at parity of
data, per seed — the full v4 model against a **config-node ablation** (``use_config_node
=False``: the model falls back to the legacy source→device chain, i.e. ≈v3). The Δ on
cred-theft AUC/recall isolates the config node's contribution (in particular the
``config→user`` binding: a stolen-credential client differs from the victim's habitual
config).

All runs use ``save=False`` — the deployable artifact in ``public/`` is never touched.
The high theft/wipe slot counts here are eval-only; the published default stays 64/16.

Multi-seed (mean ± std) because lateral/theft AUC carries ~±0.01–0.03 single-run CUDA
nondeterminism (cfr. tasks/lessons.md). Reduced stream/epochs so 2×|SEEDS| runs finish
in reasonable time; relative ordering (the Δ) is what matters, not the absolute headline.

    docker compose --profile config-eval up
"""

import dataclasses

import numpy as np

from graphagate.config import TGNConfig
from graphagate.train_tgn import train_tgn

SEEDS = [42, 7, 123]
EVENTS = 80000
EPOCHS = 12

# Theft-rich knobs: slots >> expected incidents so they never exhaust; rate high enough
# that ~20% of incidents land in the test split with a healthy positive count.
THEFT_SLOTS = 400
WIPE_SLOTS = 120
P_CRED_THEFT = 0.002
P_COOKIE_WIPE = 0.0008

VARIANTS = [
    ("v4 (config node)", dict()),
    ("~v3 (no config)", dict(use_config_node=False)),
]


def _cfg(seed):
    return dataclasses.replace(
        TGNConfig(),
        num_events=EVENTS, epochs=EPOCHS, seed=seed,
        num_theft_slots=THEFT_SLOTS, num_wipe_slots=WIPE_SLOTS,
        p_cred_theft=P_CRED_THEFT, p_cookie_wipe=P_COOKIE_WIPE,
    )


def _grab(m):
    th = m["per_type"].get("cred-theft", {})
    lat = m["per_type"].get("lateral", {})
    sc = m.get("scenario", {})
    return (
        th.get("auc", float("nan")), th.get("recall", float("nan")), th.get("n", 0),
        lat.get("auc", float("nan")), lat.get("recall", float("nan")),
        m["agg_auc"], sc.get("fpr_wiped", float("nan")), sc.get("n_wiped", 0),
    )


def _ms(vals):
    a = np.array(vals, dtype=float)
    return f"{np.nanmean(a):.3f}±{np.nanstd(a):.3f}"


def main():
    results = {name: [] for name, _ in VARIANTS}
    for seed in SEEDS:
        cfg = _cfg(seed)
        for name, flags in VARIANTS:
            print("\n" + "=" * 78)
            print(f"=== CONFIG-EVAL: {name}  (seed={seed}, theft-rich) ===")
            print("=" * 78)
            m = train_tgn(cfg, save=False, **flags)
            results[name].append(_grab(m))

    cols = ["theft_auc", "theft_rec", "theft_n", "lat_auc", "lat_rec",
            "agg_auc", "fpr_wiped", "n_wiped"]
    print("\n" + "=" * 96)
    print(f"CONFIG-NODE TARGETED VALIDATION — {len(SEEDS)} seeds {SEEDS}, "
          f"{EVENTS} events / {EPOCHS} epochs, theft-rich "
          f"(slots {THEFT_SLOTS}/{WIPE_SLOTS}, p_theft={P_CRED_THEFT}, p_wipe={P_COOKIE_WIPE})")
    print("=" * 96)
    header = (f"{'variant':18s} | {'theft AUC':>13s} | {'theft Rec':>13s} | {'theft n':>8s} | "
              f"{'lat AUC':>13s} | {'lat Rec':>13s} | {'agg AUC':>13s} | {'FPR wiped':>13s}")
    print(header)
    print("-" * len(header))

    agg = {}
    for name, _ in VARIANTS:
        arr = np.array(results[name], dtype=float)  # [seeds, len(cols)]
        agg[name] = arr
        n_theft_total = int(np.nansum(arr[:, 2]))
        n_wiped_total = int(np.nansum(arr[:, 7]))
        print(f"{name:18s} | {_ms(arr[:, 0]):>13s} | {_ms(arr[:, 1]):>13s} | "
              f"{n_theft_total:>8d} | {_ms(arr[:, 3]):>13s} | {_ms(arr[:, 4]):>13s} | "
              f"{_ms(arr[:, 5]):>13s} | {_ms(arr[:, 6]):>13s} (n={n_wiped_total})")

    # Δ (config node contribution) on the credential-theft class.
    full, abl = agg[VARIANTS[0][0]], agg[VARIANTS[1][0]]
    d_auc = np.nanmean(full[:, 0]) - np.nanmean(abl[:, 0])
    d_rec = np.nanmean(full[:, 1]) - np.nanmean(abl[:, 1])
    print(
        f"\nΔ config node (v4 − ~v3) on credential theft: AUC {d_auc:+.3f} | recall {d_rec:+.3f}. "
        "Positive Δ ⇒ the config node (config→user / source→config) adds detection power on "
        "stolen-credential clients. theft_n>0 here confirms the targeted split exposes the "
        "incidents the default 64-slot config hid in the train window."
    )


if __name__ == "__main__":
    main()
