"""Experiment: collapse every non-TPM device onto a single shared ``dev:guest`` node.

Mirror of the ``conf:guest`` design, gated by ``SyntheticConfig.guest_device_fallback``.
Instead of keying each TPM-less machine by its own persistent cookie (``ck:<id>``), all
non-TPM devices (benign machines AND the credential-theft attacker, which is TPM-less
too) map to ONE anonymous low-trust device node. The device layer then no longer
distinguishes individual cookie-keyed machines; any remaining signal must come from the
user / config / source edges.

This driver compares — at parity of data, per seed, with ``save=False`` (the deployable
artifact in ``public/`` is never touched) — the cookie baseline against the guest
fallback, on a **theft-rich** stream (so credential-theft incidents, where the attacker
device collapses to guest, land in the test split and are measurable). Multi-seed
(mean ± std) because lateral/theft AUC carries ~±0.01–0.03 single-run CUDA
nondeterminism (cfr. tasks/lessons.md).

    docker compose --profile guest-device-eval up      # if wired in compose, or:
    python tests/ablations/run_guest_device_eval.py
"""

import dataclasses

import numpy as np

from graphagate.config import TGNConfig
from graphagate.train_tgn import train_tgn

SEEDS = [42, 7, 123]
EVENTS = 80000
EPOCHS = 12

# Theft-rich knobs (identical to run_config_eval): slots >> expected incidents so they
# never exhaust; rate high enough that ~20% of incidents land in the test split.
THEFT_SLOTS = 400
WIPE_SLOTS = 120
P_CRED_THEFT = 0.002
P_COOKIE_WIPE = 0.0008

VARIANTS = [
    ("cookie (baseline)", dict(guest_device_fallback=False)),
    ("dev:guest", dict(guest_device_fallback=True)),
]


def _cfg(seed, guest_device_fallback):
    return dataclasses.replace(
        TGNConfig(),
        num_events=EVENTS, epochs=EPOCHS, seed=seed,
        num_theft_slots=THEFT_SLOTS, num_wipe_slots=WIPE_SLOTS,
        p_cred_theft=P_CRED_THEFT, p_cookie_wipe=P_COOKIE_WIPE,
        guest_device_fallback=guest_device_fallback,
    )


def _grab(m):
    th = m["per_type"].get("cred-theft", {})
    lat = m["per_type"].get("lateral", {})
    sc = m.get("scenario", {})
    return (
        lat.get("auc", float("nan")), lat.get("ap", float("nan")),
        lat.get("recall", float("nan")), lat.get("n", 0),
        th.get("auc", float("nan")), th.get("recall", float("nan")), th.get("n", 0),
        m.get("agg_auc", float("nan")), m.get("agg_recall", float("nan")),
        m.get("fpr_after", float("nan")),
        sc.get("fpr_wiped", float("nan")), sc.get("n_wiped", 0),
    )


def _ms(vals):
    a = np.array(vals, dtype=float)
    return f"{np.nanmean(a):.3f}±{np.nanstd(a):.3f}"


COLS = ["lat_auc", "lat_ap", "lat_rec", "lat_n", "theft_auc", "theft_rec", "theft_n",
        "agg_auc", "agg_rec", "fpr_after", "fpr_wiped", "n_wiped"]


def main():
    results = {name: [] for name, _ in VARIANTS}
    for seed in SEEDS:
        for name, flags in VARIANTS:
            print("\n" + "=" * 78)
            print(f"=== GUEST-DEVICE EVAL: {name}  (seed={seed}, theft-rich) ===")
            print("=" * 78)
            m = train_tgn(_cfg(seed, **flags), save=False)
            results[name].append(_grab(m))

    print("\n" + "=" * 110)
    print(f"GUEST-DEVICE EXPERIMENT — {len(SEEDS)} seeds {SEEDS}, {EVENTS} events / "
          f"{EPOCHS} epochs, theft-rich (slots {THEFT_SLOTS}/{WIPE_SLOTS}, "
          f"p_theft={P_CRED_THEFT}, p_wipe={P_COOKIE_WIPE})")
    print("=" * 110)
    header = (f"{'variant':18s} | {'lat AUC':>13s} | {'lat AP':>13s} | {'lat Rec':>13s} | "
              f"{'theft AUC':>13s} | {'theft Rec':>13s} | {'agg AUC':>13s} | "
              f"{'agg Rec':>13s} | {'FPR benign':>13s} | {'FPR wiped':>13s}")
    print(header)
    print("-" * len(header))

    agg = {}
    for name, _ in VARIANTS:
        arr = np.array(results[name], dtype=float)  # [seeds, len(COLS)]
        agg[name] = arr
        lat_n = int(np.nansum(arr[:, 3]))
        theft_n = int(np.nansum(arr[:, 6]))
        n_wiped = int(np.nansum(arr[:, 11]))
        print(f"{name:18s} | {_ms(arr[:, 0]):>13s} | {_ms(arr[:, 1]):>13s} | "
              f"{_ms(arr[:, 2]):>13s} | {_ms(arr[:, 4]):>13s} | {_ms(arr[:, 5]):>13s} | "
              f"{_ms(arr[:, 7]):>13s} | {_ms(arr[:, 8]):>13s} | {_ms(arr[:, 9]):>13s} | "
              f"{_ms(arr[:, 10]):>13s} (n={n_wiped})  [lat_n={lat_n}, theft_n={theft_n}]")

    base, guest = agg[VARIANTS[0][0]], agg[VARIANTS[1][0]]

    def _d(col):
        return np.nanmean(guest[:, col]) - np.nanmean(base[:, col])

    print(
        f"\nΔ (dev:guest − cookie):  lateral AUC {_d(0):+.3f} | lateral recall {_d(2):+.3f} | "
        f"theft AUC {_d(4):+.3f} | theft recall {_d(5):+.3f} | agg AUC {_d(7):+.3f} | "
        f"benign FPR {_d(9):+.3f}."
    )
    print(
        "Reading: collapsing TPM-less devices onto one guest node removes the per-machine "
        "device-identity tell (the attacker device also becomes guest); a NEGATIVE Δ on "
        "theft/lateral means that identity mattered, a flat Δ means config/source edges "
        "already carried the signal. n_wiped should be 0 under dev:guest (wipe neutralised)."
    )


if __name__ == "__main__":
    main()
