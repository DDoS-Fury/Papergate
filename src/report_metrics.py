"""Report helpers: multi-seed aggregation, paired comparisons, LaTeX cells, atomic JSON.

Formatting and IO only (no training logic), so drivers can import it cheaply.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def mean_std(vals) -> tuple[float, float]:
    """``(nanmean, sample nanstd)`` — the multi-seed aggregation used everywhere.

    ``nan``-aware, so a metric undefined for one seed does not poison the cell;
    ``ddof=1`` (sample std).
    """
    a = np.asarray(vals, dtype=float)
    n = int(np.count_nonzero(~np.isnan(a)))
    if n < 2:
        return (float(np.nanmean(a)) if n else float("nan")), float("nan")
    return float(np.nanmean(a)), float(np.nanstd(a, ddof=1))


def fmt_ms(vals, decimals: int = 3) -> str:
    """Plain-text ``mean±std`` cell (the format the existing drivers print)."""
    m, s = mean_std(vals)
    return f"{m:.{decimals}f}±{s:.{decimals}f}"


def latex_cell(vals, *, bold: bool = False, decimals: int = 3) -> str:
    """LaTeX math cell ``$mean\\pm std$`` matching ``tab:theft`` / ``tab:archsweep``.

    ``bold=True`` wraps the mean in ``\\mathbf{}`` for the row-winner convention.
    """
    m, s = mean_std(vals)
    body = f"{m:.{decimals}f}"
    if bold:
        body = rf"\mathbf{{{body}}}"
    if np.isnan(s):  # fewer than 2 valid seeds: no spread to report (``\pmnan`` does not compile)
        return f"${body}$"
    return rf"${body}\pm{s:.{decimals}f}$"


def paired_delta(a_vals, b_vals, decimals: int = 3, alpha: float = 0.05) -> dict:
    """Paired comparison of two arms measured on the SAME seeds.

    Per-seed differences cancel the seed-to-seed variance. Returns ``n_pairs``,
    ``mean_delta``, ``sd_delta``, ``ci95`` (bootstrap), ``p_value`` (two-sided Wilcoxon
    signed-rank) and ``significant``. Wilcoxon needs >= 6 pairs to reach p < 0.05.
    """
    a = np.asarray(a_vals, dtype=float)
    b = np.asarray(b_vals, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired_delta needs matched arms, got {a.shape} vs {b.shape}")

    ok = ~(np.isnan(a) | np.isnan(b))
    diffs = a[ok] - b[ok]
    n = diffs.size
    out = {
        "n_pairs": int(n),
        "mean_delta": round(float(np.mean(diffs)), decimals) if n else float("nan"),
        "sd_delta": round(float(np.std(diffs, ddof=1)), decimals) if n > 1 else float("nan"),
        "ci95": (float("nan"), float("nan")),
        "p_value": float("nan"),
        "significant": False,
    }
    if n < 2:
        return out

    rng = np.random.default_rng(0)  # fixed: the CI must not move between report builds
    boot = rng.choice(diffs, size=(10_000, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    out["ci95"] = (round(float(lo), decimals), round(float(hi), decimals))

    if np.allclose(diffs, 0.0):
        out["p_value"] = 1.0
        return out
    try:
        from scipy.stats import wilcoxon

        out["p_value"] = float(wilcoxon(diffs).pvalue)
        out["significant"] = bool(out["p_value"] < alpha)
    except ImportError:
        # No scipy: fall back to the bootstrap CI excluding zero.
        out["significant"] = bool(lo > 0.0 or hi < 0.0)
    return out


def _json_default(o):
    """``json.dump`` fallback for numpy scalars and arrays."""
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serialisable: {type(o)!r}")


def dump_json(payload, path) -> Path:
    """Atomically write ``payload`` as indented JSON (``.tmp`` + ``os.replace``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    os.replace(tmp, path)
    return path
