"""Shared helpers that make the LaTeX report a reproducible artifact.

Rationale (tasks/report-improvements.md, P2): report tables were hand-copied from
stdout, which is the root of the metric drift. The training entrypoint
(:func:`graphagate.train_tgn.train_tgn`) and every baseline already *compute* their
metrics as a dict; this module centralises (a) the multi-seed aggregation
(mean ± std), (b) the LaTeX cell formatting used by ``tab:theft`` / ``tab:archsweep``,
and (c) atomic JSON serialisation, so ``tests/regen_report_tables.py`` can rebuild the
Panel A / Panel B tables from fixed seeds instead of by hand.

No training logic lives here — only formatting/IO — so it is import-cheap and safe to
reuse from both the package and the test drivers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def mean_std(vals) -> tuple[float, float]:
    """``(nanmean, sample nanstd)`` — the multi-seed aggregation used everywhere.

    ``nan``-aware so a metric that is undefined for a given seed (e.g. a class with
    ``n=0`` in that seed's test split) does not poison the whole cell.

    Uses ``ddof=1`` (sample standard deviation). numpy's default of ``ddof=0`` is the
    *population* std and understates the sample std by ``sqrt((n-1)/n)`` — about 18% at
    the n=3 seeds these tables are built from, i.e. every published error bar was that
    much too tight.
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
    return rf"${body}\pm{s:.{decimals}f}$"


def paired_delta(a_vals, b_vals, decimals: int = 3, alpha: float = 0.05) -> dict:
    """Paired comparison of two arms measured on the SAME seeds.

    Every driver in ``tests/ablations/`` runs both arms under an identical seed, so the
    observations are paired and the analysis should use that: the per-seed differences
    cancel the seed-to-seed variance that dominates the raw spread.

    Returns ``mean_delta``, ``sd_delta``, ``ci95`` (bootstrap on the per-seed
    differences), ``p_value`` (Wilcoxon signed-rank, two-sided) and ``significant``.

    Note on power: the Wilcoxon signed-rank test cannot produce p < 0.05 with fewer than
    6 pairs, so at the current 3 seeds ``significant`` is essentially always ``False``.
    That is not a defect of the test — it is the honest statement that 3 seeds cannot
    establish these effects, and the reason the experimental grid needs more seeds and
    repeated runs per seed.
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
    except Exception:
        # No scipy: fall back to the bootstrap CI excluding zero.
        out["significant"] = bool(lo > 0.0 or hi < 0.0)
    return out


def delta(a_vals, b_vals, decimals: int = 3) -> tuple[float, bool]:
    """``(mean(a) - mean(b), significant)`` — thin wrapper over :func:`paired_delta`.

    Kept for the existing call sites. ``significant`` now comes from the paired test, not
    from the old ``|Δ| > max(sd_a, sd_b)`` rule, which was not a test at all: it compared
    a difference of means against a *single-arm* standard deviation, and discarded the
    seed pairing that every driver already establishes.
    """
    res = paired_delta(a_vals, b_vals, decimals=decimals)
    return res["mean_delta"], res["significant"]


def _json_default(o):
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
