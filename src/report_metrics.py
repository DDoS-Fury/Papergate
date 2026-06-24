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
    """``(nanmean, nanstd)`` of a sequence — the multi-seed aggregation used everywhere.

    ``nan``-aware so a metric that is undefined for a given seed (e.g. a class with
    ``n=0`` in that seed's test split) does not poison the whole cell.
    """
    a = np.asarray(vals, dtype=float)
    return float(np.nanmean(a)), float(np.nanstd(a))


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


def delta(a_vals, b_vals, decimals: int = 3) -> tuple[float, bool]:
    """``(mean(a) - mean(b), significant)``.

    ``significant`` is ``True`` only when ``|Δ|`` exceeds the larger of the two
    across-seed std-devs — the "survives the noise band" test P1 asks for.
    """
    am, asd = mean_std(a_vals)
    bm, bsd = mean_std(b_vals)
    d = am - bm
    return round(d, decimals), abs(d) > max(asd, bsd)


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
