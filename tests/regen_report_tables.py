"""Single orchestrator that regenerates Panel A (``tab:baselines``) and Panel B
(``tab:v3v4``) of the report from fixed seeds — the reproducible-artifact step
(tasks/report-improvements.md, P1 + P2).

Why this exists
---------------
The two headline panels were single-run (seed 42) numbers copied by hand from stdout,
so several reported Δ sat *inside* the ±0.01–0.03 CUDA noise band and the narrative
drifted (see memory ``report-tables-historical-runs``). This driver runs every cell
**multi-seed** and writes machine-readable JSON + LaTeX fragments, so the tables can be
rebuilt deterministically and each number cites the run that produced it.

What it runs (all ``save=False`` — the deployable artifact in ``public/`` is untouched)
----------------------------------------------------------------------------------------
* **Panel B** — TGN, *per-cookie* keying (``guest_device_fallback=False``), the standard
  200k/15ep stream, ``use_config_node`` on (v4) vs off (≈v3). 3 seeds × 2 = 6 runs.
* **Panel A** — same standard stream under the *deployable* config (``dev:guest``):
  the GNN / One-Class SVM / Isolation Forest / XGBoost baselines, 3 seeds each. The
  **TGN row** of Panel A is reused from Panel B's *v3 per-cookie* column (the historical
  provenance: TGN row = v3 per-cookie, baselines = deployable — kept as-is here; unifying
  the protocol is P3, deferred).

Outputs (under the component root)
----------------------------------
* ``tasks/runs/panelA.json`` / ``panelB.json`` — per-seed dicts **and** mean±std cells.
* ``docs/latex/generated/tab_baselines.tex`` / ``tab_v3v4.tex`` — ready table-body rows.

Run on the GPU box::

    docker compose --profile regen-report up
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import importlib.util
import math
from pathlib import Path

import numpy as np

from graphagate.config import REPO_ROOT, TGNConfig
from graphagate.train_tgn import train_tgn
from graphagate.report_metrics import dump_json, fmt_ms, latex_cell, mean_std

SEEDS = [42, 7, 123]
TESTS_DIR = Path(__file__).resolve().parent
RUNS_DIR = REPO_ROOT / "tasks" / "runs"
GEN_DIR = REPO_ROOT / "docs" / "latex" / "generated"

NAN = float("nan")


# --- baseline loading (the scripts live outside the package, under tests/baselines) ----
def _load(rel_path: str, attr: str):
    """Import a baseline entrypoint function by file path."""
    p = TESTS_DIR / rel_path
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return getattr(mod, attr)


BASELINES = {
    # display name -> (loader, label used in tab:baselines)
    "gnn": ("baselines/simple_gnn/simple_gnn_baseline.py", "run", "GNN non temp."),
    "ocsvm": ("baselines/ocsvm/ocsvm_baseline.py", "ocsvm_baseline", "One-Class SVM"),
    "iforest": ("baselines/isolation_forest/isolation_forest_baseline.py",
                "isolation_forest_baseline", "Isolation Forest"),
    "xgboost": ("baselines/xgboost/xgboost_baseline.py", "xgboost_baseline", "XGBoost (sup.)"),
}


# --- metric extraction -----------------------------------------------------------------
def _g(d: dict, *path, default=NAN):
    """Navigate a nested metrics dict; missing keys -> ``default`` (NaN)."""
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


def _summarise(per_seed: list[dict], cells: dict[str, tuple]) -> dict:
    """Turn a list of per-seed run dicts into ``{cell: {mean,std,vals}}``.

    ``cells`` maps a cell name to the dotted path into each run dict.
    """
    out = {}
    for name, path in cells.items():
        vals = [float(_g(m, *path)) for m in per_seed]
        m, s = mean_std(vals)
        out[name] = {"mean": m, "std": s, "vals": vals}
    return out


# Panel B cells (TGN dicts): keyed by the metric the tab:v3v4 row needs.
PANEL_B_CELLS = {
    "agg_auc": ("agg_auc",),
    "agg_ap": ("agg_ap",),
    "agg_recall_routed": ("agg_recall",),          # routed (operational)
    "lat_auc": ("per_type", "lateral", "auc"),
    "lat_ap": ("per_type", "lateral", "ap"),
    "lat_recall_routed": ("per_type", "lateral", "recall"),
    "lat_recall_global": ("lateral_recall_before",),  # @1% FPR global threshold
    "policy_auc": ("per_type", "policy", "auc"),
    "policy_recall": ("per_type", "policy", "recall"),
    "ctx_auc": ("per_type", "contextual", "auc"),
    "ctx_recall": ("per_type", "contextual", "recall"),
    "fpr_routed": ("fpr_after",),
}

# Panel A — TGN row (reuse v3 per-cookie dicts): all @global-1%FPR for apples-to-apples.
PANEL_A_TGN_CELLS = {
    "agg_auc": ("agg_auc",),
    "agg_ap": ("agg_ap",),
    "lat_auc": ("per_type", "lateral", "auc"),
    "lat_recall_global": ("lateral_recall_before",),
    "agg_recall_global": ("agg_recall_global",),
}

# Panel A — baseline rows (baseline dicts already report @global-threshold recall).
PANEL_A_BASE_CELLS = {
    "agg_auc": ("agg_auc",),
    "agg_ap": ("agg_ap",),
    "lat_auc": ("per_type", "lateral", "auc"),
    "lat_recall_global": ("per_type", "lateral", "recall"),
    "agg_recall_global": ("agg_recall",),
}


# --- runners ---------------------------------------------------------------------------
def run_panel_b(seeds: list[int]) -> dict:
    """TGN v4 vs ≈v3, per-cookie, standard 200k/15ep stream. 2 × |seeds| runs."""
    variants = [("v4", dict(use_config_node=True)), ("v3", dict(use_config_node=False))]
    raw = {name: [] for name, _ in variants}
    for seed in seeds:
        cfg = dataclasses.replace(TGNConfig(), seed=seed, guest_device_fallback=False)
        for name, flags in variants:
            print("\n" + "=" * 80)
            print(f"=== PANEL B (tab:v3v4): {name} | seed={seed} | per-cookie, "
                  f"{cfg.num_events} ev / {cfg.epochs} ep ===")
            print("=" * 80, flush=True)
            raw[name].append(train_tgn(cfg, save=False, **flags))
    summary = {name: _summarise(raw[name], PANEL_B_CELLS) for name, _ in variants}
    payload = {
        "meta": _meta(seeds, "per-cookie (guest_device_fallback=False), routed decision"),
        "raw": raw, "summary": summary,
    }
    dump_json(payload, RUNS_DIR / "panelB.json")
    _emit_panel_b_tex(summary)
    return payload


def run_panel_a_baselines(seeds: list[int]) -> dict:
    """GNN / OC-SVM / Isolation Forest / XGBoost on the deployable stream, |seeds| runs each."""
    raw = {}
    for key, (rel, attr, _label) in BASELINES.items():
        fn = _load(rel, attr)
        raw[key] = []
        for seed in seeds:
            cfg = dataclasses.replace(TGNConfig(), seed=seed)  # deployable: dev:guest, v4
            print("\n" + "=" * 80)
            print(f"=== PANEL A baseline: {key} | seed={seed} | deployable "
                  f"({cfg.num_events} ev) ===")
            print("=" * 80, flush=True)
            raw[key].append(fn(cfg))
    summary = {k: _summarise(raw[k], PANEL_A_BASE_CELLS) for k in raw}
    return {"raw": raw, "summary": summary}


def _meta(seeds, protocol):
    return {
        "seeds": seeds, "num_events": TGNConfig().num_events, "epochs": TGNConfig().epochs,
        "protocol": protocol, "save": False,
        "generated": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }


# --- LaTeX fragment emission -----------------------------------------------------------
def _cell(summary: dict, key: str, *, bold=False) -> str:
    s = summary[key]
    return latex_cell(s["vals"], bold=bold)


def _delta(a: dict, b: dict, key: str) -> str:
    """Signed Δ (mean a − mean b) with a significance marker vs the noise band."""
    am, asd = a[key]["mean"], a[key]["std"]
    bm, bsd = b[key]["mean"], b[key]["std"]
    d = am - bm
    if any(math.isnan(x) for x in (am, bm)):
        return "---"
    sig = "" if abs(d) > max(asd, bsd) else r"$^{\dagger}$"  # dagger = within noise band
    return f"${d:+.3f}${sig}"


def _emit_panel_b_tex(summary: dict):
    v4, v3 = summary["v4"], summary["v3"]
    rows = [
        ("AUC aggregata", "agg_auc"),
        ("AP aggregata", "agg_ap"),
        ("Recall aggregata (instradata)", "agg_recall_routed"),
        ("Lateral --- AUC", "lat_auc"),
        ("Lateral --- AP", "lat_ap"),
        ("Lateral --- Recall (instradata)", "lat_recall_routed"),
        ("Lateral --- Recall (@1\\%FPR)", "lat_recall_global"),
        ("Policy --- AUC", "policy_auc"),
        ("Contextual --- AUC", "ctx_auc"),
        ("FPR benigno (instradato)", "fpr_routed"),
    ]
    lines = ["% Auto-generato da tests/regen_report_tables.py — NON modificare a mano.",
             "% Sorgente numerica: tasks/runs/panelB.json",
             "% Colonne: Metrica & v3 (4 nodi) & v4 (nodo config) & $\\Delta$ (v4-v3)"]
    for label, key in rows:
        lines.append(f"{label} & {_cell(v3, key)} & {_cell(v4, key)} & "
                     f"{_delta(v4, v3, key)} \\\\")
    lines.append(r"% $^{\dagger}$ = $|\Delta|$ entro la banda di rumore multi-seed "
                 r"(non significativo).")
    (GEN_DIR / "tab_v3v4.tex").parent.mkdir(parents=True, exist_ok=True)
    (GEN_DIR / "tab_v3v4.tex").write_text("\n".join(lines) + "\n")


def _emit_panel_a_tex(tgn: dict, base_summary: dict):
    # Columns of tab:baselines are MODELS; rows are METRICS. Emit one column block per model.
    metric_rows = [
        ("AUC aggregata", "agg_auc"),
        ("AP aggregata", "agg_ap"),
        ("Lateral --- AUC", "lat_auc"),
        ("Lateral --- Recall @1\\%FPR", "lat_recall_global"),
        ("Recall aggregata @1\\%FPR", "agg_recall_global"),
    ]
    models = [("TGN (v3, per-cookie)", tgn)]
    for key, (_rel, _attr, label) in BASELINES.items():
        if key == "xgboost":
            continue  # supervised upper-bound: emit last, separately flagged
        models.append((label, base_summary[key]))
    models.append(("XGBoost (sup.)$^{\\ddagger}$", base_summary["xgboost"]))

    lines = ["% Auto-generato da tests/regen_report_tables.py — NON modificare a mano.",
             "% Sorgente numerica: tasks/runs/panelA.json",
             "% Protocollo MISTO (storico): riga TGN = v3 per-cookie; baseline = deployable (dev:guest).",
             "% Una riga per metrica; le colonne sono i modelli nell'ordine:",
             "%   " + " | ".join(m[0] for m in models)]
    for label, key in metric_rows:
        cells = " & ".join(_cell(summ, key) if key in summ else "---" for _label, summ in models)
        lines.append(f"{label} & {cells} \\\\")
    lines.append(r"% $^{\ddagger}$ XGBoost \`e supervisionato (vede le etichette): upper-bound, "
                 r"fuori categoria.")
    (GEN_DIR / "tab_baselines.tex").write_text("\n".join(lines) + "\n")


# --- console summary -------------------------------------------------------------------
def _print_summary(title, summary, keys):
    print("\n" + "#" * 80)
    print(f"# {title}")
    print("#" * 80)
    for model, cells in summary.items():
        parts = " | ".join(f"{k}={fmt_ms(cells[k]['vals'])}" for k in keys if k in cells)
        print(f"  {model:22s} {parts}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--panels", choices=["A", "B", "AB"], default="AB",
                    help="which panels to (re)generate")
    args = ap.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    GEN_DIR.mkdir(parents=True, exist_ok=True)

    panel_b = None
    if "B" in args.panels or args.panels == "AB":
        panel_b = run_panel_b(args.seeds)
        _print_summary("PANEL B (tab:v3v4) — TGN v3 vs v4, per-cookie",
                       panel_b["summary"], list(PANEL_B_CELLS))

    if "A" in args.panels or args.panels == "AB":
        # Panel A needs the v3 per-cookie TGN row; reuse Panel B if available, else run it.
        if panel_b is None:
            panel_b = run_panel_b(args.seeds)
        tgn_row = _summarise(panel_b["raw"]["v3"], PANEL_A_TGN_CELLS)
        base = run_panel_a_baselines(args.seeds)
        payload = {
            "meta": _meta(args.seeds,
                          "Panel A: TGN row = v3 per-cookie; baselines = deployable (dev:guest)"),
            "tgn_row": {"raw": panel_b["raw"]["v3"], "summary": tgn_row},
            "baselines": base,
        }
        dump_json(payload, RUNS_DIR / "panelA.json")
        _emit_panel_a_tex(tgn_row, base["summary"])
        a_summary = {"TGN (v3,pc)": tgn_row, **base["summary"]}
        _print_summary("PANEL A (tab:baselines)", a_summary, list(PANEL_A_BASE_CELLS))

    print(f"\nJSON -> {RUNS_DIR}\nLaTeX fragments -> {GEN_DIR}\nDONE_REGEN_REPORT_TABLES")


if __name__ == "__main__":
    main()
