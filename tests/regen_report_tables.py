"""Single orchestrator that regenerates the report's tables from fixed seeds as a
reproducible artifact.

Lean evaluation (generator v5)
------------------------------
The paper compares the TGN only with what it has to beat — an Isolation Forest and the
stateful lookup rules (``src/data/lookup_rules.py``) — plus one architecture ablation (the
configuration node on/off) and a data-budget curve. The GNN / One-Class SVM / XGBoost /
2-node-TGN baselines stay in ``BASELINES`` but do not run by default (``--baselines``).
The decision rules below are pre-registered in ``tasks/todo.md`` and implemented here, in
code, before any run.

Panels (all ``save=False`` — the deployable artifact in ``public/`` is untouched)
* **A (E1)** — TGN | Isolation Forest | rules | rules + validation labels | rules, no labels.
  The primary rules row (``lookup_rules``) gets ground-truth labels only through the
  training window, like the TGN; ``lookup_rules_val`` (the rule audit's protocol, an oracle
  over validation) is the sensitivity row and ``lookup_rules_all`` uses no label at all.
  One protocol for every row: default ``TGNConfig()`` stream, chronological 70/10/20,
  benign-only training, every method on every seed. Per class (lateral, cred-theft): AUC,
  AP and recall at the global 1% FPR; plus per-seed *paired* TGN-minus-baseline differences.
  E1 verdict per comparator: the TGN "beats" it only if the paired difference is
  significant (Wilcoxon) AND its bootstrap 95% CI excludes 0 upward, on lateral AND theft
  AUC.
* **B (E2)** — config node on vs off. The "on" arm IS Panel A's TGN run (reloaded from
  ``panelA.json`` and guarded by a config-equality assert): only the "off" arm trains.
* **C (E3)** — data-budget curve: TGN / IF / rules on the last N training events before an
  untouched validation and test window (``eval_common.tail_stream``); the full budget is
  Panel A's run. **Fixed recipe** (``cfg.epochs``): a small budget gets proportionally
  fewer gradient steps, so N* is a conservative (over-)estimate of the data needed.
  N* = the smallest N whose paired 95% CI of (AUC_N - AUC_full) has lower bound >=
  -``NON_INFERIORITY_MARGIN`` with half-width <= the margin, on lateral AND theft; a CI
  wider than the margin is "undetermined" (extend the seeds), never a pass.

Seeds: 1000-1009, *fresh* — the generator v5 was tuned against 1-9/42/7/123. Panel C uses
the first five. Dev / smoke runs use 2000+ and must go through ``--out-dir``.

Run on the GPU box (about 15 min per TGN run; the fixed val/test replays dominate)::

    docker compose --profile regen-report up                                   # A (E1)
    docker compose run --rm regen-report /app/tests/regen_report_tables.py --panels B
    docker compose run --rm regen-report /app/tests/regen_report_tables.py --panels C

Every training/baseline run is cached under ``<out-dir>/cache`` and reused on a rerun, so a
crash after hours resumes instead of restarting. Delete the cache after changing code, or
pass ``--fresh``.

Outputs: ``panelA.json`` / ``panelB.json`` / ``panelC.json`` (per-seed dicts, cells,
paired statistics, verdicts) and LaTeX fragments under ``<out-dir>/generated`` (default
``docs/paper/generated/``).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import importlib.util
import inspect
import json
import math
import time
from pathlib import Path

import numpy as np

from graphagate.config import REPO_ROOT, TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg
from graphagate.eval_common import tail_stream
from graphagate.report_metrics import dump_json, fmt_ms, latex_cell, mean_std, paired_delta
from graphagate.train_tgn import stream_to_data, train_tgn

SEEDS = list(range(1000, 1010))        # fresh: the generator v5 was tuned against 1-9/42/7/123
BUDGET_SEEDS = SEEDS[:5]
BUDGETS = [10_000, 25_000, 50_000, 100_000]  # the full budget (140k) is Panel A's run
NON_INFERIORITY_MARGIN = 0.02           # AUC, pre-registered (tasks/todo.md)
TESTS_DIR = Path(__file__).resolve().parent
RUNS_DIR = REPO_ROOT / "tasks" / "runs"
GEN_DIR = REPO_ROOT / "docs" / "paper" / "generated"

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
    # key -> (script, entrypoint, label used in the tables)
    "iforest": ("baselines/isolation_forest/isolation_forest_baseline.py",
                "isolation_forest_baseline", "Isolation Forest"),
    "lookup_rules": ("baselines/lookup_rules/lookup_rules_baseline.py",
                     "lookup_rules_baseline", "Stateful rules"),
    "lookup_rules_val": ("baselines/lookup_rules/lookup_rules_baseline.py",
                         "lookup_rules_val_baseline", "Stateful rules (+val labels)"),
    "lookup_rules_all": ("baselines/lookup_rules/lookup_rules_baseline.py",
                         "lookup_rules_all_baseline", "Stateful rules (no labels)"),
    # Legacy baselines: NOT run by default. ocsvm / xgboost / gnn still build their history
    # counters from test labels (see tests/baselines/README.md) — do not cite their numbers.
    "tgn_2node": ("baselines/tgn_2node/tgn_2node_baseline.py", "tgn_2node_baseline", "TGN Vanilla (2 nodi)"),
    "gnn": ("baselines/simple_gnn/simple_gnn_baseline.py", "run", "GNN non temp."),
    "ocsvm": ("baselines/ocsvm/ocsvm_baseline.py", "ocsvm_baseline", "One-Class SVM"),
    "xgboost": ("baselines/xgboost/xgboost_baseline.py", "xgboost_baseline", "XGBoost (sup.)"),
}
LEAN_BASELINES = ["iforest", "lookup_rules", "lookup_rules_val", "lookup_rules_all"]
BUDGET_BASELINES = ["iforest", "lookup_rules"]   # Panel C: the primary comparators only


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


def _pt(cls: str, metric: str) -> tuple:
    return ("per_type", cls, metric)


# Panel B cells (TGN dicts): keyed by the metric the tab:v3v4 row needs.
PANEL_B_CELLS = {
    "agg_auc": ("agg_auc",),
    "agg_ap": ("agg_ap",),
    "agg_recall_routed": ("agg_recall",),          # routed (operational)
    "lat_auc": _pt("lateral", "auc"),
    "lat_ap": _pt("lateral", "ap"),
    "lat_recall_routed": _pt("lateral", "recall"),
    "lat_recall_global": ("lateral_recall_before",),  # @1% FPR global threshold
    "policy_auc": _pt("policy", "auc"),
    "policy_recall": _pt("policy", "recall"),
    "ctx_auc": _pt("contextual", "auc"),
    "ctx_recall": _pt("contextual", "recall"),
    "fpr_routed": ("fpr_after",),
}

# Panel A cells. The two dict kinds differ ONLY where the recall lives: the TGN's
# ``per_type[..]["recall"]`` is the ROUTED (cost-sensitive) recall, so the global-1%-FPR
# column reads ``recall_global`` for the TGN and ``recall`` for the baselines — mixing the
# two is what made an earlier Panel A compare unlike operating points.
_CLASSES = {"lat": "lateral", "theft": "cred-theft"}
PANEL_A_TGN_CELLS = {"agg_auc": ("agg_auc",), "agg_ap": ("agg_ap",),
                     "agg_recall_global": ("agg_recall_global",)}
PANEL_A_BASE_CELLS = {"agg_auc": ("agg_auc",), "agg_ap": ("agg_ap",),
                      "agg_recall_global": ("agg_recall",)}
for _short, _cls in _CLASSES.items():
    PANEL_A_TGN_CELLS.update({f"{_short}_auc": _pt(_cls, "auc"), f"{_short}_ap": _pt(_cls, "ap"),
                              f"{_short}_recall_global": _pt(_cls, "recall_global")})
    PANEL_A_BASE_CELLS.update({f"{_short}_auc": _pt(_cls, "auc"), f"{_short}_ap": _pt(_cls, "ap"),
                               f"{_short}_recall_global": _pt(_cls, "recall")})

# Panel C cells: identical paths for every method (AUC / AP live at the same place), plus
# the cost axis (TGN: gradient-loop seconds; baselines: whole-call seconds).
CURVE_CELLS = {"lat_auc": _pt("lateral", "auc"), "lat_ap": _pt("lateral", "ap"),
               "theft_auc": _pt("cred-theft", "auc"), "theft_ap": _pt("cred-theft", "ap")}
PAIRED_CELLS = ("lat_auc", "lat_ap", "theft_auc", "theft_ap", "agg_auc", "agg_ap")


# --- run caching -----------------------------------------------------------------------
def _cached(kind: str, seed: int, ov: dict, fn, *, fresh: bool = False) -> dict:
    """Run ``fn()`` once per (kind, seed, overrides); a rerun reuses the JSON on disk.

    Multi-hour GPU jobs must survive a crash: every finished run is persisted, and the
    value returned is always the JSON round-trip so fresh and cached runs are identical.
    """
    key = hashlib.sha1(json.dumps([kind, seed, ov], sort_keys=True).encode()).hexdigest()[:10]
    path = RUNS_DIR / "cache" / f"{kind}_s{seed}_{key}.json"
    if path.exists() and not fresh:
        print(f"[cache] {path.name}", flush=True)
    else:
        dump_json(fn(), path)
    return json.loads(path.read_text())


def _cfg(seed: int, ov: dict) -> TGNConfig:
    return dataclasses.replace(TGNConfig(), seed=seed, **ov)


def _call(fn, cfg, stream):
    """Call a baseline, injecting the pre-built stream when the entrypoint accepts one."""
    kw = {"stream": stream} if "stream" in inspect.signature(fn).parameters else {}
    t0 = time.perf_counter()
    out = fn(cfg, **kw)
    return {**out, "wall_seconds": time.perf_counter() - t0}


# --- runners ---------------------------------------------------------------------------
def run_panel_a_baselines(seeds, keys, ov, fresh=False) -> dict:
    """Each selected baseline on the deployable stream, |seeds| runs each (cheap: run first)."""
    raw = {k: [] for k in keys}
    for seed in seeds:
        cfg = _cfg(seed, ov)
        stream = generate_streaming_data(**stream_kwargs_from_cfg(cfg))
        for key in keys:
            rel, attr, _label = BASELINES[key]
            print("\n" + "=" * 80)
            print(f"=== PANEL A baseline: {key} | seed={seed} | {cfg.num_events} ev ===")
            print("=" * 80, flush=True)
            raw[key].append(_cached(f"A_{key}", seed, ov, lambda: _call(_load(rel, attr), cfg, stream),
                                    fresh=fresh))
    return {"raw": raw, "summary": {k: _summarise(raw[k], PANEL_A_BASE_CELLS) for k in raw}}


def run_panel_a_tgn(seeds, ov, fresh=False) -> list[dict]:
    """TGN under the SAME protocol as the baselines: default ``TGNConfig()``, full model."""
    out = []
    for seed in seeds:
        cfg = _cfg(seed, ov)
        print("\n" + "=" * 80)
        print(f"=== PANEL A: TGN | seed={seed} | default config, {cfg.num_events} ev / {cfg.epochs} ep ===")
        print("=" * 80, flush=True)
        out.append(_cached("A_tgn", seed, ov, lambda: train_tgn(cfg, save=False, use_config_node=True),
                           fresh=fresh))
    return out


def _paired_vs_tgn(tgn_row: dict, base_summary: dict) -> dict:
    """Per-seed paired TGN-minus-baseline statistics for every comparator and cell.

    ``tgn_better`` is the pre-registered rule: the Wilcoxon test is significant AND the
    bootstrap 95% CI of the mean paired difference excludes 0 upward. Cells a comparator
    does not report (the rules' aggregate) are skipped.
    """
    out = {}
    for key, summ in base_summary.items():
        out[key] = {}
        for cell in PAIRED_CELLS:
            a, b = tgn_row[cell]["vals"], summ[cell]["vals"]
            if np.isnan(np.asarray(b, dtype=float)).all():
                continue
            res = paired_delta(a, b)
            res["tgn_better"] = bool(res["significant"] and res["ci95"][0] > 0)
            out[key][cell] = res
    return out


def _verdict(paired: dict) -> dict:
    """E1 (pre-registered): the TGN beats a comparator iff ``tgn_better`` on lateral AND theft AUC."""
    return {key: bool(p.get("lat_auc", {}).get("tgn_better") and p.get("theft_auc", {}).get("tgn_better"))
            for key, p in paired.items()}


def run_panel_a(seeds, keys, ov, fresh=False) -> dict:
    base = run_panel_a_baselines(seeds, keys, ov, fresh)      # cheap first: fail before the GPU hours
    tgn_raw = run_panel_a_tgn(seeds, ov, fresh)
    tgn_row = _summarise(tgn_raw, PANEL_A_TGN_CELLS)
    paired = _paired_vs_tgn(tgn_row, base["summary"])
    payload = {
        "meta": _meta(seeds, "Panel A (E1): TGN vs baselines, default config, global 1% FPR threshold", ov),
        "tgn_row": {"raw": tgn_raw, "summary": tgn_row},
        "baselines": base, "paired": paired, "verdict_tgn_beats": _verdict(paired),
    }
    dump_json(payload, RUNS_DIR / "panelA.json")
    _emit_panel_a_tex(tgn_row, base["summary"])
    _emit_paired_tex(paired)
    _print_summary("PANEL A (E1)", {"TGN": tgn_row, **base["summary"]}, list(PANEL_A_BASE_CELLS))
    print("\nE1 verdict (TGN beats comparator on lateral AND theft AUC):", payload["verdict_tgn_beats"])
    return payload


def _load_panel_a(seeds, ov):
    """Reload Panel A's per-seed runs for ``seeds`` (full-budget TGN + baselines), or fail loudly."""
    p = RUNS_DIR / "panelA.json"
    if not p.exists():
        raise SystemExit(f"{p} not found: run --panels A first (Panels B and C reuse its TGN run).")
    payload = json.loads(p.read_text())
    meta, want = payload["meta"], _meta(seeds, "", ov)
    if (any(s not in meta["seeds"] for s in seeds) or meta["num_events"] != want["num_events"]
            or meta["epochs"] != want["epochs"] or meta.get("overrides", {}) != ov):
        raise SystemExit(f"{p} was not produced with seeds={seeds}, overrides={ov}: rerun --panels A.")
    idx = [meta["seeds"].index(s) for s in seeds]
    tgn = [payload["tgn_row"]["raw"][i] for i in idx]
    base = {k: [v[i] for i in idx] for k, v in payload["baselines"]["raw"].items()}
    return tgn, base


def run_panel_b(seeds, ov, fresh=False) -> dict:
    """Config node on vs off (E2). "on" = Panel A's TGN run; only "off" trains."""
    v4_raw, _ = _load_panel_a(seeds, ov)
    for seed in seeds:  # the reuse is only valid if Panel A's config IS Panel B's "on" config
        cfg_a = _cfg(seed, ov)
        assert cfg_a == dataclasses.replace(cfg_a, guest_device_fallback=False), (
            "Panel A and Panel B configs differ: Panel A's TGN run is not Panel B's 'on' arm")
    v3_raw = []
    for seed in seeds:
        cfg = _cfg(seed, ov)
        print("\n" + "=" * 80)
        print(f"=== PANEL B: config node OFF | seed={seed} | {cfg.num_events} ev / {cfg.epochs} ep ===")
        print("=" * 80, flush=True)
        v3_raw.append(_cached("B_off", seed, ov, lambda: train_tgn(cfg, save=False, use_config_node=False),
                              fresh=fresh))
    raw = {"v4": v4_raw, "v3": v3_raw}
    summary = {name: _summarise(raw[name], PANEL_B_CELLS) for name in raw}
    paired = {c: paired_delta(summary["v4"][c]["vals"], summary["v3"][c]["vals"]) for c in PANEL_B_CELLS}
    payload = {"meta": _meta(seeds, "per-cookie, routed decision; 'on' arm = Panel A's TGN run", ov),
               "raw": raw, "summary": summary, "paired_on_minus_off": paired}
    dump_json(payload, RUNS_DIR / "panelB.json")
    _emit_panel_b_tex(summary)
    _print_summary("PANEL B (E2) — config node", summary, list(PANEL_B_CELLS))
    return payload


# --- Panel C: data-budget curve ---------------------------------------------------------
def _curve_vals(raw: dict) -> dict:
    """raw[method][N] = [per-seed run dict]  ->  vals[method][N][cell] = [per-seed float]."""
    return {m: {n: {c: [float(_g(d, *p)) for d in runs] for c, p in CURVE_CELLS.items()}
                for n, runs in by_n.items()} for m, by_n in raw.items()}


def _n_star(vals: dict, n_full: int, margin: float = NON_INFERIORITY_MARGIN) -> dict:
    """Smallest budget whose AUC is non-inferior to the full budget (pre-registered).

    ``vals``: ``{N: {cell: [per-seed values]}}`` for ONE method, including ``n_full``.
    Non-inferior at N: for lateral AND theft AUC the paired 95% CI of (AUC_N - AUC_full)
    has lower bound >= -margin AND half-width <= margin. A CI wider than the margin is
    never a pass: if no N qualifies but some CI is that wide, the answer is "undetermined"
    (extend the seeds); if none is that wide, no reduced budget is non-inferior ("full").
    """
    detail, wide_below = {}, False
    for n in sorted(k for k in vals if k < n_full):
        ok = True
        detail[n] = {}
        for cell in ("lat_auc", "theft_auc"):
            r = paired_delta(vals[n][cell], vals[n_full][cell])
            lo, hi = r["ci95"]
            hw = (hi - lo) / 2 if not (math.isnan(lo) or math.isnan(hi)) else NAN
            detail[n][cell] = {"mean_delta": r["mean_delta"], "ci95": list(r["ci95"]),
                               "half_width": hw, "n_pairs": r["n_pairs"]}
            if math.isnan(hw) or hw > margin:
                ok = False
                wide_below = True
            elif lo < -margin:
                ok = False
        if ok:
            return {"n_star": n, "status": "established", "resolution_limited": wide_below,
                    "detail": detail}
    status = "undetermined" if wide_below else "full"
    return {"n_star": n_full if status == "full" else None, "status": status,
            "resolution_limited": wide_below, "detail": detail}


def _adaptation_verdict(vals: dict, nstars: dict) -> dict:
    """The adaptation claim (pre-registered): survives only if N*_TGN <= N*_IF AND the TGN's AUC
    at its own N* beats the IF's at N*_IF and the rules' at N*_rules (paired; lateral AND theft).
    Otherwise E3 is reported as a cost curve and the claim leaves the abstract."""
    if any(nstars[m]["status"] == "undetermined" for m in nstars):
        return {"verdict": "undetermined", "reason": "some N* is undetermined: extend the seeds"}
    n = {m: nstars[m]["n_star"] for m in nstars}
    if n["tgn"] > n["iforest"]:
        return {"verdict": "rejected", "reason": f"N*_TGN={n['tgn']} > N*_IF={n['iforest']}"}
    for other in ("iforest", "lookup_rules"):
        for cell in ("lat_auc", "theft_auc"):
            r = paired_delta(vals["tgn"][n["tgn"]][cell], vals[other][n[other]][cell])
            if not (r["significant"] and r["ci95"][0] > 0):
                return {"verdict": "rejected", "reason": f"TGN not better than {other} on {cell}"}
    return {"verdict": "supported", "n_star": n}


def run_panel_c(seeds, budgets, ov, keys, fresh=False) -> dict:
    """E3: AUC(N) for TGN / IF / rules on the last N training events; full budget = Panel A."""
    if not {"iforest", "lookup_rules"} <= set(keys):
        raise SystemExit("--panels C needs the iforest and lookup_rules baselines.")
    if any("stream" not in inspect.signature(_load(*BASELINES[k][:2])).parameters for k in keys):
        raise SystemExit("--panels C runs only baselines that accept an injected stream.")
    full_tgn, full_base = _load_panel_a(seeds, ov)
    cfg0 = _cfg(seeds[0], ov)
    n_full = int(cfg0.num_events * cfg0.train_frac)
    if any(not 0 < b < n_full for b in budgets):
        raise SystemExit(f"budgets must be in (0, {n_full}) — the full budget is Panel A's run.")

    methods = ["tgn", *keys]
    raw = {m: {n_full: (full_tgn if m == "tgn" else full_base[m])} for m in methods}
    for n in budgets:
        for m in methods:
            raw[m][n] = []
    for seed in seeds:
        cfg = _cfg(seed, ov)
        full = generate_streaming_data(**stream_kwargs_from_cfg(cfg))
        for n in budgets:
            s_n, cfg_n = tail_stream(full, cfg, n)
            print("\n" + "=" * 80)
            print(f"=== PANEL C: budget N={n} | seed={seed} | val/test windows unchanged ===")
            print("=" * 80, flush=True)
            raw["tgn"][n].append(_cached(
                f"C_tgn_N{n}", seed, ov,
                lambda: train_tgn(cfg_n, dataset=stream_to_data(s_n), save=False), fresh=fresh))
            for key in keys:
                rel, attr, _ = BASELINES[key]
                raw[key][n].append(_cached(f"C_{key}_N{n}", seed, ov,
                                           lambda: _call(_load(rel, attr), cfg_n, s_n), fresh=fresh))
    vals = _curve_vals(raw)
    nstars = {m: _n_star(vals[m], n_full) for m in methods}
    cost = {m: {n: mean_std([float(_g(d, *(("train_seconds",) if m == "tgn" else ("wall_seconds",))))
                             for d in runs])[0] for n, runs in raw[m].items()} for m in methods}
    payload = {"meta": {**_meta(seeds, "Panel C (E3): data-budget curve, fixed recipe", ov),
                        "budgets": [*budgets, n_full], "margin": NON_INFERIORITY_MARGIN},
               "raw": raw, "vals": vals, "cost_seconds_mean": cost, "n_star": nstars,
               "adaptation": _adaptation_verdict(vals, nstars)}
    dump_json(payload, RUNS_DIR / "panelC.json")
    print("\nN* per method:", {m: (v["n_star"], v["status"]) for m, v in nstars.items()})
    print("Adaptation verdict:", payload["adaptation"])
    return payload


def _meta(seeds, protocol, ov):
    cfg = _cfg(seeds[0], ov)
    return {
        "seeds": list(seeds), "num_events": cfg.num_events, "epochs": cfg.epochs,
        "overrides": ov, "protocol": protocol, "save": False,
        "generated": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }


# --- LaTeX fragment emission -----------------------------------------------------------
def _cell(summary: dict, key: str, *, bold=False) -> str:
    s = summary[key]
    if math.isnan(s["mean"]):
        return "---"
    return latex_cell(s["vals"], bold=bold)


def _delta(a: dict, b: dict, key: str) -> str:
    """Signed Δ (a − b) from the PAIRED per-seed differences, with a significance marker.

    Both arms are run under the same seeds, so the comparison is paired: the per-seed
    differences are what carries the signal. The dagger marks a Δ that the paired test
    does not establish (Wilcoxon signed-rank, α=0.05).
    """
    if any(math.isnan(x) for x in (a[key]["mean"], b[key]["mean"])):
        return "---"
    res = paired_delta(a[key]["vals"], b[key]["vals"])
    sig = "" if res["significant"] else r"$^{\dagger}$"
    return f"${res['mean_delta']:+.3f}${sig}"


def _emit_panel_b_tex(summary: dict):
    v4, v3 = summary["v4"], summary["v3"]
    rows = [
        ("Aggregate AUC", "agg_auc"),
        ("Aggregate AP", "agg_ap"),
        ("Aggregate recall (routed)", "agg_recall_routed"),
        ("Lateral --- AUC", "lat_auc"),
        ("Lateral --- AP", "lat_ap"),
        ("Lateral --- recall (routed)", "lat_recall_routed"),
        ("Lateral --- recall (@1\\%FPR)", "lat_recall_global"),
        ("Policy --- AUC", "policy_auc"),
        ("Contextual --- AUC", "ctx_auc"),
        ("Benign FPR (routed)", "fpr_routed"),
    ]
    lines = ["% Generated by tests/regen_report_tables.py — do NOT edit by hand.",
             "% Numeric source: tasks/runs/panelB.json",
             "% Columns: Metric & config node off & config node on & $\\Delta$ (on-off)"]
    for label, key in rows:
        lines.append(f"{label} & {_cell(v3, key)} & {_cell(v4, key)} & "
                     f"{_delta(v4, v3, key)} \\\\")
    lines.append(r"% $^{\dagger}$ = $\Delta$ not established by the paired Wilcoxon test.")
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    (GEN_DIR / "tab_v3v4.tex").write_text("\n".join(lines) + "\n")


def _emit_panel_a_tex(tgn: dict, base_summary: dict):
    # Columns of tab:baselines are MODELS; rows are METRICS.
    metric_rows = [
        ("Aggregate AUC", "agg_auc"),
        ("Aggregate AP", "agg_ap"),
        ("Lateral --- AUC", "lat_auc"),
        ("Lateral --- AP", "lat_ap"),
        ("Lateral --- recall @1\\%FPR", "lat_recall_global"),
        ("Theft --- AUC", "theft_auc"),
        ("Theft --- AP", "theft_ap"),
        ("Theft --- recall @1\\%FPR", "theft_recall_global"),
    ]
    models = [("TGN", tgn)] + [(BASELINES[k][2], base_summary[k]) for k in base_summary]
    lines = ["% Generated by tests/regen_report_tables.py — do NOT edit by hand.",
             "% Numeric source: tasks/runs/panelA.json",
             "% One row per metric; columns are the models in this order:",
             "%   " + " | ".join(m[0] for m in models),
             "% Rules: no aggregate (--- = not reported); recall is at the smallest integer threshold",
             "% whose benign validation FPR is <= 1% (a quantile is degenerate on integer scores)."]
    for label, key in metric_rows:
        lines.append(f"{label} & " + " & ".join(_cell(summ, key) for _l, summ in models) + " \\\\")
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    (GEN_DIR / "tab_baselines.tex").write_text("\n".join(lines) + "\n")


def _emit_paired_tex(paired: dict):
    rows = [("Lateral --- AUC", "lat_auc"), ("Lateral --- AP", "lat_ap"),
            ("Theft --- AUC", "theft_auc"), ("Theft --- AP", "theft_ap")]
    keys = list(paired)
    lines = ["% Generated by tests/regen_report_tables.py — do NOT edit by hand.",
             "% Paired per-seed TGN minus comparator: mean [bootstrap 95% CI]; dagger = TGN not shown better",
             "% (Wilcoxon significant AND CI excluding 0 upward). Numeric source: tasks/runs/panelA.json",
             "%   columns: " + " | ".join(BASELINES[k][2] for k in keys)]
    for label, cell in rows:
        cells = []
        for k in keys:
            r = paired[k].get(cell)
            if r is None:
                cells.append("---")
                continue
            lo, hi = r["ci95"]
            dag = "" if r["tgn_better"] else r"$^{\dagger}$"
            cells.append(f"${r['mean_delta']:+.3f}$ [{lo:+.3f}, {hi:+.3f}]{dag}")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    (GEN_DIR / "tab_paired.tex").write_text("\n".join(lines) + "\n")


# --- console summary -------------------------------------------------------------------
def _print_summary(title, summary, keys):
    print("\n" + "#" * 80)
    print(f"# {title}")
    print("#" * 80)
    for model, cells in summary.items():
        parts = " | ".join(f"{k}={fmt_ms(cells[k]['vals'])}" for k in keys if k in cells)
        print(f"  {model:22s} {parts}")


def main():
    global RUNS_DIR, GEN_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS, help="Panels A and B (default 1000-1009)")
    ap.add_argument("--panels", choices=["A", "B", "C", "AB", "ABC"], default="A",
                    help="A = E1 (default; B and C are gated on its result), B = E2, C = E3")
    ap.add_argument("--baselines", nargs="+", choices=list(BASELINES), default=LEAN_BASELINES)
    ap.add_argument("--budget-seeds", type=int, nargs="+", default=BUDGET_SEEDS, help="Panel C seeds")
    ap.add_argument("--budgets", type=int, nargs="+", default=BUDGETS, help="Panel C training budgets (events)")
    ap.add_argument("--events", type=int, default=None, help="SMOKE ONLY: override num_events")
    ap.add_argument("--epochs", type=int, default=None, help="SMOKE ONLY: override epochs")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write JSON/cache here and LaTeX under <dir>/generated (required with --events/--epochs)")
    ap.add_argument("--fresh", action="store_true", help="ignore the run cache")
    args = ap.parse_args()

    ov = {k: v for k, v in (("num_events", args.events), ("epochs", args.epochs)) if v is not None}
    if ov and args.out_dir is None:
        raise SystemExit("--events/--epochs are smoke overrides and would overwrite the real "
                         "panelA/B/C.json: pass --out-dir.")
    if args.out_dir is not None:
        RUNS_DIR, GEN_DIR = args.out_dir, args.out_dir / "generated"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    GEN_DIR.mkdir(parents=True, exist_ok=True)

    if "A" in args.panels:
        run_panel_a(args.seeds, args.baselines, ov, args.fresh)
    if "B" in args.panels:
        run_panel_b(args.seeds, ov, args.fresh)
    if "C" in args.panels:
        run_panel_c(args.budget_seeds, args.budgets, ov, BUDGET_BASELINES, args.fresh)

    print(f"\nJSON -> {RUNS_DIR}\nLaTeX fragments -> {GEN_DIR}\nDONE_REGEN_REPORT_TABLES")


if __name__ == "__main__":
    main()
