"""The pre-registered decision rules of ``regen_report_tables.py`` (no training involved).

    pytest tests/test_regen_report.py

E1 verdict, N* (non-inferiority) and the adaptation verdict are the analysis the paper's
claims hang on; they are pinned here on synthetic per-seed values so the rule cannot drift
after the data have been seen.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parent / "regen_report_tables.py"
_spec = importlib.util.spec_from_file_location("regen_report_tables", _PATH)
rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rr)

SEEDS10 = list(range(10))
NOISE = np.array([0.004, -0.003, 0.002, -0.004, 0.003, -0.002, 0.001, -0.001, 0.0035, -0.0025])


def _row(vals):
    m, s = rr.mean_std(vals)
    return {"mean": m, "std": s, "vals": list(vals)}


def _cells(**by_cell):
    return {c: _row(v) for c, v in by_cell.items()}


def _tgn_row(lat, theft):
    base = np.full(10, 0.5)
    return _cells(lat_auc=lat, lat_ap=lat, theft_auc=theft, theft_ap=theft, agg_auc=base, agg_ap=base)


# --- E1 ---------------------------------------------------------------------------------
def test_e1_tgn_beats_only_when_lateral_and_theft_are_both_established():
    tgn = _tgn_row(0.80 + NOISE, 0.85 + NOISE)
    rule = _cells(lat_auc=0.68 + NOISE, lat_ap=0.3 + NOISE, theft_auc=0.82 + NOISE, theft_ap=0.3 + NOISE,
                  agg_auc=[np.nan] * 10, agg_ap=[np.nan] * 10)
    paired = rr._paired_vs_tgn(tgn, {"lookup_rules": rule})
    assert paired["lookup_rules"]["lat_auc"]["tgn_better"]
    assert paired["lookup_rules"]["theft_auc"]["tgn_better"]
    assert "agg_auc" not in paired["lookup_rules"]           # the rules report no aggregate
    assert rr._verdict(paired) == {"lookup_rules": True}


def test_e1_one_class_not_established_is_not_a_win():
    # Lateral clearly better, theft a coin flip: the rule needs BOTH.
    tgn = _tgn_row(0.80 + NOISE, 0.82 + NOISE)
    rule = _cells(lat_auc=0.68 + NOISE, lat_ap=0.3 + NOISE, theft_auc=0.82 - NOISE, theft_ap=0.3 + NOISE,
                  agg_auc=[np.nan] * 10, agg_ap=[np.nan] * 10)
    paired = rr._paired_vs_tgn(tgn, {"lookup_rules": rule})
    assert paired["lookup_rules"]["lat_auc"]["tgn_better"]
    assert not paired["lookup_rules"]["theft_auc"]["tgn_better"]
    assert rr._verdict(paired) == {"lookup_rules": False}


def test_e1_worse_tgn_is_never_better():
    tgn = _tgn_row(0.60 + NOISE, 0.60 + NOISE)
    rule = _cells(lat_auc=0.70 + NOISE, lat_ap=0.3 + NOISE, theft_auc=0.75 + NOISE, theft_ap=0.3 + NOISE,
                  agg_auc=[np.nan] * 10, agg_ap=[np.nan] * 10)
    assert rr._verdict(rr._paired_vs_tgn(tgn, {"r": rule})) == {"r": False}


def test_e1_three_seeds_cannot_establish_anything():
    # Wilcoxon needs >= 6 pairs: at 3 seeds even a huge gap is "not established".
    tgn = _tgn_row(np.array([0.9, 0.9, 0.9]), np.array([0.9, 0.9, 0.9]))
    tgn = {c: _row(v["vals"][:3]) for c, v in tgn.items()}
    rule = {c: _row([0.5, 0.5, 0.5]) for c in tgn}
    assert rr._verdict(rr._paired_vs_tgn(tgn, {"r": rule})) == {"r": False}


# --- E3: N* -----------------------------------------------------------------------------
N_FULL = 140_000


def _curve(by_n):
    """{N: (lat_auc_per_seed, theft_auc_per_seed)} -> the ``vals`` structure of one method."""
    return {n: {"lat_auc": list(lat), "theft_auc": list(theft)} for n, (lat, theft) in by_n.items()}


def test_n_star_is_the_smallest_noninferior_budget():
    full = 0.80 + NOISE
    vals = _curve({
        10_000: (full - 0.10, full - 0.10),                   # clearly worse
        25_000: (full - 0.05 + NOISE / 4, full - 0.05 + NOISE / 4),
        50_000: (full - 0.004 + NOISE / 4, full - 0.004 + NOISE / 4),   # within the margin, tight CI
        100_000: (full + NOISE / 4, full + NOISE / 4),
        N_FULL: (full, full),
    })
    out = rr._n_star(vals, N_FULL)
    assert out["status"] == "established" and out["n_star"] == 50_000


def test_wide_ci_is_undetermined_not_a_pass():
    rng = np.random.default_rng(0)
    full = 0.80 + NOISE
    noisy = full + rng.normal(0, 0.12, 10)                    # CI half-width >> margin
    vals = _curve({25_000: (noisy, noisy), N_FULL: (full, full)})
    out = rr._n_star(vals, N_FULL)
    assert out["status"] == "undetermined" and out["n_star"] is None


def test_no_reduced_budget_within_margin_means_full():
    full = 0.80 + NOISE
    vals = _curve({25_000: (full - 0.08, full - 0.08), 50_000: (full - 0.05, full - 0.05), N_FULL: (full, full)})
    out = rr._n_star(vals, N_FULL)
    assert out["status"] == "full" and out["n_star"] == N_FULL


def test_both_classes_must_be_noninferior():
    full = 0.80 + NOISE
    vals = _curve({50_000: (full, full - 0.08), N_FULL: (full, full)})   # lateral fine, theft not
    assert rr._n_star(vals, N_FULL)["status"] == "full"


# --- E3: adaptation verdict ---------------------------------------------------------------
def _method(auc_by_n):
    return {n: {"lat_auc": list(a), "theft_auc": list(a)} for n, a in auc_by_n.items()}


def _nstar(n, status="established"):
    return {"n_star": n, "status": status}


def test_adaptation_supported_when_tgn_needs_no_more_data_and_is_better():
    vals = {"tgn": _method({50_000: 0.85 + NOISE}), "iforest": _method({50_000: 0.60 + NOISE}),
            "lookup_rules": _method({100_000: 0.70 + NOISE})}
    ns = {"tgn": _nstar(50_000), "iforest": _nstar(50_000), "lookup_rules": _nstar(100_000)}
    assert rr._adaptation_verdict(vals, ns)["verdict"] == "supported"


def test_adaptation_rejected_when_tgn_needs_more_data_than_the_iforest():
    vals = {"tgn": _method({100_000: 0.9 + NOISE}), "iforest": _method({25_000: 0.6 + NOISE}),
            "lookup_rules": _method({25_000: 0.6 + NOISE})}
    ns = {"tgn": _nstar(100_000), "iforest": _nstar(25_000), "lookup_rules": _nstar(25_000)}
    out = rr._adaptation_verdict(vals, ns)
    assert out["verdict"] == "rejected" and "N*_TGN" in out["reason"]


def test_adaptation_rejected_when_tgn_is_not_better_than_the_rules():
    vals = {"tgn": _method({50_000: 0.70 + NOISE}), "iforest": _method({50_000: 0.60 + NOISE}),
            "lookup_rules": _method({50_000: 0.70 - NOISE})}
    ns = {m: _nstar(50_000) for m in vals}
    out = rr._adaptation_verdict(vals, ns)
    assert out["verdict"] == "rejected" and "lookup_rules" in out["reason"]


def test_adaptation_undetermined_propagates():
    ns = {"tgn": _nstar(None, "undetermined"), "iforest": _nstar(50_000), "lookup_rules": _nstar(50_000)}
    assert rr._adaptation_verdict({}, ns)["verdict"] == "undetermined"


# --- plumbing -----------------------------------------------------------------------------
def test_cell_maps_never_mix_routed_and_global_recall():
    # The TGN's per_type "recall" is the ROUTED recall; only "recall_global" is comparable.
    assert rr.PANEL_A_TGN_CELLS["lat_recall_global"] == ("per_type", "lateral", "recall_global")
    assert rr.PANEL_A_TGN_CELLS["theft_recall_global"] == ("per_type", "cred-theft", "recall_global")
    assert rr.PANEL_A_BASE_CELLS["lat_recall_global"] == ("per_type", "lateral", "recall")
    assert rr.PANEL_A_BASE_CELLS["theft_recall_global"] == ("per_type", "cred-theft", "recall")


def test_cache_returns_the_json_roundtrip_and_reuses_it(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "RUNS_DIR", tmp_path)
    calls = []

    def fn():
        calls.append(1)
        return {"x": np.float64(1.5), "t": (1, 2), "nan": float("nan")}

    a = rr._cached("k", 7, {"num_events": 10}, fn)
    b = rr._cached("k", 7, {"num_events": 10}, fn)
    assert len(calls) == 1 and a == {"x": 1.5, "t": [1, 2], "nan": a["nan"]} and b["t"] == [1, 2]
    rr._cached("k", 7, {"num_events": 11}, fn)              # different overrides -> different run
    rr._cached("k", 8, {"num_events": 10}, fn)              # different seed -> different run
    rr._cached("k", 7, {"num_events": 10}, fn, fresh=True)  # --fresh ignores the cache
    assert len(calls) == 4
    assert len(list((tmp_path / "cache").glob("*.json"))) == 3


def test_smoke_overrides_require_an_out_dir(monkeypatch):
    monkeypatch.setattr("sys.argv", ["regen_report_tables.py", "--events", "1000"])
    with pytest.raises(SystemExit) as e:
        rr.main()
    assert "--out-dir" in str(e.value)


def test_latex_cell_never_emits_pmnan():
    # One valid seed -> no spread: the cell must still compile (``\pmnan`` is undefined in LaTeX).
    assert rr.latex_cell([0.5]) == "$0.500$"
    assert rr.latex_cell([0.5, float("nan")]) == "$0.500$"
    assert rr.latex_cell([0.4, 0.6], bold=True) == r"$\mathbf{0.500}\pm0.141$"
    assert rr._cell({"k": {"mean": float("nan"), "std": float("nan"), "vals": [float("nan")]}}, "k") == "---"
