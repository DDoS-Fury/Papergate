"""Stateful lookup-rule baseline against the synthetic generator (no learning).

    uv run python scratch/generator_rule_audit.py          # current generator (v5 defaults)
    uv run python scratch/generator_rule_audit.py --v4     # the pre-v5 process (V4_KNOBS)

Rules and gates live in graphagate.data.lookup_rules (see its docstring); this script
reports them per class, per gate, over 3 seeds. tests/test_leakage_audit.py asserts bounds.

Evaluation mirrors train_tgn.per_type: benign-vs-class AUC inside the test window
(last 20%, chronological 70/10/20), recall at the benign 99th-percentile threshold.

"""

from __future__ import annotations

import dataclasses
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from graphagate.config import TGNConfig
from graphagate.data.lookup_rules import GATES, lookup_flags
from graphagate.data.stream_synthetic import V4_KNOBS, generate_streaming_data, stream_kwargs_from_cfg

SEEDS = (42, 7, 123)
LAT_RULES = ("cfg|dev_new", "cfg|usr_new", "dev|usr_new", "src|usr_new", "role_changed",
             "sensor", "stateful", "combined")
THEFT_RULES = ("cfg_new", "src_new", "dev_new", "cfg|usr_new", "dev|usr_new", "src|usr_new",
               "stateful", "combined")


def _score(s, flags, type_id: int, test_start: int):
    ty = s.types.numpy()
    te = np.arange(len(ty)) >= test_start
    ben, cls = te & (ty == 0), te & (ty == type_id)
    if cls.sum() == 0:
        return float("nan"), float("nan"), float("nan"), 0
    sel = ben | cls
    v = flags.astype(float)
    auc = roc_auc_score((ty[sel] == type_id).astype(int), v[sel])
    thr = np.quantile(v[ben], 0.99)
    return auc, float((v[cls] > thr).mean()), float((v[ben] > thr).mean()), int(cls.sum())


def run(make_cfg, tasks) -> dict:
    """``tasks``: [(title, type_id, rules)]. Returns {(title, gate, rule): (mean AUC, std, recall, fpr, n)}."""
    res = {}
    rows = {(t, g, r): [] for t, _, rules in tasks for g in GATES for r in rules}
    for seed in SEEDS:
        cfg = make_cfg(seed)
        s = generate_streaming_data(**stream_kwargs_from_cfg(cfg))
        test_start = int(len(s.y) * (cfg.train_frac + cfg.val_frac))
        for gate in GATES:
            flags = lookup_flags(s, gate, test_start)
            for title, type_id, rules in tasks:
                for r in rules:
                    rows[(title, gate, r)].append(_score(s, flags[r], type_id, test_start))
    for key, vals in rows.items():
        a = np.array([x[:3] for x in vals])
        res[key] = (a[:, 0].mean(), a[:, 0].std(ddof=1), a[:, 1].mean(), a[:, 2].mean(),
                    [x[3] for x in vals])
    return res


def _print(res, tasks):
    for title, _, rules in tasks:
        print(f"\n=== {title} ===")
        for gate in GATES:
            for r in rules:
                auc, sd, rec, fpr, n = res[(title, gate, r)]
                print(f"  gate={gate:10s} {r:13s} AUC={auc:.3f}±{sd:.3f} "
                      f"recall={rec:.3f} benignFPR={fpr:.4f} n={n}")


if __name__ == "__main__":
    if "--v4" in sys.argv:
        # The v4 default config never puts a theft event in the test window (slots run
        # out at 23-32% of the stream), so theft is audited on the theft-rich stream of
        # tests/ablations/run_config_eval.py, as in the original audit.
        v4 = lambda seed, **kw: dataclasses.replace(TGNConfig(), seed=seed, **V4_KNOBS, **kw)
        lat = [("LATERAL — v4 process, default size (200k events)", 3, LAT_RULES)]
        th = [("CREDENTIAL THEFT — v4 process, theft-rich stream (80k)", 4, THEFT_RULES)]
        _print(run(v4, lat), lat)
        _print(run(lambda seed: v4(seed, num_events=80000, num_theft_slots=400,
                                   p_cred_theft=0.002), th), th)
    else:
        tasks = [("LATERAL — v5 default config (200k events)", 3, LAT_RULES),
                 ("CREDENTIAL THEFT — v5 default config (200k events)", 4, THEFT_RULES)]
        _print(run(lambda seed: TGNConfig(seed=seed), tasks), tasks)
