"""Leave-one-group-out: revert one group of v5 knobs to its v4 value, measure lookup rules.

    uv run python scratch/knob_ablation.py

Proto-self gating (paper protocol), benign-vs-class AUC in the test window, 3 seeds.
"""
import dataclasses

import numpy as np
from sklearn.metrics import roc_auc_score

from graphagate.config import TGNConfig
from graphagate.data.lookup_rules import lookup_flags
from graphagate.data.stream_synthetic import V4_KNOBS, generate_streaming_data, stream_kwargs_from_cfg

GROUPS = {
    "v5 (all on)": [],
    "- benign IP churn": ["p_new_source"],
    "- JA3 releases": ["p_config_release"],
    "- hot-desking": ["p_hotdesk"],
    "- theft mimicry+replay": ["p_theft_mimic_config", "p_theft_known_source", "p_theft_session_replay"],
    "- credential pivot": ["p_lateral_foreign_cred"],
    "+ role spoof (v4)": ["p_lateral_role_spoof", "p_lateral_new_config"],
    "- sensor noise/legacy": ["p_sensor_fp", "p_legacy_client"],
    "all v4 knobs": list(V4_KNOBS),
}
SINGLE = ("cfg_new", "src_new", "dev_new", "cfg|dev_new", "cfg|usr_new", "dev|usr_new",
          "src|usr_new", "role_changed")

for name, keys in GROUPS.items():
    row = {3: [], 4: []}
    for seed in (42, 7, 123):
        cfg = dataclasses.replace(TGNConfig(), seed=seed, **{k: V4_KNOBS[k] for k in keys})
        s = generate_streaming_data(**stream_kwargs_from_cfg(cfg))
        ts = int(len(s.y) * 0.8)
        f = lookup_flags(s, "proto-self", ts)
        ty = s.types.numpy()
        te = np.arange(len(ty)) >= ts
        for k in (3, 4):
            sel = te & ((ty == 0) | (ty == k))
            lab = ty[sel] == k
            best = max((roc_auc_score(lab, f[r][sel].astype(float)), r) for r in SINGLE)
            row[k].append((best[0], best[1], roc_auc_score(lab, f["stateful"][sel])))
    out = []
    for k, lbl in ((3, "lateral"), (4, "theft")):
        b = np.mean([x[0] for x in row[k]]); st = np.mean([x[2] for x in row[k]])
        rules = sorted({x[1] for x in row[k]})
        out.append(f"{lbl}: best-single={b:.3f} ({'/'.join(rules)}) stateful={st:.3f}")
    print(f"{name:24s} " + " | ".join(out), flush=True)
