"""Points 1-3 of tasks/todo.md on a single training run.

The kill-chain precursor is a serving-time prior and never a trained input, so the weights
are shared across the whole grid: we train once, snapshot the runtime state the calibration
starts from, and then re-run (calibration pass A, calibration pass B, test replay) for each
candidate setting. ~14 min per configuration against ~22 min of training, and the
comparison is exactly paired — same weights, same stream, same starting memory.

Also dumps the raw anomaly logits of the reference configuration, which is what the
false-positive breakdown (point 1) and the sizing of ``precursor_max_shift`` need.

Run:
  docker compose run --rm --no-deps -T -v "$PWD/tasks:/app/tasks" \
      --entrypoint python train-tgn /app/tasks/tmp/precursor_sweep.py
"""
import copy, dataclasses, json, time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

import graphagate.train_tgn as T
from graphagate.calibration import routed_predict
from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg
from graphagate.train_tgn import fit_thresholds, stream_to_data

OUT = "/app/tasks/tmp/precursor_sweep"
HOUR = 3600.0

# One axis at a time around the centre (24 h, 4 nats, arming on): every row is
# attributable to a single change. Row 1 isolates point 4 (logit space alone, precursor
# still inert); the last row isolates point 3 (same prior, arming re-coupled to the
# decision threshold).
# (name, half_life seconds, max_shift nats, arm on threshold_clean_unsup)
GRID = [
    ("logit only (shift=0)", 600.0,     0.0,  False),
    ("hl=24h sh=2 arm",     24 * HOUR,  2.0,  True),
    ("hl=24h sh=4 arm",     24 * HOUR,  4.0,  True),
    ("hl=24h sh=8 arm",     24 * HOUR,  8.0,  True),
    ("hl=6h  sh=4 arm",      6 * HOUR,  4.0,  True),
    ("hl=72h sh=4 arm",     72 * HOUR,  4.0,  True),
    ("hl=24h sh=4 no-arm",  24 * HOUR,  4.0,  False),
]

cap = {"calls": [], "model": None, "snap": None, "logits": {}}
orig_replay = T._replay


def _snapshot(m):
    return {
        "memory": copy.deepcopy(m.memory.state_dict()),
        "msg_s_store": copy.deepcopy(m.memory.msg_s_store),
        "msg_d_store": copy.deepcopy(m.memory.msg_d_store),
        "neighbor": {k: (v.clone() if torch.is_tensor(v) else v)
                     for k, v in m.neighbor_loader.state().items()},
        "last_contact": dict(m.last_contact),
        "pair_count": dict(m.pair_count),
        "src_count": dict(m.src_count),
        "recent_alert": dict(m.recent_alert),
        "node_feat": m.node_feat.detach().clone(),
    }


def _restore(m, snap):
    m.memory.load_state_dict(copy.deepcopy(snap["memory"]))
    m.memory.msg_s_store = copy.deepcopy(snap["msg_s_store"])
    m.memory.msg_d_store = copy.deepcopy(snap["msg_d_store"])
    m.neighbor_loader.load_state(
        {k: (v.clone() if torch.is_tensor(v) else v) for k, v in snap["neighbor"].items()})
    m.last_contact = dict(snap["last_contact"])
    m.pair_count = dict(snap["pair_count"])
    m.src_count = dict(snap["src_count"])
    m.recent_alert = dict(snap["recent_alert"])
    m.node_feat.copy_(snap["node_feat"])


def recorder(model, *a, **k):
    """Capture the exact slices train_tgn replays, so the sweep re-runs them verbatim."""
    if cap["snap"] is None:          # first call = calibration pass A, pre-calibration state
        cap["model"], cap["snap"] = model, _snapshot(model)
    cap["calls"].append((a, dict(k)))
    return orig_replay(model, *a, **k)


cfg = dataclasses.replace(TGNConfig(), seed=2000,
                          precursor_half_life=GRID[0][1], precursor_max_shift=GRID[0][2])
stream = generate_streaming_data(**stream_kwargs_from_cfg(cfg))
T._replay = recorder
t0 = time.time()
res0 = T.train_tgn(cfg, dataset=stream_to_data(stream), save=False)
T._replay = orig_replay
print(f"\n[sweep] training + reference configuration: {time.time() - t0:.0f} s", flush=True)

model, snap = cap["model"], cap["snap"]
# calls: [cal A, cal B, test replay]; keep the val and test argument tuples.
assert len(cap["calls"]) == 3, [k.get("desc") for _, k in cap["calls"]]
(val_a, val_k), (_, _), (test_a, test_k) = cap["calls"]

N = len(stream.types)
train_end = int(N * cfg.train_frac)
val_end = int(N * (cfg.train_frac + cfg.val_frac))
types = stream.types.numpy()
msg_np = stream.msg.numpy()
v_types, v_msg = types[train_end:val_end], msg_np[train_end:val_end]
t_types, t_msg = types[val_end:], msg_np[val_end:]
dirty_test = T._rule_baseline(t_msg).astype(bool)


def evaluate(name, half_life, max_shift, arm):
    _restore(model, snap)
    model.precursor_half_life = half_life
    model.precursor_max_shift = max_shift
    kw = dict(val_k); kw["desc"] = f"[{name}] cal A"
    sa, la = orig_replay(model, *val_a, **kw)
    thr_a, thr_d_a, thr_arm_a, _ = fit_thresholds(sa, la, v_types, v_msg, cfg)

    _restore(model, snap)
    model.precursor_half_life = half_life
    model.precursor_max_shift = max_shift
    kw = dict(val_k)
    kw.update(threshold=thr_a, threshold_dirty=thr_d_a,
              threshold_arm=(thr_arm_a if arm else None), desc=f"[{name}] cal B")
    sb, lb = orig_replay(model, *val_a, **kw)
    thr, thr_d, thr_unsup, _ = fit_thresholds(sb, lb, v_types, v_msg, cfg)

    kw = dict(test_k)
    kw.update(threshold=thr, threshold_dirty=thr_d,
              threshold_arm=(thr_unsup if arm else None), desc=f"[{name}] test")
    sc, lc = orig_replay(model, *test_a, **kw)

    ben, lat, theft = lc == 0, t_types == 3, t_types == 4
    old = (sc >= thr_d).astype(int)
    routed = routed_predict(sc, dirty_test, thr, thr_d)
    out = {
        "name": name, "half_life_h": half_life / HOUR, "max_shift": max_shift, "arm": arm,
        "threshold_clean": float(thr), "threshold_dirty": float(thr_d),
        "threshold_clean_unsup": float(thr_unsup),
        "agg_auc": float(roc_auc_score(lc, sc)), "agg_ap": float(average_precision_score(lc, sc)),
        "lat_auc": float(roc_auc_score(np.r_[np.zeros(ben.sum()), np.ones(lat.sum())],
                                       np.r_[sc[ben], sc[lat]])),
        "theft_auc": float(roc_auc_score(np.r_[np.zeros(ben.sum()), np.ones(theft.sum())],
                                         np.r_[sc[ben], sc[theft]])),
        "lat_recall_global": float(old[lat].mean()), "fpr_global": float(old[ben].mean()),
        "lat_recall_routed": float(routed[lat].mean()), "fpr_routed": float(routed[ben].mean()),
        "agg_recall_global": float(old[lc == 1].mean()),
    }
    np.savez(f"{OUT}_{name.replace(' ', '_').replace('=', '')}.npz",
             test_scores=sc, test_labels=lc, val_scores=sb, val_labels=lb,
             test_start=val_end, thr_global=thr_d, thr_clean=thr, thr_unsup=thr_unsup)
    print(f"[sweep] {name:22s} latAUC={out['lat_auc']:.4f} latR@1%={out['lat_recall_global']:.4f} "
          f"theftAUC={out['theft_auc']:.4f} aggAUC={out['agg_auc']:.4f} "
          f"thr_clean={thr:.6f}", flush=True)
    return out


def report_logits(scores, labels, tt):
    """How far, in nats, the laterals sit below the 1%-FPR threshold — the sizing that
    says whether a bounded precursor shift can reach them at all."""
    sc = np.clip(np.asarray(scores, dtype=np.float64), 1e-300, 1 - 1e-16)
    lg = np.log(sc) - np.log1p(-sc)          # anomaly logit, recovered from the score
    ben, lat = labels == 0, tt == 3
    thr_l = float(np.quantile(lg[ben], 1.0 - cfg.target_fpr))
    gap = thr_l - lg[lat]
    print(f"\n[sweep] anomaly logit: benign p50={np.median(lg[ben]):.2f} "
          f"p99={np.quantile(lg[ben], 0.99):.2f} max={lg[ben].max():.2f} | "
          f"lateral p50={np.median(lg[lat]):.2f} max={lg[lat].max():.2f}")
    print(f"[sweep] laterals saturated at score==1.0: {(np.asarray(scores)[lat] >= 1.0).sum()} "
          f"/ {lat.sum()} | benign saturated: {(np.asarray(scores)[ben] >= 1.0).sum()}")
    print(f"[sweep] nats below the 1%-FPR threshold (lateral): p10={np.quantile(gap, .1):.2f} "
          f"p25={np.quantile(gap, .25):.2f} p50={np.median(gap):.2f}")
    for sh in (1, 2, 4, 8, 16):
        print(f"[sweep]   a +{sh:2d} nat shift would lift {(gap <= sh).mean():.3f} "
              f"of the laterals over that threshold", flush=True)


rows = []
for name, hl, sh, arm in GRID:
    t1 = time.time()
    rows.append(evaluate(name, hl, sh, arm))
    rows[-1]["seconds"] = round(time.time() - t1, 1)
    json.dump(rows, open(f"{OUT}.json", "w"), indent=1)
    if len(rows) == 1:  # reference configuration: size the shift before reading the rest
        d = np.load(f"{OUT}_{GRID[0][0].replace(' ', '_').replace('=', '')}.npz")
        report_logits(d["test_scores"], d["test_labels"], t_types)

print("\n=== precursor sweep (seed 2000, one training run, paired) ===")
hdr = f"{'config':22s} {'latAUC':>7s} {'latR@1%':>8s} {'theftAUC':>9s} {'aggAUC':>7s} {'FPR':>7s} {'routedR':>8s}"
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r['name']:22s} {r['lat_auc']:7.4f} {r['lat_recall_global']:8.4f} "
          f"{r['theft_auc']:9.4f} {r['agg_auc']:7.4f} {r['fpr_global']:7.4f} "
          f"{r['lat_recall_routed']:8.4f}")
print(f"\nreference run (train_tgn): lat_recall_before={res0['lateral_recall_before']:.4f} "
      f"fpr_before={res0['fpr_before']:.4f} agg_auc={res0['agg_auc']:.4f}")
print(f"saved -> {OUT}.json")
