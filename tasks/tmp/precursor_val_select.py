"""Select the kill-chain precursor (half-life, max shift) on the VALIDATION slice, then run
the test slice ONCE with the selected pair.

Replaces the protocol of precursor_sweep.py, which ranked its grid by test-slice lateral AUC
(the shipped 72 h / 4 nats came from there). Selection objective, fixed before any result:
lateral AUC (benign vs type-3 events) on the pass-B validation scores — the scores the
decision thresholds are fitted on. Arming is always on (the serving path now arms on the
persisted threshold_arm too); the shift=0 row is a candidate like the others, so "no prior"
wins if the prior does not help on validation.

Same mechanism as precursor_sweep.py: the precursor is a serving-time prior, never a trained
input, so one training run serves the whole grid; each candidate re-runs calibration passes
A and B from the pre-calibration snapshot. The reference training run inside train_tgn
also replays the test slice with the config defaults: those numbers are discarded, never
printed or used.

Run (workstation, ~22 min training + ~10 min per candidate):
  docker compose run --rm --no-deps -T -v "$PWD/tasks:/app/tasks" \
      --entrypoint python train-tgn /app/tasks/tmp/precursor_val_select.py
"""
import copy, dataclasses, json, time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import graphagate.train_tgn as T
from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg
from graphagate.train_tgn import fit_thresholds, stream_to_data

OUT = "/app/tasks/tmp/precursor_val_select"
HOUR = 3600.0
# (half_life seconds, max_shift nats); shift 0 = prior off (its half-life is irrelevant).
GRID = [(600.0, 0.0)] + [(hl * HOUR, sh) for hl in (6, 24, 72, 168) for sh in (2.0, 4.0, 8.0)]

cap = {"calls": [], "model": None, "snap": None}
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


def recorder(model, *a, **k):
    """Capture the exact slices train_tgn replays, so the selection re-runs them verbatim."""
    if cap["snap"] is None:          # first call = calibration pass A, pre-calibration state
        cap["model"], cap["snap"] = model, _snapshot(model)
    cap["calls"].append((a, dict(k)))
    return orig_replay(model, *a, **k)


cfg = dataclasses.replace(TGNConfig(), seed=2000)
stream = generate_streaming_data(**stream_kwargs_from_cfg(cfg))
T._replay = recorder
t0 = time.time()
T.train_tgn(cfg, dataset=stream_to_data(stream), save=False)  # test metrics discarded
T._replay = orig_replay
print(f"\n[select] training + reference run: {time.time() - t0:.0f} s", flush=True)

model, snap = cap["model"], cap["snap"]
assert len(cap["calls"]) == 3, [k.get("desc") for _, k in cap["calls"]]
(val_a, val_k), (_, _), (test_a, test_k) = cap["calls"]
# Pass A was recorded with return_edge_logits=True (train_tgn fits the edge calibration on
# it). The reference run already installed that calibration, and raw edge logits do not
# depend on the precursor, so the re-runs only need the scores.
val_k.pop("return_edge_logits", None)

N = len(stream.types)
train_end = int(N * cfg.train_frac)
val_end = int(N * (cfg.train_frac + cfg.val_frac))
types = stream.types.numpy()
msg_np = stream.msg.numpy()
v_types, v_msg = types[train_end:val_end], msg_np[train_end:val_end]
t_types = types[val_end:]
print(f"[select] validation slice: {int((v_types == 0).sum())} benign, "
      f"{int((v_types == 3).sum())} lateral", flush=True)


def lateral_auc(scores, tt):
    ben, lat = tt == 0, tt == 3
    return float(roc_auc_score(np.r_[np.zeros(ben.sum()), np.ones(lat.sum())],
                               np.r_[scores[ben], scores[lat]]))


def calibrate(half_life, max_shift):
    """Passes A and B on the validation slice, exactly as train_tgn runs them."""
    model.precursor_half_life, model.precursor_max_shift = half_life, max_shift
    _restore(model, snap)
    sa, la = orig_replay(model, *val_a, **{**val_k, "desc": "cal A"})
    thr_a, thr_d_a, thr_arm_a, _ = fit_thresholds(sa, la, v_types, v_msg, cfg)
    _restore(model, snap)
    sb, lb = orig_replay(model, *val_a, **{**val_k, "threshold": thr_a,
                                           "threshold_dirty": thr_d_a,
                                           "threshold_arm": thr_arm_a, "desc": "cal B"})
    thr, thr_d, thr_arm, _ = fit_thresholds(sb, lb, v_types, v_msg, cfg)
    return sb, (thr, thr_d, thr_arm)


rows = []
for hl, sh in GRID:
    t1 = time.time()
    sb, (thr, thr_d, thr_arm) = calibrate(hl, sh)
    rows.append({"half_life_h": hl / HOUR, "max_shift": sh,
                 "val_lat_auc": lateral_auc(sb, v_types),
                 "threshold_clean": thr, "threshold_dirty": thr_d, "threshold_arm": thr_arm,
                 "seconds": round(time.time() - t1, 1)})
    print(f"[select] hl={hl / HOUR:6.1f}h sh={sh:3.1f}  val latAUC={rows[-1]['val_lat_auc']:.4f}",
          flush=True)
    json.dump({"rows": rows}, open(f"{OUT}.json", "w"), indent=1)

best = max(rows, key=lambda r: r["val_lat_auc"])
print(f"\n[select] SELECTED on validation: hl={best['half_life_h']}h sh={best['max_shift']} "
      f"(val latAUC={best['val_lat_auc']:.4f})", flush=True)

# The single test run: the calibration state of the selected pair, then the test slice.
sb, (thr, thr_d, thr_arm) = calibrate(best["half_life_h"] * HOUR, best["max_shift"])
model.recent_alert.clear()  # as train_tgn: calibration alerts do not pre-condition the test
sc, lc = orig_replay(model, *test_a, **{**test_k, "threshold": thr, "threshold_dirty": thr_d,
                                        "threshold_arm": thr_arm, "desc": "test (selected)"})
test = {"lat_auc": lateral_auc(sc, t_types), "theft_auc": float(roc_auc_score(
    np.r_[np.zeros((t_types == 0).sum()), np.ones((t_types == 4).sum())],
    np.r_[sc[t_types == 0], sc[t_types == 4]])), "agg_auc": float(roc_auc_score(lc, sc))}
print(f"[select] TEST (selected config only): latAUC={test['lat_auc']:.4f} "
      f"theftAUC={test['theft_auc']:.4f} aggAUC={test['agg_auc']:.4f}", flush=True)
json.dump({"objective": "val lateral AUC (pass B)", "rows": rows, "selected": best,
           "test": test}, open(f"{OUT}.json", "w"), indent=1)
print(f"saved -> {OUT}.json")
