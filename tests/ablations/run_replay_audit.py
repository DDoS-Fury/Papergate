"""Replay audit of the two leakage controls of the paper (Appendix A.4.3, points 5 and 6).

Point 1 (precursor selection). The kill-chain precursor half-life / shift were chosen on the
TEST split (tasks/todo.md). The raw edge logits do not depend on the precursor (the commit gate
reads signals, not scores), so ONE real training run is enough: its val / test edge logits are
captured and the precursor feedback loop (shift -> score -> alarm -> recent_alert) is re-simulated
as plain scalar code over a (half-life, shift) grid, with the thresholds re-calibrated on the
validation slice for every cell, exactly as ``train_tgn`` does. The selection rule is fixed in
advance: the cell with the highest VALIDATION lateral AUC. Test numbers are reported for every
cell and never used to choose. The simulation is checked against the scores of the real run
(default parameters) before any cell is trusted.

Point 2 (commit gate). The offline replay commits every signal-clean event (proxy of OPA ALLOW);
OPA would DENY the policy-violation and benign-denial classes, which are signal-clean. Arm ``B``
re-runs the whole pipeline with a policy-aware gate (``not signal_dirty and policy_allows``), arm
``A`` is the current gate and ``A2`` a replicate of ``A`` (same seed): ``A`` vs ``A2`` measures the
run-to-run noise (CUDA scatter-add is not deterministic), ``A`` vs ``B`` the effect of the gate.
The gate B uses no label: the allow mask is the reference policy applied to (role, method, uri).

    python tests/ablations/run_replay_audit.py --out runs/replay_audit            # full, GPU
    python tests/ablations/run_replay_audit.py --events 6000 --epochs 1 \
        --ft-epochs 1 --arms A,B --cpu --out /tmp/audit_smoke                          # Mac smoke test

All runs use ``save=False``: the deployable artifact is never touched.
"""

import argparse
import dataclasses
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from graphagate import train_tgn as T
from graphagate.calibration import routed_predict
from graphagate.config import TGNConfig
from graphagate.data.access_policy import NUM_BASE_ROUTES, build_resource_universe, policy_allows
from graphagate.data.stream_synthetic import ROLES
from graphagate.serve_tgn import (
    anomaly_score,
    combine_edge_logits,
    event_alarm,
    precursor_shift,
    record_alert,
)

HOUR = 3600.0
HALF_LIVES_H = [1, 3, 6, 12, 24, 48, 72, 168]
SHIFTS = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
LATERAL, THEFT = 3, 4


# ---------------------------------------------------------------- policy mask (gate B)
def policy_allow_mask(cfg, data):
    """Bool ``[N]``: the reference PDP allows the request (role, method, uri); no label used."""
    route_methods, security_matrix, uris, _ = build_resource_universe(
        cfg.num_resources - NUM_BASE_ROUTES, cfg.seed)
    for r, uri in enumerate(uris):
        assert data.keys[data.neg_lo + r] == uri, "resource catalogue != stream keys"
    msg, dst = data.msg.numpy(), data.dst.numpy()
    cache, out = {}, np.empty(len(dst), dtype=bool)
    for i in range(len(dst)):
        role = ROLES[int(round(float(msg[i, 5]) * (len(ROLES) - 1)))]
        key = (role, int(msg[i, 4]), int(dst[i]) - data.neg_lo)
        if key not in cache:
            cache[key] = bool(policy_allows(role, key[1], uris[key[2]], route_methods, security_matrix))
        out[i] = cache[key]
    return out


# ---------------------------------------------------------------- replay recorder
class Recorder:
    """Stands in for ``train_tgn._replay``: forces the edge logits out, records every call and
    (arm B) injects the policy-aware commit gate into the non-label-gated replays."""

    def __init__(self, orig, data, allow=None):
        self.orig, self.data, self.allow, self.calls, self.model = orig, data, allow, [], None

    def __call__(self, model, source_nodes, device_nodes, user, dst, t, msg, y, device, **kw):
        n = int(user.shape[0])
        # ``user`` is a view of the stream: its storage offset is the slice's start index.
        off = (user.data_ptr() - self.data.user.data_ptr()) // user.element_size()
        assert 0 <= off <= len(self.data.user) - n
        if self.allow is not None and not kw.get("gate_by_label", False):
            kw["policy_allow"] = self.allow[off:off + n]
        kw["return_edge_logits"] = True
        scores, labels, extra = self.orig(model, source_nodes, device_nodes, user, dst, t, msg, y,
                                          device, **kw)
        self.model = model
        self.calls.append(dict(off=off, n=n, thr=kw.get("threshold"), scores=scores,
                               edge={k: v.copy() for k, v in extra["edge_logits"].items()},
                               shift=extra["shift"].copy()))
        return scores, labels, extra


def run_arm(name, cfg, data, allow, out_dir):
    orig = T._replay
    rec = Recorder(orig, data, allow)
    T._replay = rec
    try:
        t0 = time.time()
        m = T.train_tgn(cfg, dataset=data, save=False, return_scores=True,
                        log_dir=os.path.join(out_dir, f"arm_{name}"))
        print(f"[arm {name}] done in {time.time() - t0:.0f}s")
    finally:
        T._replay = orig
    return m, rec


# ---------------------------------------------------------------- scalar re-simulation
def simulate(comb, msg_rows, dirty, actor, t, *, thr, thr_dirty, thr_arm, half_life, max_shift):
    """The phase-2 loop of ``train_tgn._replay`` on precomputed combined logits.

    Returns ``(scores, shifts)``. ``comb`` is float32 like the real run, so the per-event
    ``comb[i] + shift`` rounds identically.
    """
    m = SimpleNamespace(use_precursor=True, recent_alert={}, precursor_half_life=half_life,
                        precursor_max_shift=max_shift)
    n = len(comb)
    scores, shifts = np.empty(n), np.empty(n)
    for i in range(n):
        sh = precursor_shift(m, actor[i], t[i])
        shifts[i] = sh
        sc = float(anomaly_score(comb[i] + sh))
        scores[i] = sc
        eff = thr_dirty if (thr_dirty is not None and dirty[i]) else thr
        if event_alarm(sc, flagged=eff is not None and sc >= eff, threshold_arm=thr_arm,
                       features=msg_rows[i]):
            record_alert(m, actor[i], t[i])
    return scores, shifts


class Split:
    """Static view of one stream slice."""

    def __init__(self, data, lo, hi, comb_passes):
        self.passes = comb_passes   # combined logits of each replay of this slice (val: A, B1..)
        self.comb = comb_passes[-1]
        self.msg = data.msg[lo:hi].numpy()
        self.rows = data.msg[lo:hi].tolist()
        self.dirty = T._rule_baseline(self.msg).astype(bool)
        self.actor = data.device_nodes[lo:hi].tolist()
        self.t = data.t[lo:hi].tolist()
        self.types = data.types[lo:hi].numpy()
        self.labels = data.y[lo:hi].numpy()


def calibrate_and_test(val, test, cfg, half_life, max_shift):
    """Mirror of the calibration + test section of ``train_tgn`` for one precursor setting."""
    kw = dict(half_life=half_life, max_shift=max_shift)
    # Pass A: sensor-only arming, scores recomputed vectorised from the logits (as train_tgn does).
    _, shift_a = simulate(val.passes[0], val.rows, val.dirty, val.actor, val.t,
                          thr=None, thr_dirty=None, thr_arm=None, **kw)
    scores = anomaly_score(val.passes[0] + shift_a)
    thr, thr_d, thr_u, _ = T.fit_thresholds(scores, val.labels, val.types, val.msg, cfg)
    thr_arm = thr_u if cfg.precursor_arm_on_score else None
    for k in range(cfg.calib_iters):
        scores, _ = simulate(val.passes[1 + k], val.rows, val.dirty, val.actor, val.t,
                             thr=thr, thr_dirty=thr_d, thr_arm=thr_arm, **kw)
        thr, thr_d, thr_u, _ = T.fit_thresholds(scores, val.labels, val.types, val.msg, cfg)
    t_scores, _ = simulate(test.comb, test.rows, test.dirty, test.actor, test.t,
                           thr=thr, thr_dirty=thr_d, thr_arm=thr_arm, **kw)
    return dict(thr=thr, thr_dirty=thr_d, val_scores=scores, test_scores=t_scores)


def type_auc(scores, types, labels, type_id):
    """AUC of benign (label 0) vs events of ``type_id``; nan when the slice has none."""
    sel = (labels == 0) | (types == type_id)
    pos = (types[sel] == type_id).astype(int)
    if pos.sum() == 0:
        return float("nan")
    return float(roc_auc_score(pos, scores[sel]))


def cell_metrics(res, val, test):
    preds = routed_predict(res["test_scores"], test.dirty, res["thr"], res["thr_dirty"])
    lat, ben = test.types == LATERAL, test.labels == 0
    return dict(
        val_lat_auc=type_auc(res["val_scores"], val.types, val.labels, LATERAL),
        val_theft_auc=type_auc(res["val_scores"], val.types, val.labels, THEFT),
        val_agg_auc=float(roc_auc_score(val.labels, res["val_scores"])),
        test_lat_auc=type_auc(res["test_scores"], test.types, test.labels, LATERAL),
        test_theft_auc=type_auc(res["test_scores"], test.types, test.labels, THEFT),
        test_agg_auc=float(roc_auc_score(test.labels, res["test_scores"])),
        test_lat_recall=float(preds[lat].mean()) if lat.any() else float("nan"),
        test_fpr=float(preds[ben].mean()),
        thr=res["thr"], thr_dirty=res["thr_dirty"],
    )


def edge_to_comb(model, edge):
    return combine_edge_logits(model, {k: torch.as_tensor(v) for k, v in edge.items()}).numpy()


# ---------------------------------------------------------------- Point 1: precursor sweep
def precursor_sweep(m, rec, cfg, data, n_train, n_val):
    n = len(data.dst)
    val_end = n_train + n_val
    first, last = rec.calls[0], rec.calls[-1]
    assert (first["off"], first["n"]) == (n_train, n_val), "first replay is not the val pass A"
    assert (last["off"], last["n"]) == (val_end, n - val_end), "last replay is not the test replay"
    assert cfg.edge_calibration and cfg.calib_iters >= 1, "the simulation assumes the defaults"

    n_val_passes = 1 + cfg.calib_iters
    val_calls = rec.calls[:n_val_passes]
    assert len(rec.calls) == n_val_passes + 1, f"unexpected replay count {len(rec.calls)}"
    val = Split(data, n_train, val_end, [edge_to_comb(rec.model, c["edge"]) for c in val_calls])
    test = Split(data, val_end, n, [edge_to_comb(rec.model, last["edge"])])

    # (1) The raw logits must not depend on the precursor parameters (the commit gate reads
    # signals only), but two replays of the same slice from the same snapshot should also agree
    # with each other. Report how well they do: a gap would be a restore flaw, not the precursor.
    for k, c in enumerate(val_calls[1:], 1):
        d = np.abs(val.passes[k] - val.passes[0])
        print(f"[check] val pass B{k} vs pass A (same snapshot, same gate) | combined logit: "
              f"max|d|={d.max():.3e} mean|d|={d.mean():.3e} frac(d>1e-4)={(d > 1e-4).mean():.4f}")

    # (2) the simulation must reproduce the real run at the default parameters
    base = calibrate_and_test(val, test, cfg, cfg.precursor_half_life, cfg.precursor_max_shift)
    real_val = rec.calls[-2]["scores"]   # last pass B (calib_iters >= 1), as in train_tgn
    d_val = float(np.abs(base["val_scores"] - real_val).max())
    d_test = float(np.abs(base["test_scores"] - last["scores"]).max())
    d_thr = abs(base["thr"] - m["threshold"])
    print(f"[check] sim vs real run at default | max|d val score|={d_val:.2e} "
          f"max|d test score|={d_test:.2e} |d thr_clean|={d_thr:.2e}")
    faithful = max(d_val, d_test, d_thr) < 1e-6
    if not faithful:
        print("[WARN] the simulation does NOT reproduce the real run: do not trust the sweep")

    n_lat_val = int((val.types == LATERAL).sum())
    n_lat_test = int((test.types == LATERAL).sum())
    print(f"[sweep] lateral events: val={n_lat_val} test={n_lat_test} | "
          f"theft: val={int((val.types == THEFT).sum())} test={int((test.types == THEFT).sum())}")
    rows = []
    for hl_h in HALF_LIVES_H:
        for sh in SHIFTS:
            res = calibrate_and_test(val, test, cfg, hl_h * HOUR, sh)
            rows.append(dict(half_life_h=hl_h, shift=sh, **cell_metrics(res, val, test)))
    return dict(rows=rows, faithful=faithful, n_lat_val=n_lat_val, n_lat_test=n_lat_test,
                default=(cfg.precursor_half_life / HOUR, cfg.precursor_max_shift))


def print_sweep(sw):
    rows = sw["rows"]
    ok = [r for r in rows if not np.isnan(r["val_lat_auc"])]
    pick = max(ok, key=lambda r: r["val_lat_auc"]) if ok else None
    pick_agg = max(rows, key=lambda r: r["val_agg_auc"])
    d_hl, d_sh = sw["default"]
    print(f"\n{'hl[h]':>6} {'shift':>5} | {'valLatAUC':>9} {'valAggAUC':>9} | {'testLatAUC':>10} "
          f"{'testTheftAUC':>12} {'latRecall':>9} {'FPR':>7}")
    for r in rows:
        tag = ""
        if pick is r:
            tag += "  <= VAL-SELECTED"
        if (r["half_life_h"], r["shift"]) == (d_hl, d_sh):
            tag += "  <= current default"
        print(f"{r['half_life_h']:6g} {r['shift']:5g} | {r['val_lat_auc']:9.4f} {r['val_agg_auc']:9.4f} | "
              f"{r['test_lat_auc']:10.4f} {r['test_theft_auc']:12.4f} {r['test_lat_recall']:9.4f} "
              f"{r['test_fpr']:7.4f}{tag}")
    if pick is None:
        print("\nNo lateral events in the validation slice: the rule cannot be applied.")
        return None
    dflt = next((r for r in rows if (r["half_life_h"], r["shift"]) == (d_hl, d_sh)), None)
    print(f"\nVAL-selected (max val lateral AUC): half_life={pick['half_life_h']}h shift={pick['shift']} "
          f"-> test lat AUC {pick['test_lat_auc']:.4f}, recall {pick['test_lat_recall']:.4f}, "
          f"FPR {pick['test_fpr']:.4f}")
    if dflt is not None:
        print(f"Current default ({d_hl:g}h / {d_sh:g}): val lat AUC {dflt['val_lat_auc']:.4f}, "
              f"test lat AUC {dflt['test_lat_auc']:.4f}, recall {dflt['test_lat_recall']:.4f}, "
              f"FPR {dflt['test_fpr']:.4f}")
    print(f"Val-AGGREGATE pick (the pick todo.md mentions): half_life={pick_agg['half_life_h']}h "
          f"shift={pick_agg['shift']}")
    print(f"Best test lat AUC over the grid (for information only, never used to choose): "
          f"{max(r['test_lat_auc'] for r in rows if not np.isnan(r['test_lat_auc'])):.4f}")
    same = (pick["half_life_h"], pick["shift"]) == (d_hl, d_sh)
    print("=> val selection " + ("COINCIDES with" if same else "DIFFERS from") + " the current default")
    return pick


# ---------------------------------------------------------------- Point 2: gate comparison
def arm_row(m):
    pt = m["per_type"]
    g = lambda k, f: pt.get(k, {}).get(f, float("nan"))
    return dict(
        agg_auc=m["agg_auc"], lat_auc=g("lateral", "auc"), lat_recall=g("lateral", "recall"),
        theft_auc=g("cred-theft", "auc"), policy_auc=g("policy", "auc"),
        denied_auc=g("benign-denied", "auc"), fpr=m["fpr_after"],
        thr_clean=m["threshold"], thr_dirty=m["threshold_dirty"],
    )


def print_arms(arms):
    keys = list(next(iter(arms.values())).keys())
    print("\n" + f"{'metric':>12} | " + " ".join(f"{a:>10}" for a in arms))
    for k in keys:
        print(f"{k:>12} | " + " ".join(f"{arms[a][k]:10.4f}" for a in arms))
    if "A" in arms and "B" in arms:
        print("\nDelta B - A" + ("" if "A2" not in arms else "   (noise floor: |A2 - A|)"))
        for k in keys:
            d = arms["B"][k] - arms["A"][k]
            nz = f"   |A2-A|={abs(arms['A2'][k] - arms['A'][k]):.4f}" if "A2" in arms else ""
            print(f"{k:>12} | {d:+10.4f}{nz}")


def gate_counts(data, allow, n_train, n_val):
    """Events the gate A commits although the PDP would deny them (label-free)."""
    dirty = T._rule_baseline(data.msg.numpy()).astype(bool)
    bad = (~dirty) & (~allow)
    types = data.types.numpy()
    for name, lo, hi in (("val", n_train, n_train + n_val), ("test", n_train + n_val, len(dirty))):
        sel = slice(lo, hi)
        c = int((~dirty[sel]).sum())
        print(f"[gate] {name}: gate A commits {c}, of which PDP-denied {int(bad[sel].sum())} "
              f"({bad[sel].sum() / max(c, 1):.2%}) | types of those: "
              f"{dict(zip(*np.unique(types[sel][bad[sel]], return_counts=True)))}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--events", type=int, default=None, help="stream length (default: TGNConfig)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--ft-epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--arms", default="A,A2,B", help="comma list of A, A2, B")
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--cpu", action="store_true",
                    help="hide MPS (the model needs ops MPS lacks: pdist, int64 scatter_reduce)")
    ap.add_argument("--out", default=os.path.join("runs", "replay_audit"))
    a = ap.parse_args()
    if a.cpu:
        torch.backends.mps.is_available = lambda: False  # train_tgn then falls back to the CPU

    over = {k: v for k, v in dict(num_events=a.events, epochs=a.epochs, ft_epochs=a.ft_epochs,
                                  seed=a.seed).items() if v is not None}
    cfg = dataclasses.replace(TGNConfig(), **over)
    os.makedirs(a.out, exist_ok=True)
    arms_req = [x.strip() for x in a.arms.split(",") if x.strip()]
    assert set(arms_req) <= {"A", "A2", "B"}, arms_req
    print(f"[cfg] events={cfg.num_events} epochs={cfg.epochs}+{cfg.ft_epochs} seed={cfg.seed} "
          f"precursor={cfg.precursor_half_life / HOUR:g}h/{cfg.precursor_max_shift:g} "
          f"calib_iters={cfg.calib_iters} arms={arms_req}")

    data = T._synthetic_stream_data(cfg)
    n = len(data.dst)
    n_train, n_val = int(n * cfg.train_frac), int(n * cfg.val_frac)
    allow = policy_allow_mask(cfg, data)
    gate_counts(data, allow, n_train, n_val)

    results, arms, recs = {}, {}, {}
    for name in arms_req:
        m, rec = run_arm(name, cfg, data, allow if name == "B" else None, a.out)
        arms[name], recs[name] = arm_row(m), (m, rec)
    print_arms(arms)
    results["arms"] = arms

    if not a.skip_sweep:
        base = "A" if "A" in recs else next(iter(recs))
        m, rec = recs[base]
        sw = precursor_sweep(m, rec, cfg, data, n_train, n_val)
        pick = print_sweep(sw)
        results["sweep"] = dict(sw, pick=pick)

    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1, default=float)
    print(f"\n[done] {os.path.join(a.out, 'results.json')}")


if __name__ == "__main__":
    main()
