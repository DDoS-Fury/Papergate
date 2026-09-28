"""Kill-chain alert state on the OPA flow: ``/infer`` → OPA → ``/update`` | ``/deny``.

* ``deny_event`` records the event's alarm and touches nothing else (memory, message store,
  neighbour history, interaction counters, registry).
* ``record_alert`` keeps the latest alert time, so a retried or late call is harmless.
* PARITY: scoring every event with ``score_event(update=False)`` and ending it in
  ``commit_event`` (signal-clean = OPA ALLOW proxy) or ``deny_event`` (signal-dirty), each
  with the echoed ``event_alarm``, reproduces the offline ``_replay(batch_size=1)``: same
  scores, same alert state, same memory — and neither path mutates the node features.
* ``threshold_arm`` survives the save / load round trip; old artifacts load as ``None``.
* The HTTP layer: ``/infer`` returns the alarm, ``/deny`` records it.

    pytest tests/test_deny_alert.py
"""

import copy
import json
import random

import numpy as np
import torch
from fastapi import BackgroundTasks

from graphagate import serve_api
from graphagate.model.registry import NodeRegistry
from graphagate.serve_tgn import (
    build_model,
    commit_event,
    deny_event,
    event_alarm,
    load_model,
    record_alert,
    save_model,
    score_event,
    signal_dirty,
)
from graphagate.train_tgn import _replay

DEVICE = torch.device("cpu")
MSG_DIM = 10
HP = {
    "schema_version": 4,
    "capacity": 128,
    "node_feat_dim": 16,
    "msg_dim": MSG_DIM,
    "memory_dim": 32,
    "time_dim": 8,
    "num_hops": 3,
    "hash_buckets": 100,
    "hash_dim": 8,
    "hist_feat_dim": 6,
    "neighbor_size": 5,
}
T0 = 100_000  # evaluation events start after the warm-up stream


def _features(rng):
    """ZTA-shaped message: JA3 mostly intact, sensor probes rare, the rest continuous."""
    return ([1.0 if rng.random() < 0.8 else 0.0]
            + [1.0 if rng.random() < 0.1 else 0.0 for _ in range(3)]
            + [rng.random() for _ in range(MSG_DIM - 4)])


def _stream(n, seed=0, t0=100):
    rng = random.Random(seed)
    for i in range(n):
        u = rng.randrange(10)
        yield dict(
            key_user=f"u{u}",
            key_device=f"tpm:{u % 6}",
            key_dst=f"/r/{rng.randrange(8)}",
            timestamp=t0 + 37 * i,
            features=_features(rng),
            key_source=f"src:10.0.0.{rng.randrange(4)}",
            key_config=f"conf:{u % 4}",
        )


def _warmed():
    torch.manual_seed(0)
    model = build_model(dict(HP), DEVICE)
    reg = NodeRegistry(capacity=HP["capacity"])
    for ev in _stream(300):
        commit_event(model, reg, device=DEVICE, **ev)
    return model, reg


def _baseline(model, reg):
    """Everything a DENY must leave alone."""
    return {
        "memory": model.memory.memory.clone(),
        "last_update": model.memory.last_update.clone(),
        "msg_s": copy.deepcopy(model.memory.msg_s_store),
        "msg_d": copy.deepcopy(model.memory.msg_d_store),
        "neighbor": {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v))
                     for k, v in model.neighbor_loader.state().items()},
        "counters": (dict(model.last_contact), dict(model.pair_count), dict(model.src_count)),
        "node_feat": model.node_feat.clone(),
        "registry": len(reg),
    }


def _assert_same(a, b):
    for k in ("memory", "last_update", "node_feat"):
        assert torch.equal(a[k], b[k]), k
    for k in ("msg_s", "msg_d"):
        assert a[k].keys() == b[k].keys(), k
        for i in a[k]:
            for x, y in zip(a[k][i], b[k][i]):
                assert torch.equal(x, y), (k, i)
    assert a["neighbor"].keys() == b["neighbor"].keys()
    for k, v in a["neighbor"].items():
        w = b["neighbor"][k]
        assert (torch.equal(v, w) if torch.is_tensor(v) else v == w), k
    assert a["counters"] == b["counters"]
    assert a["registry"] == b["registry"]


def test_deny_records_alarm_and_nothing_else():
    model, reg = _warmed()
    before = _baseline(model, reg)
    clean = [1.0, 0.0, 0.0, 0.0] + [0.5] * (MSG_DIM - 4)

    warm_alerts = dict(model.recent_alert)  # sensor alarms of the warm-up stream
    deny_event(model, reg, "u1", "tpm:1", T0, clean, alarm=True)
    assert model.recent_alert == {**warm_alerts, reg.get("tpm:1"): T0}  # keyed on the device
    deny_event(model, reg, "u2", None, T0 + 1, clean, alarm=True)
    assert model.recent_alert[reg.get("u2")] == T0 + 1  # no device: keyed on the user
    snort = list(clean)
    snort[1] = 1.0
    deny_event(model, reg, "u3", "tpm:3", T0 + 2, snort)  # the sensor alarms on its own
    assert model.recent_alert[reg.get("tpm:3")] == T0 + 2

    alerts = dict(model.recent_alert)
    deny_event(model, reg, "never-seen", "tpm:never-seen", T0 + 3, clean)  # no alarm: no-op
    assert model.recent_alert == alerts
    assert reg.get("tpm:never-seen") is None
    _assert_same(before, _baseline(model, reg))


def test_record_alert_keeps_latest_time():
    model = build_model(dict(HP), DEVICE)
    record_alert(model, 7, 200)
    record_alert(model, 7, 150)  # late / retried call
    assert model.recent_alert[7] == 200
    record_alert(model, 7, 300)
    assert model.recent_alert[7] == 300


def test_opa_flow_matches_offline_replay():
    events = list(_stream(60, seed=1, t0=T0))
    # Thresholds from a dry run, so the stream has flagged, armed-only and sensor alarms.
    probe, probe_reg = _warmed()
    dry = np.array([score_event(probe, probe_reg, 2.0, device=DEVICE, update=False, **ev)[0]
                    for ev in events])
    thr, thr_dirty, thr_arm = (float(np.quantile(dry, q)) for q in (0.9, 0.8, 0.5))

    # Serving: /infer, then /update on the ALLOW proxy (signal-clean) or /deny.
    srv, reg = _warmed()
    srv.threshold_arm = thr_arm
    nf0 = srv.node_feat.clone()
    scores, kinds = [], set()
    for ev in events:
        s, flagged, _ = score_event(srv, reg, thr, device=DEVICE, update=False,
                                    threshold_dirty=thr_dirty, **ev)
        alarm = event_alarm(s, flagged=flagged, threshold_arm=thr_arm, features=ev["features"])
        kinds.add("flagged" if flagged else "armed" if s >= thr_arm and alarm else None)
        scores.append(s)
        if signal_dirty(ev["features"]):
            deny_event(srv, reg, ev["key_user"], ev["key_device"], ev["timestamp"],
                       ev["features"], alarm=alarm)
        else:
            commit_event(srv, reg, device=DEVICE, alarm=alarm, **ev)
    assert {"flagged", "armed"} <= kinds  # both non-sensor alarm routes were exercised
    assert torch.equal(srv.node_feat, nf0)  # trust (and every node feature) stays put

    # Offline replay on an identical warm model.
    off, reg_off = _warmed()
    idx = {k: torch.tensor([reg_off.get(ev[k]) for ev in events])
           for k in ("key_user", "key_device", "key_dst", "key_source", "key_config")}
    assert all(bool((v >= 0).all()) for v in idx.values())
    nf0 = off.node_feat.clone()
    off_scores, _ = _replay(
        off, idx["key_source"], idx["key_device"], idx["key_user"], idx["key_dst"],
        torch.tensor([ev["timestamp"] for ev in events]),
        torch.tensor([ev["features"] for ev in events], dtype=torch.float),
        torch.zeros(len(events), dtype=torch.long), DEVICE, config_nodes=idx["key_config"],
        threshold=thr, threshold_dirty=thr_dirty, threshold_arm=thr_arm,
        gate_by_label=False, batch_size=1, desc="parity",
    )
    assert torch.equal(off.node_feat, nf0)

    np.testing.assert_allclose(scores, off_scores, atol=1e-6)
    assert srv.recent_alert == off.recent_alert and srv.recent_alert
    torch.testing.assert_close(srv.memory.memory, off.memory.memory, atol=1e-5, rtol=1e-5)
    assert torch.equal(srv.memory.last_update, off.memory.last_update)


def test_threshold_arm_round_trip(tmp_path):
    model, reg = _warmed()
    hp = dict(HP)
    ckpt, stats = tmp_path / "m.pt", tmp_path / "s.json"
    model.threshold_arm = 0.25
    save_model(model, reg, 0.9, hp, ckpt, stats, threshold_dirty=0.8)
    assert load_model(ckpt, stats, DEVICE)[0].threshold_arm == 0.25

    data = json.loads(stats.read_text())
    del data["threshold_arm"]  # an artifact from before the arm threshold was persisted
    stats.write_text(json.dumps(data))
    assert load_model(ckpt, stats, DEVICE)[0].threshold_arm is None


def test_api_infer_returns_alarm_and_deny_records_it():
    model, reg = _warmed()
    model.threshold_arm = 0.0  # every score arms
    st = serve_api.STATE
    st.model, st.registry, st.hp, st.device = model, reg, dict(HP), DEVICE
    st.threshold, st.threshold_dirty = 2.0, 2.0  # never flagged: the alarm is the arm alone
    ev = next(_stream(1, seed=3, t0=T0))
    ev["features"][:4] = [1.0, 0.0, 0.0, 0.0]
    try:
        out = serve_api.infer(serve_api.EventIn(**ev), BackgroundTasks())
        assert out.alarm and not out.is_anomaly
        before = _baseline(model, reg)
        serve_api.deny(serve_api.EventIn(**ev, alarm=out.alarm))
        assert model.recent_alert[reg.get(ev["key_device"])] == ev["timestamp"]
        _assert_same(before, _baseline(model, reg))
    finally:
        st.model = st.registry = None
