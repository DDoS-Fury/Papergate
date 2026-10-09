"""Incident-level detection (eval_common.incident_metrics) on hand-built streams."""

import numpy as np

from graphagate.eval_common import incident_metrics


def test_blocked_and_served():
    # Incident 0: lateral x4, second one flagged -> blocked, 1 served (share 1/4).
    # Incident 1: lateral x2, never flagged -> not blocked, 2 served (share 1).
    # Benign events (incident -1) in between are ignored, flagged or not.
    types = np.array([3, 0, 3, 3, 0, 3, 3, 3])
    incident = np.array([0, -1, 0, 1, -1, 0, 1, 0])
    pred = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    m = incident_metrics(pred, types, incident, 3)
    assert m["n_incidents"] == 2 and m["n_events"] == 6
    assert m["blocked"] == 0.5
    assert m["served_median"] == 1.5  # median of [1, 2]
    assert np.isclose(m["served_share"], (1 / 4 + 1.0) / 2)


def test_block_types_credit_chain_events():
    # Recon (type 2) flagged before any lateral event of the same episode: blocked with 0
    # lateral events served only when recon may block; a flagged policy denial (type 1) of
    # another episode does not leak across incidents.
    types = np.array([2, 3, 3, 1, 3])
    incident = np.array([0, 0, 0, 1, 1])
    pred = np.array([1, 0, 0, 1, 0])
    lat_only = incident_metrics(pred, types, incident, 3)
    chain = incident_metrics(pred, types, incident, 3, block_types=(1, 2, 3))
    assert lat_only["blocked"] == 0.0 and lat_only["served_median"] == 1.5
    assert chain["blocked"] == 1.0 and chain["served_median"] == 0.0


def test_no_incidents():
    m = incident_metrics(np.zeros(3), np.zeros(3, dtype=int), -np.ones(3, dtype=int), 4)
    assert m["n_incidents"] == 0 and np.isnan(m["blocked"])
