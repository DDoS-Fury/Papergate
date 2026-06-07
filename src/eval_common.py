"""Shared causal, runtime-derivable signals for FAIR baseline comparison + cold-start.

These mirror — in a simple, non-relational, batch form — the two tabular signals the
streaming TGN maintains online:

  * the interaction-history counters (per-pair / per-src access counts), and
  * the kill-chain precursor (a per-entity, time-decayed "recently alerted" prior).

Giving the baselines (Isolation Forest / One-Class SVM / static GNN) the *same* signals
means the TGN's remaining advantage on lateral movement is attributable to its
temporal-graph machinery (recurrent memory + temporal neighbourhood) rather than to the
counters or the precursor heuristic. Everything here is **causal** (uses only events
strictly before each event) and **benign-gated** (only benign events advance the counts),
exactly like the TGN's online state.
"""

from __future__ import annotations

import numpy as np


def causal_hist_features(src, dst, y) -> np.ndarray:
    """Per-event ``[log1p(pair_count), log1p(src_count), pair/(src+1)]`` (N, 3).

    Counts only ground-truth-benign events strictly *before* each event — the batch,
    causal analogue of :meth:`ZTATemporalGraphNetwork.compute_hist_feats`.
    """
    src = np.asarray(src); dst = np.asarray(dst); y = np.asarray(y)
    n = len(src)
    feats = np.zeros((n, 3), dtype=np.float64)
    pair: dict = {}
    srcc: dict = {}
    for i in range(n):
        s = int(src[i]); d = int(dst[i])
        pc = pair.get((s, d), 0); sc = srcc.get(s, 0)
        feats[i, 0] = np.log1p(pc)
        feats[i, 1] = np.log1p(sc)
        feats[i, 2] = pc / (sc + 1.0)
        if y[i] == 0:  # benign-gated, like the TGN's predict-then-update commit
            pair[(s, d)] = pc + 1
            srcc[s] = sc + 1
    return feats


def causal_src_seen(src, y) -> np.ndarray:
    """Boolean (N,): has this src had >=1 benign event strictly before? (cold-start split)."""
    src = np.asarray(src); y = np.asarray(y)
    seen: set = set()
    out = np.zeros(len(src), dtype=bool)
    for i in range(len(src)):
        s = int(src[i])
        out[i] = s in seen
        if y[i] == 0:
            seen.add(s)
    return out


def causal_precursor_factor(src, t, msg, half_life: float, max_boost: float) -> np.ndarray:
    """Per-event multiplicative score factor (N,) from the kill-chain precursor.

    Mirrors :func:`graphagate.serve_tgn.precursor_boost` causally so the baselines get the
    SAME prior: ``1 + max_boost * 0.5**(Δt/half_life)`` while a Snort alert on the same src
    is recent, else ``1.0``. Armed by the observable Snort flag (``msg[:, 1] > 0.5``) — the
    signal a recon event fires — computed before arming on the current event so an event is
    never boosted by itself.
    """
    src = np.asarray(src); t = np.asarray(t)
    snort = np.asarray(msg)[:, 1] > 0.5
    n = len(src)
    last: dict = {}
    fac = np.ones(n, dtype=np.float64)
    for i in range(n):
        s = int(src[i])
        la = last.get(s)
        if la is not None:
            dt = max(0.0, float(t[i]) - float(la))
            fac[i] = 1.0 + max_boost * 0.5 ** (dt / half_life)
        if snort[i]:
            last[s] = t[i]
    return fac
