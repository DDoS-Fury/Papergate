"""Evaluation helpers shared by the TGN pipeline and the baselines.

- :func:`causal_hist_features` / :func:`causal_precursor_factor`: batch mirrors of the TGN's
  online history counters and kill-chain prior, so the baselines (Isolation Forest,
  One-Class SVM, static GNN) get the same family of signals (device actor only, no binding
  counters). Causal (events strictly before) and benign-gated up to ``label_horizon``.
- :func:`causal_src_seen`: warmed / cold partition for the cold-start split.
- :func:`tail_stream`: data-budget slicing of a stream.
- :func:`binary_metrics`: precision / recall at a threshold.
"""

from __future__ import annotations

import dataclasses

import numpy as np

# Per-event tensors of a generated stream (``SyntheticStream``); everything else in it
# (node features, keys, node-space layout) describes the entity space and must not be cut.
_PER_EVENT = ("source", "config", "device", "user", "dst", "t", "msg", "y", "types", "scenario")


def tail_stream(stream, cfg, n_train: int):
    """Keep only the last ``n_train`` training events, leaving validation and test intact.

    The data-budget experiment asks how much benign history a *new deployment* needs, so
    the budget is taken from the events closest to the validation window (no temporal gap)
    and the validation / test windows must stay bit-identical to the full stream's — every
    method is then scored on the same events.

    Returns ``(stream', cfg')``. ``cfg'`` carries ``train_frac`` / ``val_frac`` chosen so
    that ``int(n' * frac)`` lands exactly on ``n_train`` and on the original validation
    length (asserted); the ``+ 0.5`` keeps the product away from the ``int`` boundary.
    Node ids, features and keys are untouched: entities absent from the kept events simply
    have no history, as in a fresh deployment.
    """
    n = len(stream.y)
    train_end = int(n * cfg.train_frac)
    val_len = int(n * cfg.val_frac)
    if not 0 < n_train <= train_end:
        raise ValueError(f"n_train must be in (0, {train_end}], got {n_train}")
    start = train_end - n_train
    n2 = n - start
    tf, vf = (n_train + 0.5) / n2, (val_len + 0.5) / n2
    assert int(n2 * tf) == n_train and int(n2 * vf) == val_len
    cut = {k: getattr(stream, k)[start:] for k in _PER_EVENT}
    return (dataclasses.replace(stream, **cut),
            dataclasses.replace(cfg, num_events=n2, train_frac=tf, val_frac=vf))


def causal_hist_features(src, dst, y, *, label_horizon: int | None = None) -> np.ndarray:
    """Per-event ``[log1p(pair_count), log1p(src_count), pair/(src+1)]`` (N, 3).

    Counts only ground-truth-benign events strictly *before* each event — the batch,
    causal analogue of :meth:`ZTATemporalGraphNetwork.compute_hist_feats`.

    ``label_horizon`` (in practice ``val_end``) is where labels stop being available: past
    it every event is committed, attacks included. Without a horizon the test counters use
    test labels (an oracle).
    """
    src = np.asarray(src); dst = np.asarray(dst); y = np.asarray(y)
    n = len(src)
    horizon = n if label_horizon is None else int(label_horizon)
    feats = np.zeros((n, 3), dtype=np.float64)
    pair: dict = {}
    srcc: dict = {}
    for i in range(n):
        s = int(src[i]); d = int(dst[i])
        pc = pair.get((s, d), 0); sc = srcc.get(s, 0)
        feats[i, 0] = np.log1p(pc)
        feats[i, 1] = np.log1p(sc)
        feats[i, 2] = pc / (sc + 1.0)
        if i >= horizon or y[i] == 0:  # benign-gated up to the label horizon, then commit-all
            pair[(s, d)] = pc + 1
            srcc[s] = sc + 1
    return feats


def causal_src_seen(src, y, *, label_horizon: int | None = None, pred=None) -> np.ndarray:
    """Boolean (N,): has this src had >=1 benign event strictly before? (cold-start split).

    Past ``label_horizon`` (in practice ``val_end``) the benign gate uses ``pred`` (the
    model's decision, 1 = flagged) instead of ``y``; ``pred=None`` counts every such event as
    benign. Without a horizon the partition uses test labels (an oracle).
    """
    src = np.asarray(src); y = np.asarray(y)
    pred = None if pred is None else np.asarray(pred)
    horizon = len(src) if label_horizon is None else int(label_horizon)
    seen: set = set()
    out = np.zeros(len(src), dtype=bool)
    for i in range(len(src)):
        s = int(src[i])
        out[i] = s in seen
        if i < horizon:
            is_benign = y[i] == 0
        else:
            is_benign = True if pred is None else pred[i] == 0
        if is_benign:
            seen.add(s)
    return out


def causal_precursor_factor(src, t, msg, half_life: float, max_boost: float) -> np.ndarray:
    """Per-event multiplicative score factor (N,) from the kill-chain precursor.

    Causal mirror of :func:`graphagate.serve_tgn.precursor_shift` with the same half-life:
    ``1 + max_boost * 0.5**(Δt/half_life)`` after a Snort alert (``msg[:, 1] > 0.5``) on the
    same src, else ``1.0``. Multiplicative because baseline scores are not logits; an event
    never boosts itself.
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


def binary_metrics(scores, labels, threshold: float) -> tuple[float, float]:
    """Precision and recall of ``scores >= threshold`` against binary ``labels``."""
    preds = (np.asarray(scores) >= threshold).astype(int)
    labels = np.asarray(labels).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall
