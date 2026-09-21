"""Stateful lookup-rule baseline (no learning) under the TGN evaluation protocol.

Why this baseline exists:
  ``graphagate.data.lookup_rules`` holds the rules a SIEM correlation engine would run:
  one dict lookup per event ("has this config ever been seen with this device / user?").
  On the closed-world generator (v4) they beat the TGN on lateral movement and credential
  theft, which is what a reviewer would run first. The paper therefore reports them next
  to the TGN: the learned model has to earn its place against this baseline, on the
  classes the contribution rests on.

Protocol (the same numbers the TGN and the Isolation Forest report):
  1. Same stream (same ``TGNConfig`` + seed), same chronological split, same test window
     (``val_end = int(n*train_frac) + int(n*val_frac)``).
  2. Score = ``stateful``: the number of history-dependent binding flags that fire
     (``cfg|dev_new``, ``cfg|usr_new``, ``dev|usr_new``, ``src|usr_new``, ``role_changed``).
     Integer 0..5.
  3. Per class (lateral, cred-theft): benign-vs-class AUC / AP inside the test window.
  4. Operating point: the smallest integer threshold ``k >= 1`` whose *benign validation*
     FPR (``score >= k``) is <= ``target_fpr``; recall / FPR are then measured on the test
     window. A quantile threshold is degenerate on integer scores, hence this definition.

The rules address only lateral movement and credential theft. Policy / contextual /
exfil belong to the OPA and the sensor layer, so no aggregate metric is reported for
this baseline (it would be dominated by classes the rules do not attempt).

``gate`` is the rule state's commit gate, see ``lookup_rules``: ``proto-self`` (ground-truth
benign commits up to a label horizon, then only events no binding rule fired on) or ``all``
(no label at all).

Label horizon of ``proto-self`` — the protocol choice that matters:
  * ``lookup_rules_baseline`` (primary): labels reach the rule state only through the
    *training* window; validation and test are self-gated. This is what the TGN gets — its
    validation / test replays commit on ``not signal_dirty``, never on labels — and what a
    deployment can have (an initial learning period, then nothing).
  * ``lookup_rules_val_baseline`` (sensitivity): the state also receives ground-truth
    labels through the *validation* window, i.e. the rule audit's protocol. It is an oracle
    over validation, hence an upper bound for the rules: measured on v5 (seeds 2000-2001,
    200k) it inflates lateral AUC by 0.04-0.08 and makes the validation-calibrated
    threshold land at ~10% benign FPR on the test window (target 1%), because the state
    regime changes at the validation/test boundary. A TGN that beats it beats the rules
    under the most favourable reading.
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from graphagate.config import TGNConfig
from graphagate.data.lookup_rules import lookup_flags
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg

# (type id, name) — names match ``train_tgn`` / the other baselines' ``per_type`` keys.
RULE_CLASSES = ((3, "lateral"), (4, "cred-theft"))


def _operating_threshold(val_benign_scores: np.ndarray, target_fpr: float) -> int:
    """Smallest integer ``k >= 1`` with ``mean(score >= k) <= target_fpr`` on benign validation."""
    k = 1
    while (val_benign_scores >= k).mean() > target_fpr:
        k += 1
    return k


def lookup_rules_baseline(cfg: TGNConfig = TGNConfig(), stream=None, gate: str = "proto-self",
                          labels_through: str = "train"):
    """Score the stateful rules under the TGN protocol; returns the baselines' metrics dict.

    ``stream`` is an optional pre-built :class:`SyntheticStream` (the data-budget curve
    passes a truncated tail); ``None`` generates it from ``cfg``. ``labels_through`` is the
    ``proto-self`` label horizon, ``"train"`` (primary) or ``"val"`` (sensitivity), see the
    module docstring. The returned dict has only ``per_type`` (no aggregate); each entry is
    ``{auc, ap, recall, fpr, threshold, n}`` with ``recall`` / ``fpr`` at the operating
    point defined above.
    """
    if labels_through not in ("train", "val"):
        raise ValueError(f"labels_through must be 'train' or 'val', got {labels_through!r}")
    if stream is None:
        print("Generating synthetic streaming data (same params as TGN)...")
        stream = generate_streaming_data(**stream_kwargs_from_cfg(cfg))

    y = stream.y.numpy()
    types = stream.types.numpy()
    n = len(y)
    train_end = int(n * cfg.train_frac)
    val_end = train_end + int(n * cfg.val_frac)

    horizon = train_end if labels_through == "train" else val_end
    score = np.asarray(lookup_flags(stream, gate, horizon)["stateful"], dtype=float)

    idx = np.arange(n)
    val_benign = (idx >= train_end) & (idx < val_end) & (types == 0)
    if not val_benign.any():
        raise RuntimeError("No benign events in the validation slice for calibration.")
    k = _operating_threshold(score[val_benign], cfg.target_fpr)

    test = idx >= val_end
    test_benign = test & (types == 0)
    fpr = float((score[test_benign] >= k).mean())

    per_type = {}
    print(f"\n--- LOOKUP RULES (gate={gate}, labels through {labels_through}) | threshold k={k} | "
          f"benign test FPR={fpr:.4f} ---")
    for type_id, name in RULE_CLASSES:
        cls = test & (types == type_id)
        if not cls.any():
            continue
        sel = test_benign | cls
        labels = (types[sel] == type_id).astype(int)
        t_auc = roc_auc_score(labels, score[sel])
        t_ap = average_precision_score(labels, score[sel])
        t_recall = float((score[cls] >= k).mean())
        per_type[name] = {"auc": float(t_auc), "ap": float(t_ap), "recall": t_recall,
                          "fpr": fpr, "threshold": int(k), "n": int(cls.sum())}
        print(f"  {name:10s} | n={int(cls.sum()):4d} | AUC: {t_auc:.4f} | AP: {t_ap:.4f} | "
              f"Recall@k: {t_recall:.4f}")
    return {"per_type": per_type}


def lookup_rules_val_baseline(cfg: TGNConfig = TGNConfig(), stream=None):
    """Sensitivity: ``proto-self`` with ground-truth labels through validation (the audit's protocol)."""
    return lookup_rules_baseline(cfg, stream, gate="proto-self", labels_through="val")


def lookup_rules_all_baseline(cfg: TGNConfig = TGNConfig(), stream=None):
    """Same rules with ``gate="all"``: the state commits every event, no label anywhere."""
    return lookup_rules_baseline(cfg, stream, gate="all")


def main():
    lookup_rules_baseline()


if __name__ == "__main__":
    main()
