# Baseline: Supervised XGBoost

This directory contains the XGBoost **supervised** baseline for Graphagate.

## Execution

Because PyTorch and baseline dependencies are not installed on the host, run the baseline inside the project's Docker container:

```bash
docker run --rm --gpus all -v "$PWD:/work" -w /work --entrypoint python graphagate \
  /work/tests/baselines/xgboost/xgboost_baseline.py
```

Alternatively, run via Docker Compose:

```bash
docker compose --profile baseline-xgboost up
```

## Details

The baseline treats every event as an independent static feature vector — the
same 45-dim vector the Isolation Forest sees (10-dim edge msg, 16-dim static
attributes of both endpoints, 3-dim causal interaction-history counts) —
ignoring relational and temporal graph structure. It trains a supervised
`XGBClassifier` (`tree_method="hist"`, `n_jobs=-1`) on **both benign and
anomalous** instances of the training split. Hyperparameters are tuned with
`RandomizedSearchCV` (10 iterations, 3-fold CV, scoring `roc_auc`, search
itself single-process `n_jobs=1` to avoid CPU oversubscription) and the best
estimator is evaluated on the test stream with the same threshold protocol as
the other baselines (99th percentile of benign validation scores, 1% target
FPR).

Metrics: aggregate AUC/AP, precision/recall at threshold, and the per-anomaly-
type breakdown (policy / contextual / lateral / cred-theft / exfil /
benign-denied), computed identically to `graphagate.train_tgn`.

Because it sees ground-truth labels during training, it is an **empirical
upper-bound reference**, not a like-for-like baseline of the one-class TGN:
it measures how much of the remaining gap is due to the label-free training
regime rather than to the feature space.
