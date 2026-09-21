# Baseline: Isolation Forest

A classic, **non-relational** anomaly detector. Every ZTA access event
is described by a 45-dim static vector: edge features `msg` (10) ⊕
static features of the device node (16) ⊕ static features of the resource node (16) ⊕
causal benign-gated history counters (3; the same *family* of statistics the TGN
maintains online, consumed here as a flat tabular vector by the device actor). No
memory, no temporal neighbourhood, and **strictly fewer signals than the TGN**: device
actor only — no user / source / config identity and no binding counters.

The counters are gated by ground-truth benign labels up to `val_end` and commit every
event after it (`label_horizon=val_end`, required by `_build_features`); the test labels
never reach the features (`tests/test_isolation_forest_baseline.py`). The TGN's own gate
past that point is `not signal_dirty`; the two differ only on signal-dirty events
(see `docs/external_dataset_optc.md`, "Stato al 2026-09-19").

Protocol identical to `graphagate.train_tgn` (same `TGNConfig`, same seed,
same chronological split 70/10/20, same precursor prior). The `IsolationForest`
(sklearn) is trained **on the benign train events only**; the 10 hyperparameter
sets sampled by `ParameterSampler` are selected
**by the AUC on the validation segment** (standard compromise of the one-class
setting: no labels enter the fit). The anomaly score is
`-score_samples(X)` (higher = more anomalous, consistent with the TGN's `1 - P(benign)`).
The threshold is calibrated on the benign validation scores at `target_fpr`
(99th percentile). Reported on test: aggregate AUC/AP, precision/recall at the
threshold and the per-type breakdown (policy / contextual / lateral / cred-theft / exfil /
benign-denied). The gap to the TGN measures how much the detection depends on the
graph structure and on the interaction history.

## Execution

Because PyTorch and baseline dependencies are not installed on the host, run the baseline inside the project's Docker container:

```bash
docker run --rm --gpus all -v "$PWD:/work" -w /work --entrypoint python graphagate \
  /work/tests/baselines/isolation_forest/isolation_forest_baseline.py
```

Alternatively, run via Docker Compose:

```bash
docker compose --profile baseline-iforest up
```
