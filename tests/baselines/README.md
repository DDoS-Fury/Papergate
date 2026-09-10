# Comparison baselines

These baselines quantify *how much* the Temporal Graph Network (TGN)
adds on top of simpler methods, going beyond the bare `rule-based baseline`
already present in `graphagate.train_tgn` (which by construction is blind to the
`policy` and `lateral` anomalies, because they share the benign edge features).

## Common protocol (for 1:1 comparability with the TGN)

All baselines:

1. Generate the stream with `generate_streaming_data(**stream_kwargs_from_cfg(cfg))`
   (`graphagate.config`): a **single** TGNConfig→generator mapping, shared by the TGN,
   the baselines, the live generator and the leakage audit. No driver passes the
   generator parameters by hand: if the TGN configuration changes, the
   baselines' stream changes too (same entity space, same counting statistics).
2. Use the **same chronological split**: train 70% / val 10% / test 20%
   (`train_frac=0.7`, `val_frac=0.1` of `TGNConfig`).
3. The one-class models (IF, OC-SVM) train **on the benign traffic only** of the
   train segment (`y == 0`), like the TGN. XGBoost is supervised (it sees the labels
   in training) and is included only as a reference upper-bound.
4. Report on the **test** segment the same metrics as the TGN:
   - `roc_auc_score` and `average_precision_score` aggregate (benign vs all anomalies);
   - breakdown **per type** (0=benign, 1=policy, 2=contextual, 3=lateral, 4=cred-theft,
     5=exfil, 6=benign-denied — same set as the `train_tgn` breakdown),
     benign-vs-that-type, with AUC / AP / Recall@threshold;
   - the threshold is calibrated on the **benign validation** segment at the `target_fpr`
     of `TGNConfig` (1%, 99th percentile of the benign scores), identical to the TGN.

The anomaly score is "higher = more anomalous", consistent with
`graphagate.serve_tgn.infer_score` (which returns `1 - P(benign)`).

## Data format

`generate_streaming_data(...)` returns a `StreamData` (torch tensors, already
time-sorted):

| field | shape | meaning |
|------|-------|-------------|
| `src` | `[N]` | source node index of the chain (v4: source IP → config → device → user) |
| `dst` | `[N]` | destination node index (resource) |
| `t`   | `[N]` | timestamp (increasing integers) |
| `msg` | `[N,10]` | dynamic v4 edge features |
| `y`   | `[N]` | binary label (0=benign, 1=anomalous) |
| `types` | `[N]` | 0=benign, 1=policy, 2=contextual, 3=lateral, 4=cred-theft, 5=exfil, 6=benign-denied |
| `node_features` | `[num_nodes,16]` | static attributes per node type |

Node indices are block-allocated per type (`num_users`, `num_ips`, `num_devices`,
`num_configs`...), defined in `stream_synthetic.py`; the `config` node collapses to
`conf:guest` for non-TLS clients according to `guest_device_fallback` (default True),
consistent with the deployable protocol.

## Status of the cited numbers

The Panel A values (Table III of the paper) come from `tasks/runs/panelA.json`
(2026-08-31) and predate the stream parity fix referred to in point 1:
the baseline rows must be **regenerated** (`docker compose --profile regen-report up`)
before the TGN-vs-baseline deltas are cited again. The TGN-2node row is already
at parity (it received all the parameters) and its values remain valid.

## Execution

Because PyTorch and baseline dependencies are not installed on the host, run baselines inside the project's Docker container.

### Direct Docker run

From the repository root (after building the `graphagate` image with `docker build -f docker/Dockerfile -t graphagate .`):

```bash
# Example: running the Isolation Forest baseline
docker run --rm --gpus all -v "$PWD:/work" -w /work --entrypoint python graphagate \
  /work/tests/baselines/isolation_forest/isolation_forest_baseline.py
```

### Via Docker Compose

Alternatively, run each baseline using its dedicated Compose profile:

```bash
docker compose --profile baseline-tgn-2node up
docker compose --profile baseline-iforest up
docker compose --profile baseline-ocsvm up
docker compose --profile baseline-gnn up
docker compose --profile baseline-xgboost up
```

## Implemented baselines

- `isolation_forest/` — Isolation Forest (sklearn) on per-event static vectors
  (edge features ⊕ static features of the two endpoints ⊕ causal history counters).
  A classic, non-relational anomaly detector: it measures what is obtained **without**
  graph structure. Hyperparameters selected on the validation AUC
  (standard compromise of the one-class setting: no labels in the fit).
- `ocsvm/` — One-Class SVM (sklearn, RBF kernel) on the **same** per-event static
  vectors as the Isolation Forest (fit on a benign subsample for scalability). The
  kernel counterpart of the non-relational "floor".
- `xgboost/` — **supervised** XGBoost (same static vectors + history
  counters). Tuning via `RandomizedSearchCV` (10 iterations, CV=3, scoring `roc_auc`)
  on the whole train segment with labels. Reference upper-bound, **not**
  a comparable baseline within the one-class paradigm.
- `simple_gnn/` — **non-temporal** GNN (GraphSAGE) on the static graph aggregated
  from the benign train + MLP link predictor. A **fair** ablation of the TGN: it keeps
  the *same* **de-circularised** curriculum (structural negative with a random
  destination in the stream's real resource range + Gaussian context, equal weights)
  and removes **only** the recurrent memory and the temporal neighbourhood. It
  isolates the contribution of the *temporal* component to the lateral movement
  detection.
- `tgn_2node/` — **2-node** TGN (User → Resource). It isolates the contribution of
  the **5-node ZTA decomposition** with respect to a classic literature TGN on the
  direct User-Resource access graph. It keeps the same recurrent memory,
  time encoding and negative sampling.
