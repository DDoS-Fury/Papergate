# Baseline: Vanilla 2-Node TGN (User → Resource)

## Why this baseline exists

It isolates the effect of the **5-node ZTA decomposition**
(Source → Config → Device → User → Resource) with respect to a **standard
2-node** Temporal Graph Network (User → Resource), as in the classic dynamic
graph link prediction literature (Rossi et al., Euler, Argus).

## Mechanism

- It keeps exactly the same temporal machinery as the main model:
  recurrent memory (TGNMemory GRU), time encoding, temporal neighbour loader,
  link predictor and structural head.
- The access is modelled only as a direct temporal edge between
  `user` and `dst` (resource), omitting the intermediate network entities,
  the TLS/JA3 configuration and the hardware device.
- Trained with the same negative-sampling curriculum as the TGN
  (InfoNCE over K random negatives + positive anchor BCE + contextual BCE).
- Evaluated with an event-by-event temporal replay and a threshold calibrated
  at the 99th percentile (1% FPR) of the benign validation segment.

## Expected and observed result

- It is **competitive** on the rest of the panel: in Panel A
  (`tasks/runs/panelA.json`) it obtains aggregate AUC 0.854 vs 0.853 for the
  5-node TGN and aggregate recall 0.625 vs 0.550 — the recurrent memory alone
  captures most of the tabular and habituality signal.
- It **loses on lateral movement**: lateral AUC 0.659 vs 0.721 for the 5-node
  TGN (margin +0.062). The multi-node decomposition adds context that the
  bipartite graph does not have.
- It **cannot represent Credential Theft**: when an attacker reuses the
  credentials of a known user from a client/configuration never associated
  with the principal, the User → Resource request is feature-identical to the
  legitimate traffic; only the Config → User binding (and the binding chain)
  exposes the structural anomaly. It is the class for which the 5-node
  decomposition is indispensable, not a bonus.

## Execution

Because PyTorch and baseline dependencies are not installed on the host, run the baseline inside the project's Docker container:

```bash
docker run --rm --gpus all -v "$PWD:/work" -w /work --entrypoint python graphagate \
  /work/tests/baselines/tgn_2node/tgn_2node_baseline.py
```

Alternatively, run via Docker Compose:

```bash
docker compose --profile baseline-tgn-2node up
```
