# Graphagate

GNN model training and serving microservice (and standalone), specialized in
**unsupervised anomaly detection** for ZTA intrusion detection/prevention systems.

## Overview (Temporal Graph Network)

Graphagate analyzes continuous, real-time ZTA access streams using a
**Temporal Graph Network (TGN)**. The model maintains a historical "memory" of
node behavior (IPs and Users) and evaluates every new interaction sequentially.

- **Unsupervised Anomaly Detection**: the TGN is trained purely on benign streaming
  interactions via negative sampling (predicting structural and contextual edge
  likelihood). The anomaly score is `1 - P(benign)`.
- **Zero Trust Edge Features**: contextual data such as `JA3_trust` and `Snort`
  alerts are injected directly as edge features, allowing the network to penalize
  anomalous requests instantly.
- **Real-time serving**: `src/serve_tgn.py` exposes `load_model` + `score_event`.
  Scoring is event-by-event; memory is updated **only for events classified benign**
  (anti-poisoning), and a `NodeRegistry` admits previously unseen entities at runtime
  (dynamic, unbounded node space). Training persists `public/tgn_checkpoint.pt`
  (weights + memory + raw-message store) and `public/tgn_stats.json` (calibrated
  threshold + registry). Verify with `python -m graphagate.verify_tgn`.

> **Known limitation:** the embedding layer message-passes over the *target edge itself*
> rather than over historical temporal neighbours (a canonical TGN uses a neighbour loader),
> and static node features are not yet consumed. This makes detection lean on contextual edge
> features; robust *structural* anomaly detection (e.g. lateral movement) needs the
> neighbour-loader step.

## Usage (Docker)

Tutte le fasi girano via Docker Compose su GPU (CUDA 13, RTX Blackwell). Dettagli in
[`docs/docker.md`](docs/docker.md).

```bash
# Training del modello streaming temporale (TGN)
docker compose --profile training-tgn up

# Verifica della correttezza del serving streaming (richiede gli artifact in public/)
docker compose --profile verify-tgn up
```

In alternativa, con comandi Docker diretti:

```bash
docker build -f docker/Dockerfile -t graphagate .
docker run --rm --gpus all -v "$PWD/public:/app/public" graphagate                       # train_tgn
docker run --rm --gpus all -v "$PWD/public:/app/public" graphagate graphagate.verify_tgn
```

## Project layout

```
src/config.py                # TGN hyper-parameters and artifact paths
src/data/stream_synthetic.py # streaming mock data generator (structural/contextual anomalies)
src/model/tgn.py             # Temporal Graph Network architecture (TGNMemory + LinkPredictor)
src/model/registry.py        # dynamic NodeRegistry: external entity keys -> memory slots
src/train_tgn.py             # self-supervised training + threshold calibration + eval
src/serve_tgn.py             # real-time serving / persistence (load_model, score_event)
src/verify_tgn.py            # serving-path verification harness
docker/Dockerfile            # GPU image for train_tgn / verify_tgn
public/                      # artifacts: tgn_checkpoint.pt, tgn_stats.json
```

## Integrazione

L'integrazione con l'orchestrator ZTA / Policy Decision Point (OPA) è descritta in
[`docs/orchestrator_integration.md`](docs/orchestrator_integration.md).
