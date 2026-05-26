# Graphagate

GNN model training and inference microservice (and standalone), specialized in
**unsupervised anomaly detection** for ZTA intrusion detection/prevention systems.

## Overview (v1)

Graphagate implements the AI core described in `docs/Architettura AI per ZTA e OPA.pdf`:
a **DOMINANT-style Graph Autoencoder (GAE)** that learns the relational topology of a
normal Zero-Trust access graph and flags deviations via **reconstruction error**.

- IAM/ZTA logs are modelled as an attributed graph: nodes are entities
  (`user, device, ip_subnet, resource, session, role`), edges are access actions.
  Heterogeneity is encoded as a node-type one-hot inside a fixed-size feature vector
  (`FEATURE_DIM = 14`), so the encoder is a plain **dense GCN** and the model exports
  to ONNX with fully **static shapes**.
- The GAE is trained **only on benign traffic** (unsupervised); at inference the
  per-node reconstruction error, normalised to `[0, 1]`, is the "anomaly / legality"
  score consumed by OPA (example decision threshold `0.15`).
- Techniques from the document: shallow message passing (2 layers), **DropEdge**
  regularisation, **static bucketing + padding/masking** for low-latency ONNX/TensorRT.

This first version covers the **model** end-to-end in Python (data → train → score →
ONNX). The Go inference layer (`onnxruntime_go`), OPA integration, in-memory graph DB
(Memgraph/FalkorDB), online RTEC and INT8/TensorRT are **out of scope** here but the
code is structured to enable them (static shapes, masking, ego-network/bucketing).

## Setup

System Python is 3.14 (no torch wheels yet); use Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```bash
# 1. Inspect synthetic data
python -m graphagate.data.synthetic --stats --anomalies

# 2. Train on benign graph -> public/checkpoint.pt + public/norm_stats.json
python -m graphagate.train

# 3. Evaluate (inject anomalies, report ROC/PR-AUC) / score benign graph
python -m graphagate.score --eval
python -m graphagate.score

# 4. Export static-bucket ONNX models (+ onnxruntime parity check)
python -m graphagate.export_onnx        # -> public/model_{S,M,L}.onnx
```

### Docker (CPU)

```bash
docker build -f docker/Dockerfile -t graphagate .
docker run --rm -v "$PWD/public:/app/public" graphagate                      # train
docker run --rm -v "$PWD/public:/app/public" graphagate graphagate.score --eval
docker run --rm -v "$PWD/public:/app/public" graphagate graphagate.export_onnx
```

## Project layout

```
src/config.py            # hyper-parameters, static buckets (S/M/L), paths
src/data/schema.py       # node/edge types, fixed-size feature layout (shared contract)
src/data/synthetic.py    # benign IAM graph generator + anomaly injection + to_dense()
src/model/gae.py         # DenseGCN encoder + structure/attribute decoders (DOMINANT)
src/model/losses.py      # reconstruction loss + per-node anomaly score
src/train.py             # unsupervised training on benign graphs
src/score.py             # inference, [0,1] calibration, ROC/PR-AUC evaluation
src/export_onnx.py       # ONNX export per bucket (dynamo=True) + parity check
docker/Dockerfile        # CPU image for train / score / export
public/                  # artifacts: checkpoint.pt, model_{S,M,L}.onnx, norm_stats.json
```

## Results

On synthetic benign graphs with injected anomalies (lateral movement, privilege
escalation, foreign IP/device, off-hours/volume/privilege attribute outliers), for
unsupervised node-level anomaly detection (averaged over **8 unseen anomaly seeds**,
not used in training):

| Version | ROC-AUC | PR-AUC | Notes |
|---------|---------|--------|-------|
| v1      | ≈ 0.77  | ≈ 0.12 | single benign graph, uniform reconstruction MSE, raw features |
| **v2**  | **0.835 ± 0.065** | **0.658 ± 0.093** | + pos-weighted structure loss, inductive multi-graph training, z-scored features |

(PR-AUC random baseline ≈ 0.05.) **v2 levers**: (1) DOMINANT positive-class
weighting on the structure loss; (2) inductive training over a pool of benign
graphs; (3) z-score standardisation of numeric features (stats in
`norm_stats.json`); (4) a decision threshold calibrated to a target benign
false-positive rate.

**Capacity / data sweep (negative result).** Increasing encoder depth (2 → 3
message-passing layers) slightly *hurts* ROC-AUC (0.835 → 0.824, GNN
over-smoothing); enlarging the benign training pool (8 → 32 → 64 graphs) leaves
both metrics flat (≈ 0.835 / 0.658). 8 graphs (~3 600 benign nodes) already
saturate the synthetic benign distribution, so the bottleneck is neither model
capacity nor sample count. The default stays **2 layers / 8 graphs**.

**Threshold.** The calibrated threshold targets a 5 % benign FPR and measures
≈ 4.7 % on clean benign graphs (8 seeds). On graphs *with* injected anomalies the
benign FPR rises (≈ 20 %) because anomalies inflate the reconstruction error of
nearby benign nodes — expected contamination, which keeps operating-point
precision low (≈ 0.16–0.20) but is often useful in an IDS (attack blast radius).
AUC remains the threshold-independent headline metric; the operating point should
still be tuned per deployment.

> **Deployment note.** Feature standardisation is preprocessing *outside* the model.
> At ONNX/Go inference the `feat_mean` / `feat_std` stored in `norm_stats.json` must
> be applied to inputs before the model is called.
