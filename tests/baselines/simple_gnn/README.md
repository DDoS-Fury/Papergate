# Baseline: Non-Temporal GNN (TGN Ablation)

A **static** GNN that isolates the contribution of the TGN's *temporal*
component. It builds a graph by aggregating the benign edges of the train
segment only (`y==0`), node features = the static `node_features [N,16]` matrix.
A GraphSAGE encoder produces the `z` embeddings; an MLP link predictor over
`[z_src ‖ z_dst ‖ msg ‖ hist]` gives a benignity logit.

Benign-only self-supervised training with the **same curriculum as the TGN**
(for a fair ablation): positive = real edge; **structural negative** =
`(src, random resource)` sampled in the stream's **real resource range**
(`stream.res_lo`/`stream.res_num` — not an arbitrary slot of the total node
space, which would count users/source/config); **contextual negative** =
same dst with `msg` corrupted by Gaussian noise; equal weights. No
habituality/authorization-based hard-negative ×10: that construction was
circular (it used the very notion the ablation had to measure) and was
removed. `BCEWithLogitsLoss`. Anomaly score = `1 - sigmoid(link_pred)`
(higher = more anomalous). Threshold calibrated on the benign validation
slice at `target_fpr` (1%); aggregate metrics + per-type breakdown as in
`train_tgn`.

The only difference with the TGN is the absence of **recurrent memory** and
**temporal neighbourhood**: this isolates the contribution of the temporal
component alone. On Panel A (`tasks/runs/panelA.json`, to be regenerated — see
the baselines README) the static GNN stalls at lateral AUC 0.602, practically
on the single-feature floor (0.603), against 0.721 for the TGN: the aggregated
graph flattens the chronology that reveals the lateral.

## Execution

Because PyTorch and baseline dependencies are not installed on the host, run the baseline inside the project's Docker container:

```bash
docker run --rm --gpus all \
  -v "$PWD:/work" -w /work \
  --entrypoint python \
  graphagate /work/tests/baselines/simple_gnn/simple_gnn_baseline.py
```

Alternatively, run via Docker Compose:

```bash
docker compose --profile baseline-gnn up
```
