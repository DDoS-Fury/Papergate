# TASKS — Graphagate v1 (GAE per anomaly detection ZTA)

Tracciamento esecuzione. Legenda: ⬜ da fare · 🔄 in corso · ✅ fatto

| # | Task | Stato | Note |
|---|------|-------|------|
| 1 | Setup venv (python3.12) + pyproject + requirements + install | ✅ | torch 2.12, pyg 2.7, onnx 1.21, onnxruntime 1.26, onnxscript 0.7, numpy 2.4, sklearn 1.8 |
| 2 | Contratto condiviso: `src/config.py`, `src/data/schema.py` | ✅ | FEATURE_DIM=14 (6 tipi nodo one-hot + 8 numeriche), buckets S/M/L=64/128/256 |
| 3 | Data layer: `src/data/synthetic.py` | ✅ | Subagente A — gen benigno coerente + iniezione anomalie (strutturali/attributo) + `to_dense` |
| 4 | Modello: `src/model/gae.py` + `src/model/losses.py` | ✅ | Subagente B — DOMINANT, DenseGCN, DropEdge, ONNX-friendly |
| 5 | Training: `src/train.py` | ✅ | Adam + early stopping; salva `public/checkpoint.pt` + `public/norm_stats.json` |
| 6 | Scoring/eval: `src/score.py` | ✅ | score [0,1]; `--eval` ROC-AUC≈0.77 (5 seed), recall 0.91@0.15 |
| 7 | Export ONNX: `src/export_onnx.py` | ✅ | bucket S/M/L con `dynamo=True`; parità vs onnxruntime ~2e-7 |
| 8 | Dockerfile + verifica end-to-end | ✅ | `docker/Dockerfile` CPU (8.57 GB); pipeline train→score→export verificata nel container (ONNX parità ~1e-7) |

## v2 — miglioramento precisione / PR-AUC

Obiettivo: ridurre i falsi positivi e alzare PR-AUC mantenendo invariati architettura (DenseGCN 2 layer) e contratto `forward`/export ONNX.

| # | Leva | File | Stato | Note |
|---|------|------|-------|------|
| v2-0 | Contratto: nuovi campi `ModelConfig` + firma loss `struct_pos_weight` | `src/config.py` | ✅ | `struct_pos_weight`, `standardize_features`, `num_train_graphs`, `num_val_graphs`, `target_fpr` |
| v2-1 | Pos-weight loss strutturale (DOMINANT θ) | `src/model/losses.py` | ✅ | Subagente A — peso ×θ sugli archi reali; `None`=auto (#neg/#pos clamp [1,50]) |
| v2-2 | Training induttivo su pool di K grafi benigni | `src/train.py` | ✅ | Subagente B — 8 grafi train + 2 val held-out (seed disgiunti) |
| v2-3 | Standardizzazione feature numeriche (z-score) | `src/features.py`, `train.py`, `score.py` | ✅ | Subagente B — `feat_mean`/`feat_std` benigni in `norm_stats`, riapplicati in inferenza |
| v2-4 | Calibrazione soglia su FPR target | `src/train.py`, `src/score.py` | ✅ | Subagente B — `threshold` da quantile benigno (FPR 5%); in `norm_stats` |
| v2-5 | Ablation + aggiornamento README/TASKS | — | ✅ | 8 seed unseen: ROC-AUC 0.835±0.065, PR-AUC 0.658±0.093 (era 0.77/0.12); ONNX parità ~6e-7 invariata |
| v2-6 | Sweep profondità (2 vs 3 layer) + dimensione pool (8/32/64 grafi) | — | ✅ | Risultato negativo: 3 layer peggiora (oversmoothing), più grafi invariato. Config invariata 2 layer/8 grafi. |

> Architettura: valutata e mantenuta invariata (no oversmoothing/overfit, export ONNX-friendly). Confermato da v2-6: 3 layer non aiuta. Eventuali modifiche architetturali → v3, gated su nuove idee (non capacità/numerosità).

### Risultati v2 (8 seed unseen, riverificati 2026-05-26)
- **ROC-AUC 0.835±0.065** (v1: 0.77) · **PR-AUC 0.658±0.093** (v1: 0.12, baseline 0.05) → salto principale dalla pos-weight + training induttivo. (Nota: i precedenti 0.87/0.69 erano seed favorevoli; questi sono la media su 8 seed di anomalie mai visti.)
- **Soglia calibrata** = 0.493 (target FPR 5%): su grafi benigni *puliti* FPR reale **4.7%±2.2%** (≈ target). Su grafi con anomalie iniettate l'FPR sale a ~20% per *contaminazione* (le anomalie alzano lo score dei vicini benigni) → precision al punto operativo bassa (~0.16–0.20); atteso, e in un IDS spesso utile (raggio d'azione).
- **Sweep capacità/dati (v2-6, negativo)**: 2→3 layer ROC 0.835→0.824 (oversmoothing); pool 8→32→64 grafi metriche piatte (~0.835/0.658). 8 grafi (~3.600 nodi) saturano già la distribuzione benigna → il collo di bottiglia non è capacità né numerosità.
- `norm_stats.json` esteso: `+ threshold, target_fpr, standardize_features, feat_mean[8], feat_std[8]`.
- **Nota deploy**: la standardizzazione è preprocessing *fuori* dal modello → al deploy ONNX/Go le stesse `feat_mean`/`feat_std` vanno applicate agli input prima dell'inferenza.

## Risultati v1
- **ROC-AUC ≈ 0.77** (media 5 seed) sull'individuazione non supervisionata di nodi anomali; PR-AUC ≈ 0.12 (baseline 0.05).
- Pipeline completa funzionante: `train` → `score --eval` → `export_onnx` (3 ONNX statici, parità numerica).
- Soglia di decisione 0.15 (esempio dal PDF) → recall alta, precisione bassa: la soglia va calibrata in deployment (AUC è la metrica headline per la v1).

## Contratto di interfaccia (condiviso A↔B)
- `x` shape `[N, FEATURE_DIM=14]` float32 (via `schema.node_feature`).
- `to_dense(data, size)` → `(x_pad [size,F], adj [size,size] 0/1 simmetrica, mask [size])`.
- `DominantGAE.forward(x, adj)` → `(a_hat [N,N]∈[0,1], x_hat [N,F], z [N,embed])`; self-loop+norm interni; DropEdge solo in training.
- `reconstruction_loss(adj,a_hat,x,x_hat,alpha,mask)` scalare; `node_anomaly_score(...)` → `[N]`.

## Note ambiente
- venv: `./.venv/bin/python` (Python 3.12). Moduli via `./.venv/bin/python -m graphagate.<modulo>`.
- Training/inferenza su CPU (arm64). I bucket statici sono per inferenza/ONNX (ego-network); il training usa il grafo intero.

## Fuori scope v1 (fasi successive)
- Inferenza Go (`onnxruntime_go`), integrazione OPA (Rego + custom built-in), Memgraph/FalkorDB, RTEC online, quantizzazione INT8/TensorRT.
- Architettura del codice già predisposta: shape statiche + masking + estrazione ego-network/bucketing.
