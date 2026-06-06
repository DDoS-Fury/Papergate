# Baselines di confronto

Queste baseline servono a quantificare *quanto* il Temporal Graph Network (TGN)
aggiunga rispetto a metodi più semplici, andando oltre la sola `rule-based baseline`
già presente in `graphagate.train_tgn` (che per costruzione è cieca alle anomalie
`policy` e `lateral`, perché condividono le edge feature benigne).

## Protocollo comune (per confrontabilità 1:1 col TGN)

Tutte le baseline **devono**:

1. Generare i dati con `graphagate.data.stream_synthetic.generate_streaming_data`
   usando gli stessi iperparametri di `graphagate.config.TGNConfig`
   (`num_users=50, num_ips=100, num_resources=20, num_events=50000, seed=42`).
2. Usare lo **stesso split cronologico**: train 70% / val 10% / test 20%
   (`train_frac=0.7`, `val_frac=0.1`).
3. Addestrare **solo su traffico benigno** del segmento di train (`y == 0`):
   è anomaly detection senza etichette di anomalia, come il TGN.
4. Riportare sul segmento di **test** le stesse metriche del TGN:
   - `roc_auc_score` e `average_precision_score` aggregate (benigno vs tutte le anomalie);
   - breakdown **per tipo** (`types`: 1=policy, 2=contextual, 3=lateral) calcolato
     benigno-vs-quel-tipo, con AUC / AP / Recall@threshold;
   - la soglia si calibra sul segmento di **validazione benigno** al `target_fpr`
     di `TGNConfig` (1%, ovvero 99° percentile degli score benigni), identico al TGN.

Lo score di anomalia deve essere "più alto = più anomalo", coerente con
`graphagate.serve_tgn.infer_score` (che restituisce `1 - P(benign)`).

## Formato dei dati

`generate_streaming_data(...)` restituisce (tensori `torch`, già ordinati nel tempo):

| nome | shape | significato |
|------|-------|-------------|
| `src` | `[N]` | indice nodo sorgente (IP) |
| `dst` | `[N]` | indice nodo destinazione (risorsa) |
| `t`   | `[N]` | timestamp (interi crescenti) |
| `msg` | `[N,6]` | edge feature `[ja3, snort, s1, s2, s3, method]` |
| `y`   | `[N]` | label binaria (0=benigno, 1=anomalo) |
| `types` | `[N]` | 0=benigno, 1=policy, 2=contextual, 3=lateral |
| `node_features` | `[total_nodes,16]` | attributi statici (ruolo/clearance/tier) |
| `resource_uris` | `list[str]` | URI delle risorse |

Spazio indici nodi: `[0,num_users)` utenti, `[num_users,num_users+num_ips)` IP,
`[num_users+num_ips, total_nodes)` risorse.

## Esecuzione

torch non è installato sull'host: eseguire dentro l'immagine Docker del progetto.

```bash
# dalla root del repo
docker compose run --rm --no-deps train-tgn python -m graphagate.data.stream_synthetic  # esempio
# oppure montare i tests ed eseguire lo script della baseline:
docker run --rm --gpus all -v "$PWD:/work" -w /work graphagate \
  python tests/baselines/isolation_forest/isolation_forest_baseline.py
```

## Baseline implementate

- `isolation_forest/` — Isolation Forest (sklearn) su vettori statici per-evento
  (edge feature ⊕ feature statiche dei due endpoint). Detector di anomalie
  classico, non relazionale: misura quanto si ottiene **senza** struttura del grafo.
- `ocsvm/` — One-Class SVM (sklearn, kernel RBF) sugli **stessi** vettori statici
  per-evento dell'Isolation Forest (fit su subsample benigno per scalabilità). La
  controparte kernel del "pavimento" non relazionale.
- `simple_gnn/` — GNN **non temporale** (GraphSAGE) su grafo statico aggregato dal
  train benigno + link predictor MLP. Ablation **equa** del TGN: mantiene lo *stesso*
  curriculum **de-circolarizzato** (negativo strutturale a destinazione casuale +
  contestuale gaussiano, pesi uguali — niente più hard-negative ×10 basato
  sull'abitualità/autorizzazione, che era circolare), e rimuove **solo** la memoria
  ricorrente e il vicinato temporale. Isola così il contributo della sola componente
  *temporale* alla detection del lateral movement (lateral AUC 0.59 vs 0.71 del TGN).
