# Baseline: GNN non temporale (ablation del TGN)

GNN **statico** che isola il contributo della componente *temporale* del TGN.
Costruisce UN grafo non orientato aggregando gli archi benigni del solo segmento
di train (`y==0`), feature dei nodi = matrice statica `node_features [N,16]`.
Un encoder 2-layer GraphSAGE (hidden=64, ReLU + dropout 0.1) produce embedding
`z`; un link-predictor MLP su `[z_src ‖ z_dst ‖ msg]` dà un logit di benignità.
Training self-supervised solo-benigno con lo **stesso curriculum del TGN** (per
un'ablation equa): positivo = arco reale; negativo strutturale = `(src, nodo_casuale)`
in `[num_users, total_nodes)`; **hard-negative ×10** = `(src, risorsa NON abituale)`,
con l'abitualità IP→risorsa letta dal grafo statico (esiste un arco benigno di train);
negativo contestuale = stessa dst, 20% dei bit del `msg` invertiti. `BCEWithLogitsLoss`,
`AdamW` lr=1e-3, 15 epoche. Score di anomalia = `1 - sigmoid(link_pred)`
(più alto = più anomalo). Soglia calibrata sul benigno di validazione al
`target_fpr` (1%); metriche aggregate + breakdown per tipo come `train_tgn`.

L'unica differenza col TGN è l'assenza di **memoria ricorrente** e **vicinato
temporale** (più testa a coseno e identità hashata): si isola così il contributo
della sola temporalità. Esito atteso e osservato: pareggia il TGN su *policy* e
*contextual*, ma resta molto sotto sul *lateral movement* (Recall ~9% vs ~50%),
perché un grafo statico aggregato appiattisce la cronologia che lo rivela.

Avvio riproducibile anche via profilo Compose dedicato: `docker compose --profile baseline-gnn up`.

## Esecuzione (torch non è sull'host: usare l'immagine Docker del progetto)

```bash
# dalla root del repo, dopo aver buildato l'immagine `graphagate`
# (docker compose --profile training-tgn build, oppure docker build -f docker/Dockerfile -t graphagate .)
docker run --rm --gpus all \
  -v "$PWD/tests:/app/tests" \
  --entrypoint python \
  graphagate /app/tests/baselines/simple_gnn/simple_gnn_baseline.py
```
