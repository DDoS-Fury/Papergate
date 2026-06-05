# Baseline: Isolation Forest

Detector di anomalie classico e **non relazionale**. Ogni evento di accesso ZTA
viene descritto da un vettore statico di 38 dim: edge feature `msg` (6) ⊕ feature
statiche del nodo sorgente IP (16) ⊕ feature statiche del nodo risorsa (16). Nessuna
memoria, nessun vicinato temporale: è il massimo che un detector può vedere di un
singolo evento isolato.

Protocollo identico a `graphagate.train_tgn` (stesso `TGNConfig` + seed, stesso split
cronologico 70/10/20). L'`IsolationForest` (sklearn, `n_estimators=200`,
`contamination='auto'`) è addestrato **solo sugli eventi benigni del train**. Lo score
di anomalia è `-score_samples(X)` (più alto = più anomalo, coerente con `1 - P(benign)`
del TGN). La soglia si calibra sugli score benigni di validazione al `target_fpr` (99°
percentile). Si riportano su test: AUC/AP aggregate, precision/recall alla soglia e il
breakdown per tipo (policy / contextual / lateral). Il gap con il TGN misura quanto la
detection dipenda dalla struttura del grafo e dalla storia delle interazioni.

## Esecuzione (torch non è installato sull'host: usare l'immagine Docker)

```bash
# dalla root del repo
docker run --rm -v "$PWD:/work" -w /work graphagate \
  python tests/baselines/isolation_forest/isolation_forest_baseline.py
```
