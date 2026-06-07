# Esecuzione con Docker Compose

Per semplificare l'utilizzo del progetto Graphagate su GPU (ottimizzato per RTX Blackwell, CUDA 13), è stato predisposto un file `docker-compose.yml` con profili dedicati. 
Questa struttura permette di eseguire le diverse fasi del progetto isolando gli ambienti e senza dover ricordare complessi comandi Docker.

## Prerequisiti
- Docker e Docker Compose installati
- NVIDIA Container Toolkit installato e configurato (per l'utilizzo della GPU)
- Driver NVIDIA compatibili con CUDA 13 (es. driver per RTX 5090 / B200 Blackwell)

## Profili Disponibili

Il `docker-compose.yml` contiene sei profili: `training-tgn`, `verify-tgn`, `serve-tgn` e i tre profili di confronto `baseline-iforest`, `baseline-ocsvm` e `baseline-gnn`.

### 1. Training (Modello Temporale TGN)
Avvia il container per l'addestramento della rete dinamica basata su grafi temporali (Temporal Graph Network). Questo profilo genera dati in stream continuo con contesti Zero Trust (JA3, alert snort, sonde). Gli artifact risultanti vengono salvati nella cartella `public/`.

```bash
docker compose --profile training-tgn up
```
*(Esegue in background `python -m graphagate.train_tgn`)*

### 2. Verifica correttezza streaming (TGN)
Dopo il training TGN (che salva `public/tgn_checkpoint.pt` e `public/tgn_stats.json`),
questo profilo ricarica l'artifact e verifica le proprietà di serving real-time:
determinismo del reload, gate anti-poisoning della memoria (un evento anomalo non
aggiorna la baseline) e ammissione di entità mai viste (nodi dinamici).

```bash
docker compose --profile verify-tgn up
```
*(Esegue `python -m graphagate.verify_tgn`)*

### 3. Servizio di inferenza (TGN, long-running)
A differenza dei due profili batch, questo avvia un **servizio HTTP persistente**:
carica gli artifact da `public/` ed espone l'API REST/JSON (`graphagate.serve_api`)
sulla porta `8088`, pensata per essere consumata dall'orchestrator ZTA (vedi
[`orchestrator_integration.md`](orchestrator_integration.md)). Lo stato evoluto dagli
eventi approvati viene riscritto su `public/` via `POST /persist` e automaticamente
allo spegnimento del container.

> **Prerequisito**: gli artifact (`tgn_checkpoint.pt`, `tgn_stats.json`) devono già
> esistere in `public/`. Vanno prodotti una volta dal profilo `training-tgn` **prima**
> di avviare il servizio.

```bash
docker compose --profile serve-tgn up
```
*(Esegue `python -m graphagate.serve_api`; healthcheck su `GET /health`)*

Avvio standalone equivalente (stessa immagine, senza Compose):

```bash
docker run --rm --gpus all -p 8088:8088 \
  -v "$PWD/public:/app/public" graphagate graphagate.serve_api
```

Configurazione via variabili d'ambiente (tutte opzionali): `GRAPHAGATE_CHECKPOINT` e
`GRAPHAGATE_STATS` (path degli artifact), `GRAPHAGATE_HOST` (default `0.0.0.0`),
`GRAPHAGATE_PORT` (default `8088`).

> **Una sola replica.** Il modello è uno stato mutabile in RAM: il servizio gira con un
> singolo worker e **non** va scalato orizzontalmente.

Gli artifact TGN persistiti in `public/` sono:
- `tgn_checkpoint.pt` — pesi (inclusi identità di nodo e teste di scoring) + buffer di
  memoria + raw-message store + buffer del neighbour loader (per continuare esattamente
  lo stato temporale e la storia dei vicini tra riavvii);
- `tgn_stats.json` — soglia di decisione calibrata, `capacity` e mappatura
  `NodeRegistry` (entità esterne → slot di memoria).

### 4. Baseline di confronto (batch, one-shot)
Due detector più semplici, eseguibili in container dedicati per un confronto
riproducibile col TGN (stesso stream sintetico, split cronologico, soglia calibrata
all'1% di FPR e protocollo di valutazione). Dettagli e metriche attese in
[`../tests/baselines/README.md`](../tests/baselines/README.md).

- **Isolation Forest** (sklearn): detector non relazionale sulle sole feature
  statiche per-evento — il "pavimento" privo di informazione strutturale.

  ```bash
  docker compose --profile baseline-iforest up
  ```
  *(Esegue `python tests/baselines/isolation_forest/isolation_forest_baseline.py`)*

- **One-Class SVM** (sklearn, kernel RBF): controparte kernel dell'Isolation Forest,
  anch'essa non relazionale sulle sole feature statiche per-evento (fit su subsample
  benigno per scalabilità).

  ```bash
  docker compose --profile baseline-ocsvm up
  ```
  *(Esegue `python tests/baselines/ocsvm/ocsvm_baseline.py`)*

- **GNN non temporale** (GraphSAGE): ablation del TGN su grafo statico aggregato,
  con lo **stesso curriculum de-circolarizzato** del TGN (negativo strutturale a
  destinazione casuale + contestuale gaussiano, pesi uguali) ma senza memoria
  ricorrente né vicinato temporale — isola il contributo della sola componente
  temporale (lateral AUC 0.59 vs 0.71 del TGN completo).

  ```bash
  docker compose --profile baseline-gnn up
  ```
  *(Esegue `python tests/baselines/simple_gnn/simple_gnn_baseline.py`)*

> **Prerequisito**: i container delle baseline montano la cartella `./tests`
> dell'host (gli script non sono copiati nell'immagine) e sovrascrivono l'entrypoint
> `python -m`. Non producono artifact in `public/`: stampano le metriche a console.

### 5. Ablations e Test Unitari
Esegue gli esperimenti multi-seed di rimozione strutturale (ablation) per certificare matematicamente il contributo di specifiche componenti del modello (come history causale e precursor) isolandole dal resto. Questo profilo viene anche utilizzato per lanciare unit-test specifici sovrascrivendo l'entrypoint.

```bash
docker compose --profile ablations up
```

### 6. Validità Esterna (Dataset LANL Reale)
Avvia la pipeline di inferenza isolata sul dataset governativo pubblico **LANL Comprehensive Multi-Source**. Fondamentale per provare sul campo l'efficacia del *cost-sensitive routing* su attacchi laterali reali (red-team).

> **Prerequisito**: I file pesanti `auth.txt.gz` e `redteam.txt` (scaricati manualmente dal sito csr.lanl.gov) devono risiedere nella cartella `./data/` locale (ignorata su GitHub).

```bash
docker compose --profile eval-lanl up
```

## Dettagli tecnici
Gli artifact (`tgn_checkpoint.pt`, `tgn_stats.json`) sono automaticamente persistiti tramite un volume bind-mount sulla cartella `./public` dell'host. Tali artefatti potranno poi essere utilizzati dai microservizi di validazione ZTA (Zero Trust Architecture).
