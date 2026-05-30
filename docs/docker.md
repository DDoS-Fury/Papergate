# Esecuzione con Docker Compose

Per semplificare l'utilizzo del progetto Graphagate su GPU (ottimizzato per RTX Blackwell, CUDA 13), è stato predisposto un file `docker-compose.yml` con profili dedicati. 
Questa struttura permette di eseguire le diverse fasi del progetto isolando gli ambienti e senza dover ricordare complessi comandi Docker.

## Prerequisiti
- Docker e Docker Compose installati
- NVIDIA Container Toolkit installato e configurato (per l'utilizzo della GPU)
- Driver NVIDIA compatibili con CUDA 13 (es. driver per RTX 5090 / B200 Blackwell)

## Profili Disponibili

Il `docker-compose.yml` contiene due profili: `training-tgn` e `verify-tgn`.

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

Gli artifact TGN persistiti in `public/` sono:
- `tgn_checkpoint.pt` — pesi (inclusi identità di nodo e teste di scoring) + buffer di
  memoria + raw-message store + buffer del neighbour loader (per continuare esattamente
  lo stato temporale e la storia dei vicini tra riavvii);
- `tgn_stats.json` — soglia di decisione calibrata, `capacity` e mappatura
  `NodeRegistry` (entità esterne → slot di memoria).

## Dettagli tecnici
Gli artifact (`tgn_checkpoint.pt`, `tgn_stats.json`) sono automaticamente persistiti tramite un volume bind-mount sulla cartella `./public` dell'host. Tali artefatti potranno poi essere utilizzati dai microservizi di validazione ZTA (Zero Trust Architecture).
