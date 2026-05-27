# Esecuzione con Docker Compose

Per semplificare l'utilizzo del progetto Graphagate su GPU (ottimizzato per RTX Blackwell, CUDA 13), è stato predisposto un file `docker-compose.yml` con profili dedicati. 
Questa struttura permette di eseguire le diverse fasi del progetto isolando gli ambienti e senza dover ricordare complessi comandi Docker.

## Prerequisiti
- Docker e Docker Compose installati
- NVIDIA Container Toolkit installato e configurato (per l'utilizzo della GPU)
- Driver NVIDIA compatibili con CUDA 13 (es. driver per RTX 5090 / B200 Blackwell)

## Profili Disponibili

Il `docker-compose.yml` contiene due profili principali: `training` e `inference`.

### 1. Training (GAE Originale)
Avvia il container per l'addestramento non supervisionato del GAE sui dati benigni (generati o forniti). Il modello risultante, così come le statistiche di normalizzazione, verranno salvati nella cartella `public/`.

```bash
docker compose --profile training up
```
*(Esegue in background `python -m graphagate.train`)*

### 1.1 Training (Nuovo Modello Temporale TGN)
Avvia il container per l'addestramento della nuova rete dinamica basata su grafi temporali (Temporal Graph Network). Questo profilo genera dati in stream continuo con contesti Zero Trust (JA3, alert snort, sonde).

```bash
docker compose --profile training-tgn up
```
*(Esegue in background `python -m graphagate.train_tgn`)*

### 2. Inferenza ed Esportazione (ONNX)
Questo profilo gestisce sia la valutazione delle anomalie (anomaly score) sia l'esportazione del modello per i deployment in produzione (es. OPA + onnxruntime_go).

```bash
docker compose --profile inference up
```
Questo comando avvia due servizi in parallelo:
- **score**: Inietta anomalie e valuta i risultati tramite ROC-AUC e PR-AUC. *(Esegue `graphagate.score --eval`)*
- **export**: Esporta i pesi e la rete nei formati ONNX statici (bucket S/M/L) in `public/`. *(Esegue `graphagate.export_onnx`)*

## Dettagli tecnici
I file di output come `checkpoint.pt`, `norm_stats.json`, e i modelli `model_{S,M,L}.onnx` sono automaticamente persistiti tramite un volume bind-mount sulla cartella `./public` dell'host. Tali artefatti potranno poi essere utilizzati dai microservizi di validazione ZTA (Zero Trust Architecture).
