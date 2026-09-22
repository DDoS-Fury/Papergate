# Graphagate / Papergate — analisi del repository

> Documento di orientamento redatto il 22 settembre 2026 su MacBook Pro M5 Pro (arm64,
> Docker Desktop 4.88, Python 3.12). Tutte le affermazioni marcate «verificato» sono state
> eseguite su questa macchina; il resto è lettura del codice e della documentazione.

---

## 1. Cos'è

**Graphagate** è un microservizio di *anomaly detection* per architetture Zero Trust (ZTA).
Analizza in tempo reale lo stream di accessi di un'organizzazione e assegna a ogni singola
richiesta un punteggio di anomalia fra 0 e 1, che un orchestratore di sicurezza passa poi al
proprio Policy Decision Point (OPA) per decidere se consentire l'accesso.

Il modello è una **Temporal Graph Network (TGN)** addestrata in regime *one-class*: vede in
addestramento **solo traffico benigno** e impara la "normalità comportamentale" di ogni entità.
Non è un classificatore supervisionato — le etichette servono solo a selezionare il training set
e a calibrare la soglia operativa, mai a calcolare un punteggio di test.

Ogni richiesta HTTP viene modellata come una **catena causale a 5 nodi**:

```
source (IP) → config (JA3) → device (TPM/cookie) → user → resource (URI)
```

più un arco di *binding* `config → user`, per un totale di 5 archi temporali per evento. Ogni
entità possiede uno stato ricorrente (memoria GRU) e una cronologia limitata dei suoi vicini
temporali recenti.

### Le quattro classi di anomalia

| Classe | Descrizione | Chi la cattura davvero |
|---|---|---|
| **Policy** | Ruolo/clearance/tier insufficienti | È di OPA, a monte — il modello non aggiunge valore |
| **Contextual** | JA3 compromesso, alert Snort/sensori | Banale, ~97% catturata da una regola |
| **Lateral movement** | Accesso autorizzato ma **non abituale** | ⭐ **Il vero bersaglio ML** |
| **Credential theft** | Client/tool diverso da quello abituale che riusa credenziali valide | ⭐ **Il vero bersaglio ML** (introdotto dallo schema v4) |

Il punto centrale del progetto: *lateral* e *credential theft* hanno feature d'arco **identiche**
al traffico benigno. Si distinguono solo dal **pattern temporale e relazionale** — ed è
esattamente quello che memoria ricorrente e vicinato temporale modellano.

### Papergate vs Graphagate

Il repository si chiama **Papergate** ed è il ramo destinato alla pubblicazione scientifica;
il prodotto software si chiama **Graphagate** (è il nome del pacchetto Python e il titolo del
README). I remote lo confermano:

```
origin    https://github.com/DDoS-Fury/Papergate.git     (questo repo)
upstream  https://github.com/DDoS-Fury/Graphagate.git    (il progetto software)
```

---

## 2. Mappa del repository

```
src/                          ~4.700 righe Python — il cuore
├── config.py                 TGNConfig: tutti gli iperparametri in un dataclass congelato
├── data/stream_synthetic.py  generatore dello stream sintetico ZTA (1.102 righe)
├── model/
│   ├── tgn.py                architettura: TGNMemory + identità hashed + GNN + doppio scorer
│   ├── neighbor.py           MessageNeighborLoader: ring buffer limitato in RAM
│   └── registry.py           NodeRegistry: chiavi esterne → slot di memoria, con eviction LRU
├── train_tgn.py              training + calibrazione soglia + valutazione per classe (1.188 righe)
├── serve_tgn.py              primitive di serving e persistenza (606 righe)
├── serve_api.py              microservizio REST/JSON FastAPI — il deployabile (389 righe)
├── calibration.py            soglie cost-sensitive e routing per segnale
└── verify_tgn.py             harness di verifica della correttezza di streaming

tests/                        suite pytest + baseline comparative + ablazioni + dataset esterni
├── baselines/                Isolation Forest, OC-SVM, GNN statica, TGN 2-nodi, XGBoost
├── ablations/                studi di ablazione (config node, guest device, sweep architetturale)
├── datasets/                 adattatori LANL auth e PicoDomain
└── verify_replay_batching.py ⭐ il test di contratto centrale (vedi §4)

docs/
├── paper/                    manoscritto IEEEtran (main.tex, results.tex, refs.bib)
├── orchestrator_integration.md   come l'orchestratore ZTA/OPA parla col servizio
├── data_validation_methodology.md  difesa metodologica dei dati (in italiano)
└── policies.txt              policy OPA/Rego di riferimento del sistema ZTA ospite

gui/                          visualizzatore 3D WebGL (Three.js) — non documentato nel README
notebooks/train_and_eval.ipynb  percorso di training locale con accelerazione MPS
tasks/runs/                   log e JSON delle run citate nel paper (panelA.json, panelB.json, …)
docker/Dockerfile             immagine GPU (CUDA 13) per train/verify/serve
docker-compose.yml            17 profili: training, serving, baseline, ablazioni, valutazioni
public/                       artefatti generati dal training — gitignored, attualmente VUOTO
```

---

## 3. Come funziona

### 3.1 Architettura del modello

```
evento → NodeRegistry → MessageNeighborLoader → embed() → score() → soglia → decisione
```

**Componenti** (tutti in `src/model/tgn.py`, classe `ZTATemporalGraphNetwork`):

- **TGNMemory** — stato ricorrente per nodo aggiornato da una GRU dai messaggi d'evento.
  È la "memoria storica" del comportamento di ogni entità (`memory_dim=256`).
- **Hashed Identity** — embedding appreso tramite hashing deterministico della chiave
  (BLAKE2b, `stable_hash`). Mantiene il modello **100% induttivo**: un'entità mai vista prima
  riceve comunque un'identità consistente fra processi e riavvii. L'`hash()` builtin di Python
  è salato per processo e avrebbe rotto la riproducibilità — errore già individuato e corretto,
  con test di regressione dedicato.
- **MessageNeighborLoader** — ring buffer `[num_nodes, K=30]` con gli ultimi vicini temporali
  di ogni nodo, **senza alcun database a grafo**. Memoria costante `O(num_nodes·K·msg_dim)`.
- **GraphAttentionEmbedding** — stack `TransformerConv` multi-hop (3 hop, 4 teste, residuali).
- **Doppia testa di scoring**:
  - *feature head* (MLP, obiettivo InfoNCE) → anomalie policy e contextual;
  - *structural head* (coseno scalato fra proiezioni) → lateral movement.
  - `logit = feat_logit + struct_logit`, `anomaly_score = 1 − σ(logit)`.
- **Kill-chain precursor** — prior moltiplicativo *a serving time* (non addestrato) che alza il
  punteggio di un'entità subito dopo un suo alert, con decadimento `0.5^(Δt/half_life)`.
  Modella la sequenza recon → lateral.

### 3.2 Il flusso per evento (serving)

1. `NodeRegistry` mappa le 5 chiavi esterne → slot di memoria, ammettendo entità mai viste.
2. Il neighbor loader espande i nodi al loro vicinato temporale storico.
3. `embed()` legge la memoria, concatena identità e feature statiche, esegue la GNN.
4. `score()` somma le due teste → logit → punteggio di anomalia.
5. **Il decisore esterno (OPA) risponde ALLOW/DENY.** Solo su ALLOW la memoria viene aggiornata
   e l'arco inserito nel neighbor loader (*predict-then-update*).

### 3.3 Il gate anti-poisoning — il punto concettuale più interessante

La memoria si aggiorna **solo** sugli eventi ammessi dal decisore esterno. Questo evita che un
attaccante avveleni la baseline semplicemente generando traffico. Il README è esplicito su una
distinzione che è facile perdere:

> **Il gate misurato è quello di OPA, non il punteggio del modello.**

Esistono due modalità:

| Modalità | Endpoint | Chi decide il commit | Usata per le misure? |
|---|---|---|---|
| **OPA-in-the-loop** | `/infer` → OPA → `/update` | OPA (proxy offline: `not signal_dirty`) | ✅ Sì |
| **OPA-less** | `/score` | Il modello stesso | ❌ No — disponibile ma non misurata |

La seconda esiste ed è comoda per i test, ma lasciare che il modello decida da solo cosa
memorizzare fa esplodere il tasso di falsi positivi. È una scelta di misurazione conservativa
e onesta, dichiarata invece che nascosta.

### 3.4 Calibrazione della soglia

Il modello *ordina* bene le anomalie (AUC lateral ~0,72) ma una singola soglia che tenga l'1% di
falsi positivi finisce sopra il punto dove si addensano i lateral, e il recall operativo crolla.
La soluzione (`src/calibration.py`) è duplice:

- **Soglia cost-sensitive**: minimizza `cost_ratio · FN + FP` con `cost_ratio=20`, con un tetto
  di sicurezza al 5% di FPR benigno.
- **Signal routing**: la soglia orientata al recall si applica solo allo stream *signal-clean*
  (dove benigno e lateral sono altrimenti indistinguibili); gli eventi *signal-dirty* — già
  catturati da una regola banale — usano la soglia conservativa all'1%.

---

## 4. Stato verificato su questa macchina

Tutti i comandi sotto sono stati eseguiti il 22 settembre 2026 e sono passati.

```
pytest                              36 passed, 9 skipped     (skip = PicoDomain assente)
pytest tests/test_leakage_audit.py  14 passed
python tests/test_stable_hash.py    ALL CHECKS PASSED
python tests/verify_replay_batching.py   OVERALL PARITY: PASS
```

### Il test di contratto centrale

`tests/verify_replay_batching.py` merita una menzione a parte. La valutazione offline
(`train_tgn._replay`) **non** chiama le primitive di serving: è una re-implementazione
vettorizzata indipendente. Ciò che lega i due percorsi non è codice condiviso ma un test di
equivalenza. Risultati misurati qui:

```
[v3 3-edge | calib(gate_by_label)]   max|Δ| = 3,563e-07   → PASS
[v3 3-edge | eval(routed)]           max|Δ| = 1,834e-07   → PASS
```

Tolleranza 1e-5, quindi due ordini di grandezza di margine. Una versione precedente del README
sosteneva che i due percorsi condividessero il codice: non era vero, ed è stato corretto.

### Il drift da batching — da tenere a mente

Lo stesso test stampa un `DRIFT` informativo che **non** fa fallire la suite:

```
DRIFT bs=64  vs bs=1:  max|Δ| = 0,120 – 0,243
DRIFT bs=256 vs bs=1:  max|Δ| = 0,138 – 0,301
```

È coerente col design (un blocco viene scorato contro lo snapshot di memoria a inizio batch), ma
ha una conseguenza operativa: **`eval_batch_size > 1` non produce gli stessi punteggi del
percorso di serving.** Se in futuro citi numeri ottenuti con `eval_batch_size=1024` — l'opzione
documentata in `config.py` per saturare la GPU su dataset grandi come LANL — non sono
confrontabili con quelli a `batch_size=1`.

---

## 5. Utilizzo

### 5.1 Docker (il percorso documentato)

L'immagine è unica per tutti i profili; l'`ENTRYPOINT` è `["python", "-m"]`.

```bash
docker build -f docker/Dockerfile -t graphagate .

docker compose --profile training-tgn up    # training → genera public/
docker compose --profile verify-tgn up      # verifica correttezza streaming
docker compose --profile serve-tgn up       # servizio HTTP (host 8888 → container 8088)
```

Altri profili disponibili: `baseline-iforest`, `baseline-ocsvm`, `baseline-gnn`,
`baseline-xgboost`, `baseline-tgn-2node`, `ablations`, `config-eval`, `guest-device-eval`,
`guest-standard-eval`, `arch-sweep`, `regen-report`, `eval-lanl`, `eval-picodomain`,
`ablation-no-device`.

### 5.2 Su Apple Silicon — serve un override

⚠️ **Il compose base non parte su Mac.** Ogni servizio riserva un device NVIDIA e il daemon
rifiuta: `could not select device driver "nvidia" with capabilities: [[gpu]]`.

Sono stati creati due file di override locali (`docker-compose.cpu.yml` e
`docker-compose.test.cpu.yml`, esclusi da git tramite `.git/info/exclude`):

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml --profile training-tgn up
```

Dettaglio tecnico non ovvio: `devices: []` **non** funziona, perché Compose *appende* alla lista
invece di sostituirla. Serve il tag `!reset`:

```yaml
services:
  train-tgn:
    deploy: !reset null
```

### 5.3 Percorso nativo (venv)

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -e '.[dev]'
./.venv/bin/python -m pytest
```

Il training nativo funziona su Apple Silicon (vedi §7.1), ma **su questo modello conviene
forzare la CPU**: Metal è circa 4,5× più lento (misure in §5.6).

```bash
GRAPHAGATE_DEVICE=cpu ./.venv/bin/python -m graphagate.train_tgn
```

### 5.6 MPS o CPU? — misurato

Controintuitivo ma netto. Stessa configurazione (4000 eventi, 2 epoche), stesso seed:

| Device | Wall time |
|---|---|
| `cpu` | **38,2 s** |
| `mps` | 177,5 s (4,6× più lento) |

**Non è colpa del fallback su CPU introdotto da §7.1**: quelle chiamate pesano 5,2 s su 171 s,
il **3,1%** del totale. È Metal stesso. Il modello è piccolo (`memory_dim=256`, `batch_size=200`)
e ogni evento fa 5 espansioni di vicinato e 5 forward GNN: tantissimi kernel minuscoli, dove il
costo fisso di dispatch su GPU domina e i core CPU di Apple Silicon — con i tensori che stanno
in cache — vincono nettamente.

L'auto-detect sceglie MPS perché è la scelta giusta su hardware NVIDIA e una scelta ragionevole
in generale; qui è quella sbagliata. Per questo `GRAPHAGATE_DEVICE` esiste.

#### Quanto dura il training completo (200k eventi × 15 epoche)

Scaling misurato su CPU a 2 epoche: 4k → 38,2 s · 8k → 83,0 s · 16k → 194,2 s. È
**superlineare**, con esponente in crescita (1,12 → 1,23): evidenza empirica diretta di §7.9,
i dizionari non limitati che crescono con ogni coppia vista.

Scomponendo a 16k eventi — (2 ep, 194,2 s) e (6 ep, 262,8 s) — si ottiene **160 s di costo
fisso e 17 s per epoca**. Il grosso non è il training ma la generazione dello stream e i replay
sequenziali di calibrazione e test a `eval_batch_size=1`.

| Ipotesi di scaling | Stima a 200k × 15 epoche |
|---|---|
| `n^1.15` | ~2,1 h |
| `n^1.25` (più probabile) | ~2,7 h |
| `n^1.35` | ~3,5 h |

Quindi **2-3,5 ore su CPU**, contro le ~10-15 ore che servirebbero su MPS. Da estrapolazione, non
da un run completo: l'esponente cresce con `n` e potrebbe peggiorare ancora oltre i 16k misurati.

> Nota per §8.2: poiché il costo è dominato dalla parte fissa e non dalle epoche, l'opzione B
> (6 run invece di 3) costa davvero circa il doppio in tempo macchina — la stima «×2» è corretta.

#### Verifica sul run reale (22/09, M5 Pro)

L'estrapolazione qui sopra è stata poi verificata su un'epoca reale a 200k eventi, con
`OMP_NUM_THREADS=5` (il perché è nella sottosezione seguente):

| Misura | Valore |
|---|---|
| Epoca completa, 200k eventi | **9 min 28 s** |
| 15 epoche | ~2 h 22 min |
| Run completo, calibrazione ed eval inclusi | **3-4 h** |
| CPU occupata | 322% → 3,2 core sui 15 |
| **Picco RSS** | **15,4 GB su 24 (64%)** |

Due correzioni alle stime precedenti:

* le **2-3,5 h** valevano a thread liberi; a 5 thread il run costa **3-4 h**, ed è un prezzo che
  conviene pagare (sotto);
* il vincolo operativo vero non è il tempo ma la **memoria**. 15,4 GB di picco su 24 lasciano
  meno di 9 GB al resto del sistema: è §7.9 che smette di essere debito teorico e diventa il
  limite pratico del run. È anche la spiegazione più probabile del `JetsamEvent` che il sistema
  ha registrato durante un tentativo precedente.

#### Perché 5 thread e non tutti e 15

A thread liberi macOS ha sospeso la macchina per **emergenza termica tre volte in un'ora** — con
il Mac in carica:

```
12:44:32  Entering Sleep state due to 'Thermal Emergency Sleep'  (Charge 62%)
13:16:54  Entering Sleep state due to 'Thermal Emergency Sleep'  (Charge 82%)
13:51:31  Entering Sleep state due to 'Thermal Emergency Sleep'  (Charge 97%)
```

In quello stato il display non si riaccende e gli input vengono registrati ma ignorati
(`pmset -g log` mostra i `sleepDisplayTickle` da tastiera e da tasto di accensione, seguiti da
`kIOMessageSystemWillSleep`): dall'esterno è indistinguibile da un blocco totale, e invita a uno
spegnimento forzato che costa l'intero run.

Passare a MPS **non** risolve il problema: CPU e GPU condividono lo stesso die e lo stesso budget
termico, e un run 4,6× più lungo incontra più finestre di emergenza termica, non meno. In più, il
profilo dominato dal dispatch (sopra) tiene comunque un core CPU occupato al 100% mentre accende
la GPU. La leva giusta è abbassare i watt, non spalmarli: con `OMP_NUM_THREADS=5` il processo usa
~3,2 core e l'emergenza termica non è scattata.

Da tenere presente anche il `sleep 1` in `pmset -g custom`: la macchina si sospende dopo **un
minuto** di inattività dell'utente, e il carico CPU non lo impedisce. Un run lasciato solo va
avviato sotto `caffeinate -i`.

### 5.4 API HTTP

| Endpoint | Cosa fa | Muta lo stato? |
|---|---|---|
| `GET /health` | Readiness; 503 finché il checkpoint carica | no |
| `POST /infer` | Scora senza avanzare la memoria | solo ammissione slot |
| `POST /update` | Committa un evento già giudicato benigno (post-ALLOW) | **sì** |
| `POST /score` | Scora + gate interno + update condizionale (OPA-less) | **sì** |
| `POST /persist` | Riscrive lo stato in RAM su `public/` | scrive su disco |
| `WS /stream` | Stream live per la GUI 3D | — |

**Il servizio deve girare con un solo worker.** Il modello è stato mutabile in RAM; più worker
avrebbero copie divergenti e si sovrascriverebbero al salvataggio.

### 5.5 Paper

```bash
./scripts/build_paper.sh          # autodetect engine, compila, pulisce → docs/paper/main.pdf
```

---

## 6. Punti di forza

**1. Onestà metodologica fuori dal comune.** È la qualità più rara di questo repository. Il
README *ritratta esplicitamente* numeri che non hanno un log a supporto, invece di ammorbidirli:

> «un numero senza un log a supporto non si ammorbidisce, si ritratta»

I delta di ablazione per componente sono marcati RETRACTED perché due serie precedenti si
contraddicevano e nessuna aveva un file di run dietro. Le righe baseline del Panel A sono
dichiarate non comparabili finché non vengono rigenerate. Anche la portata del claim `O(1)` è
delimitata con precisione, elencando cosa *non* è O(1). Questo atteggiamento vale più di
qualunque numero: rende il lavoro difendibile in revisione.

**2. L'audit di de-leakage è un guard-rail permanente, non un controllo una-tantum.**
`tests/test_leakage_audit.py` verifica su 3 seed che nessuna colonna singola separi una classe
(AUC ≤ 0,60 sul lateral), che non esistano impronte a valore costante e che benigno e lateral
condividano la distribuzione marginale delle destinazioni (test KS). La storia è istruttiva: una
versione precedente del generatore aveva una colonna (`node_feat[dst,3]`, l'indice di risorsa)
che da sola raggiungeva AUC 0,92 sul lateral. Dopo la correzione il floor è 0,603. Quella
colonna oggi è tenuta a 0,0 *di proposito*, come invariante di regressione.

**3. La de-degenerazione del task.** Con `benign_explore_prob=0.15` il traffico benigno compie
a volte accessi legittimi non abituali. Senza questo, il task sarebbe la tautologia
«non-abituale ⟺ malevolo» e qualunque modello lo risolverebbe. È una scelta che *abbassa* i
numeri pubblicati e li rende veri.

**4. Il contratto train/serve è testato, non assunto.** Vedi §4.

**5. Statefulness gestita con criterio.** Nessun database vettoriale o a grafo: buffer
pre-allocati a dimensione fissa, lookup O(1) sulla riga del nodo coinvolto, ring buffer per la
storia. Per un servizio che deve girare indefinitamente è la scelta giusta, e i limiti di questa
scelta sono dichiarati invece che nascosti.

**6. Documentazione del codice di qualità superiore alla media.** I docstring spiegano il
*perché*, non il *cosa* — incluse le decisioni negative ("questa matrice del generatore non è
usata di proposito, userebbe reintrodurre circolarità").

---

## 7. Punti deboli e cose da sistemare

Ordinati per priorità pratica.

### 7.1 ✅ RISOLTO — percorso MPS nativo

**Era:** `train_tgn.py` selezionava MPS quando CUDA è assente, ma PyTorch su Metal non supporta
`scatter_reduce_` con `int64` e `reduce='max'` — operazione che `torch_geometric.nn.models.tgn`
esegue sui **timestamp** a ogni forward. `python -m graphagate.train_tgn` su Mac crashava con
`RuntimeError: not supported for torch.int64`. La monkeypatch che lo aggirava viveva solo in
`notebooks/train_and_eval.ipynb`.

**Risolto** con il nuovo modulo `src/mps_compat.py`, che accentra due cose:

- `resolve_device()` — unica sede della scelta del device, con override `GRAPHAGATE_DEVICE`
  (`cpu` | `mps` | `cuda` | `cuda:1`). Senza variabile: auto-detect CUDA → MPS → CPU. Un
  backend richiesto ma non disponibile solleva un errore invece di degradare in silenzio.
- `apply_mps_compat_patches()` — avvolge `scatter` e `scatter_argmax` dentro il namespace di
  `torch_geometric.nn.models.tgn`, eseguendo su **CPU** le sole riduzioni int64 che Metal
  rifiuta. Idempotente, trasparente per i tensori float.

> ⚠️ **La patch del notebook era sottilmente sbagliata e va sostituita, non copiata.**
> Convertiva a `float32`, che è esatto solo fino a `2**24 = 16 777 216`. Gli epoch unix sono
> ~1,76e9, dove la spaziatura fra valori rappresentabili è di 128 secondi. Misurato:
>
> ```
> t       = [1758531600, 1758531603, 1758531607, 1758531611]   (due gruppi da due)
> esatto  = [1758531603, 1758531611]
> float32 = [1758531584, 1758531584]     # i due gruppi collassano sullo stesso valore
> ```
>
> Errore di 19-27 s e gruppi indistinguibili. Poiché `last_update` alimenta ogni Δt del modello
> — recency della coppia, attività della sorgente, time encoder — avrebbe corrotto in silenzio
> proprio il segnale temporale che la TGN esiste per modellare: nessun crash, solo risultati su
> Mac diversi da quelli su CUDA. Il fallback su CPU è esatto per qualsiasi magnitudine e costa
> nulla (riduce tensori di timestamp da poche centinaia di elementi, non le feature).
>
> La cella è stata sostituita nel notebook.

**Verificato:** patch esatta contro il riferimento CPU (`scatter` e `scatter_argmax`), i quattro
percorsi di `GRAPHAGATE_DEVICE`, un training end-to-end su MPS, il percorso CPU in container, e
la suite completa senza regressioni (36 passed, 9 skipped).

**Portabilità — nessun impatto fuori da Apple Silicon.** La patch si applica *solo* se il device
risolto è MPS; altrove `torch_geometric.nn.models.tgn.scatter` resta la funzione originale.
Verificato per simulazione su tutte le combinazioni e, per il caso Linux, dentro un container
reale:

| Piattaforma | Device | Patch applicata |
|---|---|---|
| Windows / Linux + CUDA | `cuda` | no |
| Windows / Linux senza GPU | `cpu` | no |
| Mac Intel | `cpu` | no |
| torch privo dell'attributo `mps`, o con `mps` che solleva | `cpu` | no |
| Mac Apple Silicon | `mps` | sì |

L'ordine di auto-detect (CUDA → MPS → CPU) è identico al codice precedente, quindi per chi non è
su Apple Silicon il comportamento non cambia. `_mps_available()` è anzi più difensivo
dell'originale, che chiamava `torch.backends.mps.is_available()` senza guardia. `GRAPHAGATE_DEVICE`
è opt-in: non impostata, il comportamento è quello di prima. L'unica differenza osservabile è la
riga di log, che ora indica l'origine della scelta — nessuno la parsifica.

> **Nota per Docker:** il compose monta `./src`, quindi il nuovo modulo è già visibile. Per un
> `docker run` senza mount serve ricostruire l'immagine.

### 7.2 🔴 `torch.load(..., weights_only=False)` + endpoint non autenticati

`src/serve_tgn.py:584` carica il checkpoint con `weights_only=False`, che consente esecuzione di
codice arbitrario se il file non è fidato. Preso da solo è accettabile (il checkpoint lo produci
tu). Diventa serio in combinazione con il fatto — già dichiarato nei Limiti del README — che
`/update`, `/score` e `/persist` non hanno autenticazione: chi raggiunge il servizio può alterarne
lo stato e, tramite `/persist`, scrivere su disco. Il design presuppone un orchestratore fidato
su rete privata, ma vale la pena scriverlo come requisito di deployment esplicito, non solo come
limite noto.

### 7.3 🟠 L'immagine Docker porta l'intero stack CUDA che non userà mai su arm64

La base `nvidia/cuda:13.2.1-base-ubuntu24.04` ha un manifest arm64, quindi builda. Ma pip tira
`torch 2.12.1+cu130` e con esso 17 pacchetti `nvidia-*` più `cuda-toolkit-13.0.2` e `triton`:

```
nvidia-cublas-13.1.1.3  nvidia-cudnn-cu13-9.20.0.48  nvidia-nccl-cu13-2.29.7
nvidia-cusolver  nvidia-cusparse  nvidia-cufft  nvidia-curand  …
```

Sono ~2,5 GB dei 3,46 GB totali, e su Mac non sono solo inutilizzati: sono **incaricabili**
(`torch.cuda.is_available()` → `False`, verificato). Build: 256 s di pip + 95 s di export layer.

**Fix**: una riga nel Dockerfile prima di `pip install -e .`

```dockerfile
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.12.*
```

Immagine attesa attorno a 1 GB.

### 7.4 🟠 MPS assente dal percorso di serving

`train_tgn.py` gestisce MPS; `serve_api.py:114` e `verify_tgn.py:54` fanno solo
`cuda if available else cpu`. Su Mac servi in CPU pur avendo la GPU disponibile.
Incoerenza fra i due percorsi. Ora che §7.1 ha centralizzato la scelta in
`graphagate.mps_compat.resolve_device()`, allinearli è una riga per file.

### 7.5 🟠 `requirements.txt` in drift con `pyproject.toml`

`requirements.txt` non elenca `fastapi`, `uvicorn`, `scipy` né `aiohttp`. Il Dockerfile usa
`pip install -e .`, quindi la fonte di verità è `pyproject.toml` e `requirements.txt` è
fuorviante per chi lo prende alla lettera. O si allinea o si elimina.

### 7.6 🟡 `gui/serve.py:24` — bug specifico di macOS

```python
if e.errno == 98:  # Address already in use
```

`98` è `EADDRINUSE` su Linux; su macOS è **48** (verificato). Il fallback "porta occupata → prova
la successiva" non scatta mai e lo script solleva l'eccezione. Fix di una riga:
`if e.errno == errno.EADDRINUSE:`.

### 7.7 🟡 `emissions.csv` versionati

Sono presenti in `emissions.csv` e `tests/emissions.csv`: output di codecarbon, non dovrebbero
stare sotto controllo di versione.

### 7.8 🟡 La GUI non è documentata e dipende da un CDN

`gui/` (1.202 righe fra JS, HTML e CSS) non compare nella sezione «Project layout» del README,
insieme a `notebooks/` e `tasks/`. Inoltre `gui/index.html` carica Three.js da `unpkg.com`
tramite importmap: senza connessione a internet il visualizzatore non parte.

### 7.9 🟡 Lo stato del modello non è interamente limitato

Il README lo dichiara già, ma vale la pena tenerlo in cima alla lista dei debiti tecnici:
`last_contact`, `pair_count`, `src_count` e `recent_alert` (`src/model/tgn.py:156-169`) sono
dizionari Python **non limitati**, crescono con ogni coppia `(src, dst)` mai committata e vengono
ripuliti solo per-slot all'eviction. Inoltre l'eviction stessa non è O(1): `_select_eviction` è
un `min` sull'intera capacità e `reset_node` fa una scansione `O(num_nodes × K)`. Per un servizio
che deve girare indefinitamente è *il* limite di scalabilità da affrontare per primo.

---

## 8. Punti aperti e posizioni prese per il paper

> **Fuori scope.** La rigenerazione dei delta di ablazione per componente e delle righe baseline
> del Panel A — ritrattati nel README — riguardava il progetto universitario e **non** il paper.
> Non è un punto aperto per questo lavoro.

### 8.1 Validità esterna — ✅ da dichiarare nei Limiti

**Posizione presa:** è un problema reale, va dichiarato esplicitamente nella sezione Limiti del
paper, inquadrato anche come **gap noto del campo** e non come debolezza di questo lavoro.

Il fatto di supporto è forte: non esiste alcun dataset ZTA pubblico che fornisca insieme
identità utente, fingerprint TLS e ground truth di lateral movement. I benchmark di
autenticazione (LANL Cyber1, DARPA OpTC) omettono i fingerprint TLS; quelli di traffico di rete
(CIC-IDS2017) omettono le identità utente o il lateral etichettato.

Quello che va dichiarato con precisione:

- **PicoDomain** è l'unico corpus in cui tutti e 5 i nodi poggiano su campi reali (`ssl.log`
  porta il `ja3`), ed è misurato in **una singola run** — AUC aggregata 0,6658,
  `tasks/runs/picodomain_eval_docker.log`. Il recall @threshold non è riportato per costruzione
  dello split.
- Su **LANL** il nodo Configuration degenera e la classe credential-theft **non è valutabile**:
  è di fatto l'ablazione «senza config node», non un benchmark alla pari. Presentarlo come
  validazione esterna piena sarebbe scorretto.

### 8.2 Varianza run-to-run vs cross-seed — 🔶 decisione aperta

**Stato:** in valutazione fra due opzioni. Le σ attualmente pubblicate conflano le due sorgenti
di variabilità e sottostimano l'errore.

Il disegno attuale è **3 seed `[42, 7, 123]` × 1 run** — confermato dai metadati di entrambi
gli artefatti (`tasks/runs/panelA.json`, `panelB.json`: `"seeds": [42, 7, 123]`) e da 7 punti
del README. Nel paper vanno dichiarati 3, non 5.

| | Disegno | Run totali | Costo | Cosa si ottiene |
|---|---|---|---|---|
| **A** | 3 seed × 1 run, dichiarando che la σ è **solo cross-seed** | 3 | zero | Onestà piena, isolamento run-to-run come lavoro futuro |
| **B** | 3 seed × 2 repliche | 6 | ×2 training | Stima grezza di entrambe le componenti |

Nota di contesto per la decisione: il README stesso indica come standard auspicato una griglia
**≥5 seed × 3 repliche** (15 run). Né A né B ci arrivano — B è un compromesso pragmatico, non lo
standard che il documento si era dato. Se scegliete B, conviene dichiararlo come tale nel paper:
un revisore che legge il repo noterebbe altrimenti la discrepanza fra lo standard dichiarato e
quello applicato.

### 8.3 Recall operativo sul lateral — ✅ da presentare come trade-off

**Posizione presa:** non è un problema da risolvere ora. Va presentato con onestà come
**compromesso regolabile**, non come risultato consolidato.

Dal 16,1% (soglia globale all'1% FPR) si sale al 48,5% con il routing cost-sensitive e il
config node (v4, FPR benigno 5,7%). La leva è `cost_ratio` / `clean_fpr_cap`: alzare il recall
costa falsi positivi che l'orchestratore deve ri-sfidare. È un punto operativo scelto, non un
ottimo.

### 8.4 Punti tecnici non ancora affrontati

**La structural head potrebbe essere superflua.** Il README la descrive come «marginale» in
misurazioni precedenti e la marca come candidata alla semplificazione. Se confermato, è codice
e complessità da rimuovere.

**Il cold start non è coperto.** Un'entità nuova non ha abitudini da cui deviare. Nello stream
sintetico tutti i lateral atterrano su entità già calde (`n_cold=0`), quindi lì non è il collo
di bottiglia — ma in produzione lo sarebbe. Candidato naturale per i Limiti, accanto a §8.1.

**Prerequisito manuale non automatizzato.** `docker-compose.test.yml` richiede
`cp -r public .test_public` eseguito a mano prima del primo test live, altrimenti fallisce in
modo poco esplicito.

**Ambiguità di naming.** Repo `Papergate`, pacchetto e README `Graphagate`, container
`graphagate-*`. Non è un errore ma va spiegato a chi arriva nuovo (vedi §1).

---

## 9. Da dove ripartire

Stato attuale: **`public/` è vuoto** (gli artefatti sono gitignored e non sono mai stati
generati su questa macchina). Senza checkpoint, `serve-tgn` e `verify-tgn` falliscono con
`FileNotFoundError: /app/public/tgn_checkpoint.pt` — verificato.

Nell'ordine:

1. **Generare gli artefatti.** Il training è 200k eventi × 15 epoche. Il percorso più rapido su
   questa macchina è quello nativo **su CPU** — non su Metal, che qui è 4,6× più lento (§5.6):

   ```bash
   nohup caffeinate -i env OMP_NUM_THREADS=5 GRAPHAGATE_DEVICE=cpu \
     ./.venv/bin/python -u -m graphagate.train_tgn \
     > tasks/runs/tgn_full_cpu.log 2>&1 &
   # → public/tgn_checkpoint.pt + public/tgn_stats.json
   ```

   **3-4 ore**, misurate (§5.6). `caffeinate -i` neutralizza il `sleep 1` delle impostazioni
   energia, `OMP_NUM_THREADS=5` tiene il SoC sotto la soglia di emergenza termica e `nohup` fa
   sopravvivere il run alla chiusura del terminale. Chiudere prima i browser: il picco è 15,4 GB
   su 24.

   Il training salva lo stato a ogni epoca in `public/tgn_resume.pt` e riparte da lì: una
   sospensione termica, un kill per memoria o un'interruzione di corrente costano al massimo
   un'epoca. `GRAPHAGATE_RESUME=0` forza un run pulito.
2. **Fix rapidi e isolati**: §7.6 (errno), §7.5 (requirements), §7.7 (emissions.csv).
   Aggiungere §7.4, ora che è una riga per file.
3. **Debito strutturale**: §7.9, i dizionari non limitati — il vero limite alla messa in
   produzione.
4. **Opzionale**: il profilo `regen-report` resta utile se si vuole decidere sulla structural
   head (§8.4), non per altro.

Sul fronte paper, l'unica decisione che blocca del lavoro è **§8.2**: se scegliete l'opzione B
servono 3 run di training in più, e conviene saperlo prima di pianificare. §8.1 e §8.3 sono
già risolte come posizioni editoriali e non richiedono nuove misure.
