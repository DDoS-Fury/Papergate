# Integrazione Orchestrator ZTA e Modello TGN

Questo documento chiarisce l'architettura di integrazione tra il Security Orchestrator (che comunica con il Policy Decision Point, es. OPA) e il microservizio AI basato su TGN (Temporal Graph Network).

## Nessun Database Vettoriale Necessario

Una domanda comune nell'integrazione di modelli AI per l'Anomaly Detection strutturale (come i grafi) è se sia necessario mantenere un vector database esterno (es. Milvus, Pinecone) per storicizzare gli embedding o le tuple delle richieste passate.

**La risposta per il TGN è no.**

Il modello TGN è stato progettato appositamente per essere **stateful** e gestire autonomamente la propria memoria temporale in RAM tramite tensori PyTorch. Anche la storia strutturale (gli ultimi `K` vicini temporali di ogni entità) è mantenuta in RAM da un **neighbour loader bounded** (`MessageNeighborLoader`, un ring-buffer a dimensione fissa `O(num_nodes·K)`): è ciò che consente il rilevamento del *lateral movement* **senza** alcun graph database esterno.

### Flusso di Esecuzione (Serving)

1. **Inoltro della Richiesta (Tupla)**
   L'orchestrator ZTA non deve pre-processare vettori né interrogare database storici. Deve semplicemente inoltrare la singola transazione (o evento) grezza all'API di serving del modello (`src/serve_tgn.py -> score_event`). La tupla minima richiesta include:
   - Identificativo sorgente (es. IP, Username).
   - Identificativo destinazione (es. URI della risorsa).
   - Timestamp (es. Unix epoch).
   - Array di feature contestuali dell'arco (es. trust di JA3, allarmi Snort, sonde, metodo HTTP).
   - **Attributi statici delle entità** (`src_feat` / `dst_feat`): ruolo, clearance,
     device tier. L'orchestrator/OPA li conosce già per ogni richiesta, quindi vengono
     passati per-evento (nessun datastore aggiuntivo). Sono il segnale che permette al
     modello di rilevare le **violazioni di policy** — anomalie che hanno feature d'arco
     identiche al traffico benigno. Poiché il training avviene su dati sintetici, gli
     utenti in produzione saranno tutti "nuovi": è **obbligatorio** passare queste feature
     perché il modello conosca i privilegi dell'utente reale appena incontrato.

2. **Gestione del `NodeRegistry`**
   All'arrivo di una tupla, il TGN utilizza il suo `NodeRegistry` per mappare le chiavi alfanumeriche (es. un nuovo indirizzo IP mai visto prima) in indici interi in tempo reale. Il sistema supporta l'ingresso di nodi non visti durante il training (spazio dei nodi dinamico e illimitato).

3. **Integrazione con la Memoria TGN e il vicinato**
   Il modello accede allo stato storico dei nodi coinvolti leggendo i propri tensori interni: la memoria ricorrente (`model.memory`) **e** il vicinato temporale recente (`model.neighbor_loader`). Concatena la memoria con l'identità apprendibile di nodo, fa girare la GNN sui vicini reali e combina la *feature head* (policy/contestuale) con la *structural head* (lateral movement). Viene calcolato l'anomaly score (`1 − P(benigno)`, da `0.0` a `1.0`) e restituito all'orchestrator, che lo girerà ad OPA.

4. **Aggiornamento "Anti-Poisoning" (Gatekeeper OPA)**
   Affinché OPA sia il vero decisore finale, l'Orchestrator gestisce le primitive del modello in due step (invece del gate interno di `score_event`):
   - Chiama **`infer_score`** per ottenere l'anomaly score (operazione di sola lettura: non muta né memoria né vicinato).
   - Invia la richiesta e lo score a OPA.
   - Se **OPA risponde ALLOW** (l'evento è totalmente lecito e non anomalo), l'Orchestrator chiama **`update_memory`**, che avanza la memoria TGN **e** inserisce l'arco nel neighbour loader.
   - Se **OPA risponde DENY**, la chiamata a `update_memory` viene omessa. Questo impedisce in modo assoluto agli attaccanti di fare "poisoning" sul modello, garantendo che il TGN impari solo da ciò che OPA ha esplicitamente approvato — sia nella memoria sia nella storia dei vicini.

   > Nota: `infer_score` / `update_memory` lavorano su indici di slot già mappati dal
   > `NodeRegistry`; gli attributi statici per-evento vanno scritti nello slot prima dello
   > scoring (è ciò che fa internamente `score_event`).

### Persistenza

L'unico storage richiesto per questo strato AI è il filesystem. Il comando di salvataggio del modello (`save_model`) serializza sul disco l'intero stato:
- I pesi addestrati della rete (inclusi l'identità di nodo e le due teste di scoring).
- I tensori in memoria con le cronologie degli accessi (TGN Memory) + il raw-message store.
- I buffer del neighbour loader (gli ultimi `K` vicini temporali per nodo).
- Il dizionario del NodeRegistry.

Questo file (`public/tgn_checkpoint.pt`) assieme ai metadati (`public/tgn_stats.json`) consente al microservizio AI di ripartire esattamente dal punto in cui era stato interrotto senza perdere il contesto storico degli utenti.

## API HTTP (servizio di inferenza)

Le primitive descritte sopra sono esposte come **microservizio REST/JSON** da
`src/serve_api.py` (FastAPI + uvicorn), avviato con `python -m graphagate.serve_api`
(profilo Docker Compose `serve-tgn`, porta `8088`). L'orchestrator Go vi parla con
`net/http` + `encoding/json` — nessun `.proto`/gRPC da mantenere.

### Avvio del servizio

**Prerequisito**: il servizio carica gli artifact `public/tgn_checkpoint.pt` e
`public/tgn_stats.json`. Vanno prodotti **prima**, una volta, dal training
(`docker compose --profile training-tgn up`). Senza di essi il servizio non parte.

Avvio come servizio (long-running):

```bash
# Via Docker Compose (profilo dedicato, espone :8088 e l'healthcheck su /health)
docker compose --profile serve-tgn up

# Oppure standalone, riusando la stessa immagine
docker run --rm --gpus all -p 8088:8088 \
  -v "$PWD/public:/app/public" graphagate graphagate.serve_api
```

Il servizio è pronto quando `GET /health` risponde `{"status":"ok","model_loaded":true,...}`
(in Compose l'healthcheck del container lo fa già: dipendere da
`condition: service_healthy` dal lato orchestrator garantisce l'ordine di avvio).

Configurazione via variabili d'ambiente (tutte opzionali):

| Variabile | Default | Ruolo |
|---|---|---|
| `GRAPHAGATE_CHECKPOINT` | `public/tgn_checkpoint.pt` | path del checkpoint (pesi + memoria + vicinato) |
| `GRAPHAGATE_STATS` | `public/tgn_stats.json` | path di soglia calibrata + `NodeRegistry` |
| `GRAPHAGATE_HOST` | `0.0.0.0` | indirizzo di bind |
| `GRAPHAGATE_PORT` | `8088` | porta di bind |

### Endpoint

| Metodo · path | Ruolo | Muta lo stato? |
|---|---|---|
| `GET /health` | Readiness + parametri caricati (device, soglia, dimensioni, slot registry) | no |
| `POST /infer` | Calcola l'anomaly score **senza** avanzare memoria/vicinato (ammette solo l'entità nel registry) — *passo 1* del flusso anti-poisoning | no (solo admission) |
| `POST /update` | Committa un evento **già approvato** (post-ALLOW di OPA): avanza memoria + storia dei vicini | sì |
| `POST /score` | Score + gate interno + update condizionale (uso senza OPA / test) | sì se benigno |
| `POST /persist` | Riscrive lo stato evoluto su `public/` (anche automatico allo shutdown) | scrive su disco |

### Schema della richiesta (eventi)

`/infer`, `/update`, `/score` accettano lo stesso corpo JSON:

```json
{
  "key_src": "10.0.0.7",          // chiave entità sorgente (string o int)
  "key_dst": "https://crm/db",    // chiave entità destinazione
  "timestamp": 1717000000,         // intero (es. Unix epoch)
  "features": [1.0, 0.0, 0.0, 0.0, 0.0, 2.0], // messaggio d'arco (array di 6 float):
                                              // [0] JA3: 1.0 (ok), 0.0 (anomalia)
                                              // [1] Snort: 0.0 (ok), 1.0 (alert)
                                              // [2-4] Sonde s1, s2, s3 (0.0 o 1.0)
                                              // [5] Metodo HTTP (0=GET, 1=POST, 2=PUT, 3=DELETE, 4=PATCH)
  "src_feat": [/* ... */],         // opz., attributi statici, len == node_feat_dim (16)
  "dst_feat": [/* ... */]          // opz.
}
```

Risposta di `/infer` e `/score`:

```json
{ "anomaly_score": 0.83, "is_anomaly": true, "threshold": 0.6264 }
```

### Mapping del flusso anti-poisoning (gatekeeper OPA)

Lo schema in due step della sezione precedente si realizza così:

1. Orchestrator → `POST /infer` → ottiene `anomaly_score` (sola lettura).
2. Orchestrator → OPA con richiesta + score.
3. Se **ALLOW** → `POST /update` (committa nel modello). Se **DENY** → nessuna chiamata
   a `/update`: l'evento ostile non entra mai nella baseline.

### Gestione Identità (Nuovi Utenti e Guest)

Essendo stato addestrato su dati sintetici, in produzione il modello vedrà solo entità (utenti/IP) mai viste prima. Grazie alla gestione dinamica della memoria, il modello alloca in tempo reale un nuovo slot in RAM per ogni identità sconosciuta (cold-start).

Per questo motivo, l'Orchestrator deve iniettare i privilegi a runtime tramite `src_feat`:

- **Utenti Autenticati (Nuovi nodi)**: L'Orchestrator deve calcolare ruolo e clearance (es. estratti dal JWT) in valori float e passarli in `src_feat`. Il modello li scriverà nello slot appena allocato, e da quel momento saprà applicare le policy corrette per quell'utente.
- **Utenti Guest (Non autenticati)**: Quando la richiesta (es. a `/login` o endpoint pubblici) arriva da un IP senza sessione, la chiave sorgente sarà l'indirizzo IP, e `src_feat` dovrà essere un array di zeri (`[0.0, 0.0, ...]`). Questo corrisponde al livello minimo di privilegi (Clearance=0, Tier=0). Il modello permetterà le chiamate alle rotte pubbliche, ma bloccherà come anomalo qualsiasi tentativo verso endpoint protetti. Appena l'utente farà login, l'Orchestrator comincerà a passare le sue feature reali, "promuovendone" di fatto i privilegi.

### Vincoli operativi

- **Un solo processo/replica.** Il modello è uno stato mutabile in RAM (memoria,
  vicinato, registry); più worker/replica divergerebbero e si sovrascriverebbero in
  `/persist`. Avviare con un singolo worker uvicorn (già impostato) e **non** scalare
  orizzontalmente questo servizio.
- **Continuità delle chiavi.** Il registry serializzato dal training usa le chiavi viste
  in addestramento. Per riconoscere un'entità nota, l'orchestrator deve inviare la
  *stessa* chiave; una chiave nuova viene ammessa dinamicamente e parte "cold-start"
  (si affida a memoria e vicinato man mano che accumula storia approvata).

### Esempio: chiamata diretta (curl)

```bash
# Score read-only di un evento
curl -s -X POST http://localhost:8088/infer \
  -H 'Content-Type: application/json' \
  -d '{"key_src":"10.0.0.7","key_dst":"https://crm/db","timestamp":1717000000,"features":[1.0,0.0,0.0,0.0,0.0,2.0]}'
# -> {"anomaly_score":0.83,"is_anomaly":true,"threshold":0.6264}
```

### Esempio: integrazione dall'orchestrator (Go)

Il flusso anti-poisoning in tre passi (`/infer` → OPA → `/update`) si scrive con la sola
standard library:

```go
type Event struct {
    KeySrc    string    `json:"key_src"`
    KeyDst    string    `json:"key_dst"`
    Timestamp int64     `json:"timestamp"`
    Features  []float64 `json:"features"`
    SrcFeat   []float64 `json:"src_feat,omitempty"`
    DstFeat   []float64 `json:"dst_feat,omitempty"`
}
type ScoreResp struct {
    AnomalyScore float64 `json:"anomaly_score"`
    IsAnomaly    bool    `json:"is_anomaly"`
    Threshold    float64 `json:"threshold"`
}

func post(base, path string, in, out any) error {
    b, _ := json.Marshal(in)
    resp, err := http.Post(base+path, "application/json", bytes.NewReader(b))
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    if resp.StatusCode != http.StatusOK {
        return fmt.Errorf("graphagate %s: status %d", path, resp.StatusCode)
    }
    if out != nil {
        return json.NewDecoder(resp.Body).Decode(out)
    }
    return nil
}

// Per ogni evento di accesso:
ev := Event{KeySrc: srcIP, KeyDst: resURI, Timestamp: time.Now().Unix(),
    Features: edgeSignals, SrcFeat: srcAttrs, DstFeat: dstAttrs}

var s ScoreResp
if err := post(base, "/infer", ev, &s); err != nil { /* fail-closed */ }

allow := opa.Decide(req, s.AnomalyScore)   // OPA è il decisore finale
if allow {
    _ = post(base, "/update", ev, nil)     // committa SOLO se approvato
}
```

> **Fail-closed**: se `/infer` non risponde (timeout, servizio non pronto), trattare
> l'evento come sospetto a livello di policy invece di lasciarlo passare.
