# Integrazione Orchestrator ZTA e Modello TGN

Questo documento chiarisce l'architettura di integrazione tra il Security Orchestrator (che comunica con il Policy Decision Point, es. OPA) e il microservizio AI basato su TGN (Temporal Graph Network).

## Nessun Database Vettoriale Necessario

Una domanda comune nell'integrazione di modelli AI per l'Anomaly Detection strutturale (come i grafi) è se sia necessario mantenere un vector database esterno (es. Milvus, Pinecone) per storicizzare gli embedding o le tuple delle richieste passate.

**La risposta per il TGN è no.**

Il modello TGN è stato progettato appositamente per essere **stateful** e gestire autonomamente la propria memoria temporale in RAM tramite tensori PyTorch. 

### Flusso di Esecuzione (Serving)

1. **Inoltro della Richiesta (Tupla)**
   L'orchestrator ZTA non deve pre-processare vettori né interrogare database storici. Deve semplicemente inoltrare la singola transazione (o evento) grezza all'API di serving del modello (`src/serve_tgn.py -> score_event`). La tupla minima richiesta include:
   - Identificativo sorgente (es. IP, Username).
   - Identificativo destinazione (es. URI della risorsa).
   - Timestamp (es. Unix epoch).
   - Array di feature contestuali (es. trust di JA3, allarmi Snort, metodo HTTP).

2. **Gestione del `NodeRegistry`**
   All'arrivo di una tupla, il TGN utilizza il suo `NodeRegistry` per mappare le chiavi alfanumeriche (es. un nuovo indirizzo IP mai visto prima) in indici interi in tempo reale. Il sistema supporta l'ingresso di nodi non visti durante il training (spazio dei nodi dinamico e illimitato).

3. **Integrazione con la Memoria TGN**
   Il modello accede allo stato storico dei nodi coinvolti leggendo i propri tensori interni (`model.memory`). Viene calcolato l'anomaly score (da `0.0` a `1.0`) e restituito all'orchestrator, che lo girerà ad OPA.

4. **Aggiornamento "Anti-Poisoning" (Gatekeeper OPA)**
   Affinché OPA sia il vero decisore finale, l'Orchestrator non deve usare la funzione generica `score_event`, ma gestire le primitive del modello in due step:
   - Chiama **`infer_score`** per ottenere l'anomaly score (operazione di sola lettura).
   - Invia la richiesta e lo score a OPA.
   - Se **OPA risponde ALLOW** (l'evento è totalmente lecito e non anomalo), l'Orchestrator chiama **`update_memory`** per aggiornare lo stato del modello.
   - Se **OPA risponde DENY**, la chiamata a `update_memory` viene omessa. Questo impedisce in modo assoluto agli attaccanti di fare "poisoning" sul modello, garantendo che il TGN impari solo da ciò che OPA ha esplicitamente approvato.

### Persistenza

L'unico storage richiesto per questo strato AI è il filesystem. Il comando di salvataggio del modello (`save_model`) serializza sul disco l'intero stato:
- I pesi addestrati della rete.
- I tensori in memoria con le cronologie degli accessi (TGN Memory).
- Il dizionario del NodeRegistry.

Questo file (`public/tgn_checkpoint.pt`) assieme ai metadati (`public/tgn_stats.json`) consente al microservizio AI di ripartire esattamente dal punto in cui era stato interrotto senza perdere il contesto storico degli utenti.
