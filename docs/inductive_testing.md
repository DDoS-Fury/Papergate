# Inductive Testing & Lateral Movement Validation

Questo documento riassume le sfide e le soluzioni architetturali affrontate per abilitare e validare il rilevamento dei **movimenti laterali** tramite il Temporal Graph Network (TGN), in particolare analizzando il concetto di *induttività* e *data poisoning* durante le fasi di testing.

## Potenziamento Strutturale (Deep GNN + MLP)
Inizialmente, il modello faticava a rilevare i movimenti laterali (Recall ~11%). Questo tipo di attacco (un accesso formalmente autorizzato ma "non abituale") richiede una profonda comprensione della geometria e della storia del grafo. Per sbloccare questa capacità:
1. **Rete più profonda:** Abbiamo aumentato i salti topologici (`num_hops` da 2 a 3) nella Graph Attention Network.
2. **Geometria complessa:** La proiezione lineare per la similarità strutturale è stata sostituita da un *Multi-Layer Perceptron (MLP)*, permettendo alla rete di calcolare spazi vettoriali non lineari.
3. **Loss Penalties:** Il peso degli *Hard Negatives* (falsi movimenti laterali generati durante l'addestramento) è stato raddoppiato per "forzare" l'apprendimento su questo specifico pattern.

Questo ha innalzato le performance a livello teorico (durante l'addestramento) fino al 40-50% di recall.

## Il Paradosso dell'Induttività nei Test (Streaming)
Durante i test in streaming (`test_client.py`), i risultati inizialmente non combaciavano con la validazione offline (crollando al 16-22% di recall). 

Il motivo risiedeva in un malinteso sul concetto di **induttività**:
* Il TGN è induttivo: alloca dinamicamente memoria per nodi mai visti prima senza andare in crash.
* Tuttavia, per rilevare un'anomalia *strutturale*, la rete ha un disperato bisogno di un **Temporal Neighborhood** (una cronologia storica del nodo).

Nel test client originario, la funzione `event_generator()` non utilizzava un seed. Questo generava una disconnessione (ground truth sfasata): il test client creava identità, ruoli e abitudini casuali per ogni utente, ma li inviava all'API sotto mentite spoglie (utilizzando ID discordanti dal training, e comportamenti casuali). 
L'API, ricevendoli, doveva azzerare le loro cronologie, trattando il 100% degli eventi come "Utenti nati al momento". Essendo utenti senza storia pregressa, la testa strutturale della rete non poteva rilevare "deviazioni dalle abitudini", poiché le abitudini stesse erano vuote!
Sincronizzando il `seed` e passando gli ID interi originali, l'universo di test si è allineato all'universo appreso, permettendo al modello di sfruttare il grafo storico reale.

## Mitigazione del Data Poisoning (Cold Start)
Un altro problema rilevato nel test era il *Grace Period*. Durante i primissimi eventi di un nuovo utente (cold start), la rete non ha dati per valutarlo e l'Orchestrator si fida ciecamente delle regole OPA (che lasciano passare i movimenti laterali).
Se il Grace Period è troppo ampio (es. 50 eventi), eventuali movimenti laterali dell'attaccante vengono assorbiti dal modello e catalogati come "nuova abitudine" (Data Poisoning). 
Riducendo il Grace Period a **5 eventi**, l'AI interviene rapidamente non appena ha sufficienti interazioni per farsi un'idea della baseline dell'utente, minimizzando la finestra di vulnerabilità.

## Risultati Reali in Streaming
Applicando le modifiche architetturali e i fix al testing environment, il sistema unsupervised in streaming ha ottenuto risultati eccezionali in un ambiente Zero Trust (tempi di decisione P50 ~ 7ms):

* **Accuratezza globale:** 94.64%
* **Precisione complessiva (Anomalie):** 45.85%
* **Recall Anomalie di Contesto:** 97.12%
* **Recall Anomalie di Policy:** 80.17%
* **Recall Movimenti Laterali:** 39.86% (quasi quadruplicato rispetto all'architettura base)
