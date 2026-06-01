# Piano di Implementazione: Modello Induttivo Puro (TGN)

Questo documento descrive il piano di implementazione passo-passo per convertire il modello TGN in un "Modello Induttivo Puro", al fine di risolvere il problema dell'embedding transduttivo (Transductive Embedding Problem).

## Passo 1: Modifiche a `src/model/tgn.py`
- **Rimozione di `self.node_id`:** Rimuovere il modulo `nn.Embedding` precedentemente utilizzato per generare gli embedding transduttivi basati sugli ID dei nodi.
- **Aggiornamento della GNN:** Modificare la rete neurale su grafo (GNN) in modo da concatenare le feature iniziali dei nodi (`self.node_feat[n_id]`) direttamente con lo stato della memoria `z`, eliminando l'uso di `id_emb`.
- **Aggiornamento di `in_channels`:** Ricalcolare e aggiornare la dimensione dei canali di input (`in_channels`) per la GNN e i relativi moduli di aggiornamento per riflettere la rimozione di `id_dim`.

## Passo 2: Modifiche a `src/config.py`
- **Rimozione di `id_dim`:** Rimuovere il parametro `id_dim` dal file di configurazione, poiché non sarà più necessario configurare la dimensione per l'embedding degli ID dei nodi.

## Passo 3: Modifiche a `src/data/stream_synthetic.py`
- **Problema delle Feature delle Risorse:** Attualmente, le risorse vengono create con `node_feat` composti interamente da zeri. Senza l'embedding degli ID, la GNN non sarebbe in grado di distinguere una risorsa dall'altra.
- **Soluzione Induttiva:** Proponiamo di codificare il tipo di risorsa (es. usando il one-hot encoding o un indice categorico) direttamente nell'array `node_features`. In questo modo, il modello potrà distinguere la tipologia e le caratteristiche delle risorse puramente attraverso le feature, in modo induttivo.

## Passo 4: Modifiche a `src/train_tgn.py` e `src/serve_api.py`
- **Propagazione delle Modifiche:** Assicurarsi che la rimozione del parametro `id_dim` e dei relativi riferimenti venga gestita in entrambi gli script principali.
- **Aggiornamento dell'Inizializzazione:** Correggere l'istanziazione del modello, eliminando qualsiasi parametro ridondante e assicurandosi che i pesi della rete riflettano la nuova dimensione determinata solo dai feature originari.

---

**Azione Richiesta:** È richiesta l'approvazione esplicita da parte dell'utente prima di procedere con l'implementazione pratica. Si prega di confermare se si desidera avviare le modifiche al codice.
