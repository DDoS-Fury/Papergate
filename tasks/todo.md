# Task: Miglioramenti TGN Graphagate (valutazione critica → implementazione)

Piano completo: `C:\Users\Gabs\.claude\plans\obiettivo-valuta-in-modo-graceful-valiant.md`

## Vincoli
- Nessuna regressione. Preservare: shared train/serve path, gate anti-poisoning,
  predict-then-update, split cronologico, semantica binaria di `y`.
- Sotto-punti *opzionali* (flip negative contestuale, dati più difficili) NON vanno
  toccati senza ok esplicito separato.

## Status: 🛠️ Implementazione (Fasi 1/2/4 fatte; Fase 3 in attesa di decisione)

### Fase 1 — Riproducibilità e igiene
- [x] `train_tgn.py`: seed del modulo `random` (`random.seed(cfg.seed)`)
- [x] `stream_synthetic.generate_streaming_data`: param `seed` opzionale che seedi random/numpy

### Fase 2 — Feature statiche dei nodi (real-time feasible)
- [x] `tgn.py`: buffer `node_feat [num_nodes, node_feat_dim]` + LinkPredictor ampliato + forward
- [x] `train_tgn.py`: popolare `model.node_feat[:total_nodes]` da `node_features`
- [x] `serve_tgn.py`: `score_event` con `src_feat`/`dst_feat` opzionali + `_reset_slot` azzera node_feat
- [x] `verify_tgn.py`: compatibile via default (nessuna modifica necessaria)

### Fase 3 — Neighbor loader (in-memory, no graph DB, bounded) + scorer strutturale — ✅ FATTA
Design bounded: `MessageNeighborLoader` (ring-buffer `[num_nodes, size]` per
neighbors/e_id/last_t/last_msg), O(num_nodes·K·msg_dim), nessun DB.
- [x] `src/model/neighbor.py`: `MessageNeighborLoader` (mirror esatto di PyG 2.7.0
      `LastNeighborLoader`, verificato via introspezione) con `insert`/`__call__`/
      `reset_state`/`reset_node`/`state`/`load_state`
- [x] `tgn.py`: `embed()` (contesto vicinato) + `init_neighbor_loader()`
- [x] `train_tgn.py`: loop usa loader (reset per epoca, embed unico, insert benigni);
      4ª head di negativi *hard non-abituali* (risorsa fuori dal vicinato di src)
- [x] `serve_tgn.py`: `infer_score`/`update_memory` via loader; `build_model` crea+`eval()`;
      `save_model`/`load_model` persistono lo stato del loader; `_reset_slot` → `reset_node`
- [x] `config.py`: `neighbor_size=10`; `epochs` 3→10
- [x] **Bug pre-esistente trovato e corretto**: doppio flush del message store al reload
      (`model.eval()` dopo `load_state_dict`) → `build_model` ora fa `eval()` prima
- [x] **Redesign scorer** (approvato): identità di nodo apprendibile (`nn.Embedding`) data
      in input alla GNN + head di compatibilità strutturale dedicata (cosine·temperatura
      apprendibile) sommata alla head a feature. `model.score()` condivisa train/serve.
- [x] Verifica: training exit 0, `verify_tgn` 4/4 PASS, reload determinism *genuino*,
      **lateral movement ora rilevato (AUC 0.90 vs 0.46)**.

### Fase 4 — Rigore di valutazione
- [x] `stream_synthetic.py`: emette vettore `types` (0=benign,1=policy,2=context), 7° valore
- [x] `train_tgn.py`: metriche AUC/AP/recall per-tipo (policy vs contextual)
- [x] `train_tgn.py`: baseline a regole + recall su anomalie di policy

### Fase 6 — API di inferenza HTTP (servizio per orchestrator ZTA in Go) — ✅ FATTA
REST/JSON (FastAPI+uvicorn); `/infer`+`/update` separati per il flusso anti-poisoning
con OPA, più `/score` combinato; save-back su disco. Lo strato server chiama solo le
primitive di `serve_tgn.py` (single source of truth). Stato mutabile in RAM → un solo
worker + lock.
- [x] `serve_tgn.py`: `load_model` → 4-tupla `(model, registry, threshold, hp)`;
      nuovo `commit_event` (commit incondizionato post-ALLOW, riusa `_reset_slot`/
      `_set_node_features`/`update_memory`)
- [x] `serve_api.py` (NEW): FastAPI con lifespan (load all'avvio, save-back allo
      shutdown), lock di modulo, schemi Pydantic con validazione dimensioni;
      `GET /health`, `POST /infer|/update|/score|/persist`; `main()` → uvicorn workers=1
- [x] `verify_tgn.py`: aggiornato l'unico unpack di `load_model`
- [x] `pyproject.toml`: aggiunte `fastapi`, `uvicorn[standard]`
- [x] `docker-compose.yml`: profilo/servizio `serve-tgn` (porta 8088, healthcheck Python)
- [x] Docs: `orchestrator_integration.md` (API HTTP + flusso OPA + vincoli),
      `docker.md` (profilo serve-tgn), `README.md` (servizio + esempio submodule + layout)
- [x] Verifica via Docker: `verify-tgn` **4/4 PASS** (no regression); `serve-tgn` healthy
      in ~8s; smoke endpoint OK (/health; /infer ×2 score identico = read-only; /score
      anomalo is_anomaly=true; /update ok; dimensioni errate → 422; /persist scrive gli
      artifact); save-back verificato (`last_update[60]=1e9` persistito nel checkpoint)

### Fase 5 — Verifica
- [x] Riproducibilità: due run del generator con seed → tensori identici (smoke `repro: True`)
- [x] Smoke: forward con feature statiche + `score_event` dinamico OK
- [x] `docker compose --profile training-tgn` completa + nuove metriche per-tipo
- [x] `docker compose --profile verify-tgn` → 4/4 check PASS (nessuna regressione)

## Review

Tutte le fasi (1–4) implementate e verificate end-to-end via Docker.

Metriche test finali (training completo, seed=42, loader + identità + head strutturale, 10 epoche):
- Overall:    AUC 0.9690 | AP 0.8583 | (thr 0.6264) Precision 0.307 Recall 0.915
- Policy:     AUC 0.9988 | AP 0.9589 | Recall@thr 1.0000   (n=174)
- Contextual: AUC 0.9999 | AP 0.9969 | Recall@thr 1.0000   (n=171)
- Lateral:    AUC 0.8981 | AP 0.2417 | Recall@thr 0.7162   (n=148)  ← NUOVO, prima ≈ caso (0.46)
- Baseline a regole: recall policy = 0.0000, recall lateral = 0.0000 → il modello rileva
  proprio ciò che le sole regole non vedono (violazioni di policy e lateral movement).

Progressione del lateral movement durante la Fase 3 (diagnosi → fix):
1. loader cablato + negativi facili .............. AUC 0.49 (caso)
2. + negativi hard (risorsa casuale) + 10 epoche . AUC 0.53 (collisione con abituali)
3. + negativi hard *non-abituali* ................ AUC 0.46 (blocco rappresentazionale)
   → diagnostico: scorer concat-MLP ignora la compatibilità src↔dst; risorse senza identità
4. + identità di nodo nella GNN + head strutturale  AUC 0.90 ✅

Verify serving: 4/4 PASS (reload determinism *genuino*, benign update, anti-poisoning, dynamic node).

Bug pre-esistente corretto: doppio flush del message store di `TGNMemory` al reload
(`model.eval()` ri-applicava lo store sopra il buffer già completo) — mascherato dal
vecchio check di determinismo che confrontava due reload tra loro.

Nessuna regressione: shared train/serve path, gate anti-poisoning, predict-then-update,
split cronologico, semantica binaria di `y` tutti invariati. Le Fasi 2/3 cambiano il
formato del checkpoint → richiedono retraining (la pipeline lo fa già).
