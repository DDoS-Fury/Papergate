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

### Fase 3 — Neighbor loader (in-memory, no graph DB) — ⏸️ IN ATTESA
Bloccante di design emerso: lo store messaggi per-edge (per `e_id`) cresce illimitato in
streaming long-running. Serve decisione su store a dimensione fissa (ring-buffer ~`num_nodes×K`).
- [ ] (da decidere con l'utente prima di implementare)

### Fase 4 — Rigore di valutazione
- [x] `stream_synthetic.py`: emette vettore `types` (0=benign,1=policy,2=context), 7° valore
- [x] `train_tgn.py`: metriche AUC/AP/recall per-tipo (policy vs contextual)
- [x] `train_tgn.py`: baseline a regole + recall su anomalie di policy

### Fase 5 — Verifica
- [x] Riproducibilità: due run del generator con seed → tensori identici (smoke `repro: True`)
- [x] Smoke: forward con feature statiche + `score_event` dinamico OK
- [x] `docker compose --profile training-tgn` completa + nuove metriche per-tipo
- [x] `docker compose --profile verify-tgn` → 4/4 check PASS (nessuna regressione)

## Review

Fasi 1, 2, 4 implementate e verificate end-to-end via Docker.

Metriche test (training completo, seed=42):
- Overall: AUC 0.9401 | AP 0.8070 | (thr 0.7005) Precision 0.278 Recall 0.863
- Policy:     AUC 0.8849 | Recall@thr 0.7322   (n=239)
- Contextual: AUC 0.9947 | Recall@thr 0.9917   (n=242)
- Baseline a regole: recall su policy = 0.0000 → conferma il valore aggiunto delle
  feature statiche (Fase 2): il modello rileva le violazioni di policy che una baseline
  a sole regole non vede.

Verify serving: 4/4 PASS (reload determinism, benign update, anti-poisoning, dynamic node).

Nessuna regressione: shared train/serve path, gate anti-poisoning, predict-then-update,
split cronologico, semantica binaria di `y` tutti invariati.

Fase 3 (neighbor loader): NON implementata — vedi blocco design sopra, in attesa di decisione.
Sotto-punti opzionali (flip negative contestuale, dati più difficili / lateral movement):
non implementati, richiedono ok esplicito.
</content>
