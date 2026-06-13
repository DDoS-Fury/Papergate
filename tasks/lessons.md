# Lessons

## Testare i pesi senza riaddestrare / senza sovrascriverli
- `serve_api` (lifespan) RISALVA lo stato in `public/` allo shutdown (SIGTERM del container). Un test
  live via compose che monta `./public` SOVRASCRIVE quindi i pesi puliti ad ogni stop. **Regola:**
  per i test live montare una COPIA (`cp -r public .test_public` + `docker-compose.test.yml`), mai
  `./public`. Verificare sempre con `sha256sum -c` prima/dopo.
- Per le metriche TGN sintetiche (AUC/AP) sui pesi correnti NON serve riaddestrare-e-salvare: usare
  `train_tgn(save=False)` — ricalcola l'eval sullo stesso stream (stessi dati/seed delle baseline)
  e NON tocca `public/`. Non esiste un entrypoint di solo-eval su checkpoint (la memoria salvata è
  post-training, non ri-replayabile 1:1). `stats.json` persiste solo l'operating-point routed
  (recall/precision/FPR), NON gli AUC/AP → quelli vanno (ri)calcolati.
- `lateral AUC` è single-run-instabile (GPU non-det. ±0.01–0.03): un singolo 0.90 NON sostituisce il
  valore multi-seed pubblicabile (~0.77). Aggiornare la tabella single-run è ok se è coerente
  (tutte le righe single-run, stessi dati), ma marcare il caveat e non riscrivere la narrativa.
- XGBoost NON è installato nell'immagine (`pip install xgboost` a runtime) ed è SUPERVISIONATO →
  upper-bound, non baseline non supervisionata comparabile col TGN.


## Batched offline eval (`_replay`) — fidelity is dataset-dependent
- The batched-TGN "score block against start-of-batch memory, commit updates after" regime is
  exact at `batch_size=1` and is the same regime training uses. BUT on LANL the auth stream is
  **bursty per host** (one source computer fires many auths back-to-back), so a block of 1024
  consecutive events reuses the same nodes heavily ⇒ within-batch memory staleness is **not**
  negligible. Measured: mean|Δscore|~0.5%, Spearman 0.94→0.91 (bs 128→2048), ~1% of benign
  flip across q99, AUC ~−0.02 vs sequential. The drift is ~independent of batch size (you pay
  it as soon as you batch at all).
- Rule: for deployment-faithful / publishable numbers run `eval_batch_size=1`; use batching only
  to iterate (~48× faster on the replay). A stderr warning fires when `eval_batch_size>1`.
- Verifying fidelity: thresholded metrics (AUC/FPR) on a tiny-positive subset are dominated by
  noise (non-monotonic) — measure **score-level** drift (mean/max |Δ|, Spearman, FP-flip) with a
  bs=1-vs-bs=1 noise floor instead. See `tests/verify_lanl_scores.py`.

## tqdm in docker (non-TTY)
- `tqdm.set_postfix(...)` defaults to `refresh=True`, which forces a redraw **every call**,
  bypassing `mininterval` → floods non-TTY docker logs (2218 lines vs 2). Use
  `set_postfix(..., refresh=False)` and let the iterator's throttled `update()` draw it.
- `_pbar` helper: non-TTY ⇒ `mininterval=10s, ascii=True`; TTY ⇒ smooth. Never `disable` —
  watching progress in `docker compose logs` is the point.

## LANL loader windowing
- `load_lanl_stream` window defaults to `[rt_min - window_pad, rt_max + window_pad]` and the gz is
  time-ordered, so with a large `window_pad` you must consume many `max_events` of pre-attack
  benign before reaching red-team (LANL benign volume is huge). For a quick subset with red-team,
  use `window_pad=0` (w_lo = rt_min) so laterals appear immediately. Load is fast (~10s) because
  rt_min is early in the file. Red-team is bursty: a chronological val slice can have 0 laterals
  (calibration then falls back to the conservative threshold — handled, deterministic).
