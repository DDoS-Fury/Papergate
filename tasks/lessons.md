# Lessons

## Validazione mirata cred-theft + sweep architetturale v4 — 2026-06-18 (multi-seed [42,7,123])
- **cred-theft/wiped-cookie n=0 nel test è un artefatto di split, NON un bug del modello.**
  `num_theft_slots=64`/`p_cred_theft=0.0012` → tutti gli incidenti partono entro ~ev.53k
  (dentro il 70% train). Per misurarli serve uno stream theft-rich SOLO per l'eval
  (`run_config_eval.py`: slot 400/120, p_theft=0.002, p_wipe=0.0008) → porta n=452 theft
  nel test. NON cambiare i default in config.py (deployable invariato): usare `save=False`.
- **Contributo del nodo config (v4 vs ablazione `use_config_node=False`, ≈v3, a parità di
  dati):** theft recall 0.206→0.314 (+0.108), theft AUC 0.676→0.729 (+0.053, ma std±0.088
  → segnale robusto = recall), lateral recall 0.334→0.469 (+0.135). Lieve ↑ FPR cookie-wipe
  (0.038→0.060). Il `config→user` è il binding che discrimina il furto credenziali.
- **Capacità NON è il collo di bottiglia del laterale.** Sweep 40k/12ep ×3 seed: +1 layer MLP
  neutro (AUC +0.002 entro rumore, recall ↓); memory_dim=384 e gnn_heads=8 PEGGIORANO
  (AUC −0.024/−0.033) e aumentano la varianza (heads: recall ±0.161). Coerente con
  l'ablation: il laterale è signal-bound (history feats +0.163), non parametri. Regola:
  prima di proporre più capacità, ricontrollare che il limite non sia il segnale.
- I knob `gnn_heads`/`link_pred_hidden_layers` sono in hp del checkpoint con default
  (4, 2) = architettura storica; `LinkPredictor.lin_extra` è una ModuleList VUOTA a
  default → state_dict dei checkpoint v4 esistenti carica senza modifiche (back-compat).


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
- **LANL red-team is front-sparse, mid-dense.** rt spans t=150885→2557047 (749 events). First 200k
  events from rt_min hold only ~10 laterals. The densest region is a 5-day window [725488–1157488]
  (503 laterals, 67%), peaking day 7 [755685–777285] (157 in 6h). For a focused ~200k-event eval
  with the day-7 peak in TEST: `window=(500000,1157488)`, `benign_stride=360`, `max_events=200000`
  → 200k ev, 265 laterals, 3.75 d, first lateral at idx≈0.25. Use `train_frac=0.25 val_frac=0.15`
  (val gets 3 laterals, test 261). LANL benign density ≈57 ev/s raw → at stride 16, 200k ev spans
  only ~15h (won't reach a burst): raise stride to cover more time.

## Stale ablation numbers in report — RE-VERIFIED 2026-06-15 (multi-seed [42,7,123], 40k/12ep)
- full lateral AUC = **0.882±0.012** (NOT the report's "~0.77 multi-seed" — that was stale and
  *undersold* the model; single-run 0.90 and multi-seed 0.882 actually agree).
- Δ lateral AUC vs full: **hist feats +0.163** (dominant), hashed-id +0.046, **precursor +0.013**
  (marginal!), struct head +0.007 (marginal). Report claimed hist +0.066 / precursor +0.073 —
  both wrong; precursor is now marginal like the struct head (hist absorbed its role).
- agg AUC full = 0.947±0.004 → matches the published table exactly.
- Rule: ablation deltas drift as the model/data evolve (de-circularization, benign_explore_prob).
  Re-run `--profile ablations` before quoting any +ΔAUC in the report; don't trust prior prose.

## LANL faithful (bs=1) ≪ batched full-span headline
- Focused window above, 2 epochs, bs=1: agg/lat AUC **0.776**, lateral recall **15.3% @2.5%FPR**
  (global threshold); cost-sensitive routing collapses (val 3 laterals → threshold≈1.0 → 0% recall).
  The report's 0.8824 / >73% / 2.18% were a **batched full-span** run (cfr. batched-eval lesson)
  — not reproducible faithfully on a focused window. User removed the LANL section entirely
  (possibly not the right dataset for a streaming O(1) model).
