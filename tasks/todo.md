# TODO — batching GPU per `eval-lanl` + progress bar

Piano completo: vedi piano approvato (batching del path offline `_replay`, produzione intatta).

## Implementazione
- [x] `src/config.py`: aggiungere `eval_batch_size: int = 1` a `TGNConfig`.
- [x] `src/train_tgn.py`: helper `_pbar()` (tqdm TTY-aware, ok per docker logs).
- [x] `src/train_tgn.py`: riscrivere `_replay()` in versione batchata (param `batch_size`, `desc`):
      - Fase 1 scoring batchato (riusa pattern training, gruppi-arco a livello dataset)
      - Fase 2 feedback sequenziale O(1) (precursor/trust/decisione) — esatto a ogni batch
      - Fase 3 update memoria batchato + counters in loop (ordine serving src→dev→u→d)
- [x] `src/train_tgn.py`: progress bar nel loop di training; passare `cfg.eval_batch_size` alle 2 chiamate `_replay`.
- [x] `tests/eval_lanl.py`: arg `--eval-batch-size` (default 1024) → `dataclasses.replace`.
- [x] `docker-compose.yml`: aggiungere `--eval-batch-size 1024` al profilo `eval-lanl`.
- [x] `requirements.txt` + `pyproject.toml`: aggiungere `tqdm`.

## Verifica
- [x] Test parità: `_replay(bs=1)` ≡ vecchio path per-evento → max|Δ|≈1.5e-7 (PASS), su sintetico (3 archi).
      Drift sintetico bs 64/256 alto (mean ~0.02) MA è il caso peggiore (poche entità, riuso massivo) e il
      sintetico gira comunque a bs=1 di default.
- [x] Drift LANL (subset reale, 1 arco): replay 7m16s→9s (~48×); pipeline 18×. lateral-recall preservata, AUC ~−0.02.
- [x] Fedeltà a livello punteggio (isolata): rumore GPU ~0; drift batchato monotòno ma ~indipendente dalla taglia
      (mean|Δ|~0.5%, Spearman 0.94→0.91, ~1% FP-flip). LANL auth è bursty ⇒ riuso intra-batch non trascurabile.
- [x] Progress bar: replay pulita (47 refresh); training corretta da 2218→2 refresh (set_postfix refresh=False).
- [x] Nessuna regressione eval sintetica (default `eval_batch_size=1` ⇒ path bit-identico, provato dalla parità).
- [x] Warning automatico su stderr quando `eval_batch_size>1` (i punteggi sono approssimati, non per numeri finali).

## Review
- Implementato batching del solo path offline `_replay`; serving (`serve_tgn`/`serve_api`) intatto e sequenziale.
- Parità esatta a bs=1 (Δ≈1.5e-7) ⇒ refactor senza cambio di semantica; default config `eval_batch_size=1` (fail-safe).
- CLI/compose default 1024 per iterazione veloce; warning protegge i numeri da pubblicare.
- DECISIONE APERTA (utente): il drift NON è strettamente trascurabile su LANL (bursty). Per paper/SOTA usare bs=1
  per i numeri finali; batching solo per sviluppo. Vedi lessons.md.
- File di verifica: `tests/verify_replay_batching.py` (parità, sintetico, permanente),
  `tests/verify_lanl_drift.py` e `tests/verify_lanl_scores.py` (tool LANL, richiedono il dataset).
