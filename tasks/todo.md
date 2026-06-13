# TODO — Verifica test_client + run live/baseline + aggiornamento report

## Contesto
- Pesi PULITI già riaddestrati (public/tgn_checkpoint.pt @ 2026-06-13 08:47). NON riaddestrare.
- Nessun path di eval-only sul checkpoint ⇒ AUC/AP TGN sintetiche in tab:baselines restano
  (ultimo training pulito, riproducibili ±0.01 GPU-noise). Si rinfrescano: colonne baseline,
  operating-point routed (da stats.json pulito), numeri live (latenza/recall operativo).

## 1. Verifica/fix test_client
- [x] Tracciata pipeline /infer→/update vs score_event/commit_event: client FEDELE al design
      (predict-then-update, grace=5, OPA su etype==1, dual-threshold API-side, precursor auto-armato).
- [x] Cleanup: rimossa riga morta `anomaly_score = resp_data.get(...)` (inutilizzata).

## 2. Run live test_client (protezione pesi OBBLIGATORIA)
- [x] Copia public/ → .test_public; montata su serve-tgn via docker-compose.test.yml (shutdown-save → copia, non public).
- [x] Run STANDARD (with-device): 15361 ev | P50/P90/P99 = 5.99/6.22/6.71 ms | VRAM 2441 MB | lateral 47.4% | benign 76.3%.
- [x] Run NO-DEVICE (ablation): 20776 ev | P50/P99 = 4.43/4.99 ms | lateral 44.6% | benign 77.7%.
- [x] public/ invariato (sha256 OK prima/durante/dopo).

## 3. Run baseline (tutte e 4, stessi dati/tipo via TGNConfig seed=42, 200k)
- [x] iforest: AUC 0.775 AP 0.628 | lat AUC 0.639 rec 0.010
- [x] gnn:     AUC 0.574 AP 0.569 | lat AUC 0.468 rec 0.139
- [x] ocsvm:   AUC 0.845 AP 0.793 | lat AUC 0.632 rec 0.047
- [x] xgboost (SUPERVISED): AUC 0.977 | lat AUC 0.942 rec 0.346  (pip install a runtime; non in immagine)
- [x] TGN coerente via train_tgn(save=False): AUC 0.947 AP 0.870 | lat AUC 0.900 | routed lat-rec 0.286 (FPR 0.060) agg-rec 0.703.

## 4. Aggiornato report.tex
- [x] tab:baselines: 4 colonne unsupervised (TGN/GNN/OCSVM/IForest), righe AUC/AP/lat-AUC/lat-Recall@1%FPR freschi; caption + caveat single-run; XGBoost come upper-bound supervisionato in nota.
- [x] §intro tipi + §4 itemize: aggiunte OCSVM (col) e XGBoost (nota); GNN lat-AUC 0.486→0.468.
- [x] §4.3 + §5.2: routing lat-recall 13.0%→28.6% (FPR 2.3%→6.0%), agg-rec 70.3%, lat-AUC 0.90 (caveat multi-seed ~0.77).
- [x] §4.2 clean_fpr_cap: AUC 0.76→0.90 + tradeoff curve.
- [x] §Deployment: latenza P50/P99 6.0/6.7 ms (with-dev) e 4.4/5.0 (no-dev), VRAM ~2.4 GB, nota live cold-start.

## 5. Verifica finale
- [x] public/ invariato (sha256 OK). .test_public rimossa + gitignored. Container test rimossi.
- [x] docker-compose.test.yml MANTENUTO (harness weight-safe riusabile) + documentato prereq `cp -r public .test_public`.
- [~] LaTeX compile: pdflatex non installato sull'host; tabella verificata a mano (5 col, 4 `&`/riga).

## Review
- test_client già corretto rispetto al modello: unica modifica una riga morta. Nessun bug funzionale.
- DECISIONE UTENTE: niente riscrittura aggressiva della narrativa lateral (0.76→0.90); tabella single-run + caveat multi-seed.
- I numeri TGN del report ora vengono da train_tgn(save=False) → coerenti con le baseline (stessi dati 200k/seed42), pesi puliti NON toccati.
- XGBoost è supervisionato → upper-bound, non baseline comparabile (scelta utente: nota separata).
- Caveat aperto: lateral AUC 0.90 è single-run; per pubblicazione confermare multi-seed [42,7,123] (lessons.md).
