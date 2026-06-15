# TODO — Verifica metriche report.tex (pesi vergini) + re-run ablation + LANL faithful

## Contesto
Verifica empirica delle metriche del report (sospette contraddizioni) con pesi VERGINI per ogni
test (tutti i run `train_tgn(save=False)` → ri-addestra da zero, `public/` MAI toccato).
Guard sha256 di `public/` invariato inizio→fine.

## Run eseguiti (2026-06-15)
- [x] **Run B — sintetico seed 42** (`train_tgn(save=False)`, 200k/15ep/bs=1, solo `src` montato):
      agg AUC 0.952 | AP 0.896 | agg-recall 70.9% | lat-AUC 0.894 | lat-AP 0.464 |
      lat-rec@1%FPR(before) 0.141 @FPR 1.3% | routed(after) 0.255 @FPR 4.8%.
      → Tabella baselines (riga TGN 0.947/0.870/0.900/0.130) CONFERMATA entro rumore single-run.
- [x] **Run A — ablation multi-seed** (`--profile ablations`, seeds [42,7,123], 40k/12ep):
      full lat-AUC **0.882±0.012** | agg-AUC 0.947±0.004 (= tabella, esatto).
      Δ lateral AUC vs full: hist **+0.163** | hashed-id **+0.046** | precursor **+0.013** |
      struct head **+0.007**.
- [x] **Run C — LANL faithful** (`eval_lanl.py`, window 500000–1157488, stride 360, 200k ev,
      265 laterali, picco red-team giorno 7, 2ep, train .25/val .15, **bs=1**):
      agg/lat AUC **0.776** | lat-rec 15.3% @FPR 2.5% (soglia globale) | routing collassato
      (val 3 laterali → soglia 0.9992 → recall 0%) | cold-start: warmed AUC 0.74 (151) / cold (110).

## Contraddizioni trovate e risolte nel report
1. **lateral AUC "~0.77 multi-seed"** → FALSO. Reale 0.882±0.012 (3 seed), coerente col 0.90
   single-run. → corretto ovunque (cap.2 caption tab, §4.3, §5.2). [DECISIONE UTENTE: correggi]
2. **Δ ablation**: hist +0.066→**+0.163**; precursor +0.073→**+0.013** (DEMOTO a marginale, come
   struct head); struct 0.778/0.770→0.882/0.875. → corretto + precursore declassato. [UTENTE: ok]
3. **LANL 0.8824/73%/2.18%** = full-span BATCHATO (inaffidabile). Faithful focused = 0.776/15.3%.
   → **RIMOSSA tutta la sezione LANL** dal report (titolo §4.3 → "Cost-sensitive routing";
   eliminati i 2 paragrafi LANL+SOTA). [DECISIONE UTENTE: elimina, forse dataset non adatto]

## Non toccato (entro rumore / non richiesto)
- Tabella baselines sintetica (0.947/0.870/0.900/0.130): confermata multi-seed, lasciata.
- Numeri routing sintetico §4.3 (13.0→28.6%, FPR 2.3→6.0%, agg 70.3%): single-run come Run B
  (14.1→25.5%, 1.3→4.8%, 70.9%), entro rumore; non flaggati dall'utente → lasciati.

## Verifica finale
- [x] `public/` sha256 invariato (guard /tmp/public_guard.sha256, `sha256sum -c` OK).
- [x] Nessun riferimento LANL/0.8824/73%/2.18/SOTA residuo nel report.
- [x] Ogni numero scritto tracciabile a un output di run (B/A/C) loggato.
- [~] pdflatex non installato sull'host: struttura/tabella verificate a mano (5 col invariate).
