# Valutazione snella v5 — runbook per la workstation

Confronto **TGN vs Isolation Forest vs regole stateful** sul generatore v5, più l'ablation del nodo
configurazione (E2) e la curva del budget di dati (E3). Stato al 2026-09-21: codice pronto e testato in
unità; Panel A e B provati con uno smoke su CPU, **Panel C mai eseguito end-to-end**; nessun run su GPU.
Le cifre qui sotto sono stime (dal log v4: ≈15 min per run del TGN, dominati dai replay di validation e
test), non misure su v5.

Orchestratore: `tests/regen_report_tables.py`. Scelte di disegno e motivazioni: sezione finale e
`tasks/todo.md` (ignorato da git).

## Prima di cominciare

- `git pull` su `fix/generator`. Nessun rebuild dell'immagine: il servizio `regen-report` monta
  `src/`, `tests/`, `tasks/` e `docs/`.
- **Non lanciare questi job sul Mac** (niente MPS per il TGN, e satura la macchina): solo sulla workstation.
- Ordine e gate: **smoke → E1 (A) → decisione → E2 (B) → E3 (C)**. B e C rileggono `tasks/runs/panelA.json`
  e falliscono con un errore esplicito se manca o se i seed non coincidono.

## 1. Smoke (≈10 min) — soprattutto per Panel C

```bash
docker compose run --rm regen-report /app/tests/regen_report_tables.py \
  --panels ABC --seeds 2000 --budget-seeds 2000 --budgets 2000 5000 \
  --events 20000 --epochs 1 --out-dir /tmp/smoke
```

- Passa se termina con `DONE_REGEN_REPORT_TABLES` e nessun traceback. Le cifre sono prive di significato
  (1 epoca, 20k eventi, 1 seed): con un seed il verdetto E1 esce `False` e quello di adattamento
  `undetermined`, ed è normale.
- Scrive in `/tmp/smoke` dentro il container: non tocca `tasks/runs/`. Con `--events`/`--epochs` senza
  `--out-dir` lo script rifiuta di partire, proprio per non sovrascrivere i JSON veri.
- Se Panel C dà errore, mandare il traceback.

## 2. E1 — Panel A (≈2,5 h)

```bash
docker compose --profile regen-report up 2>&1 | tee tasks/runs/lean_panelA.log
```

Equivale a `docker compose run --rm regen-report /app/tests/regen_report_tables.py --panels A`: 10 seed
(1000–1009), TGN + Isolation Forest + tre varianti delle regole. Scrive `tasks/runs/panelA.json` e, in
`docs/paper/generated/`, `tab_baselines.tex` e `tab_paired.tex`.

Da leggere in `panelA.json`: `verdict_tgn_beats` (un booleano per comparatore) e `paired` (differenze
appaiate per seed, con IC95 e p di Wilcoxon).

## 3. Gate

Vedi "Criteri di decisione". Se E1 **non** regge contro `lookup_rules`, B e C non servono a difendere il
claim: si riposiziona il paper. Se regge, si prosegue.

## 4. E2 — Panel B, nodo config spento (≈2,5 h)

```bash
docker compose run --rm regen-report /app/tests/regen_report_tables.py --panels B \
  2>&1 | tee tasks/runs/lean_panelB.log
```

Il braccio "on" è il run di Panel A (riletto da `panelA.json`, con un assert sull'uguaglianza delle
configurazioni): si addestra solo il braccio "off". Scrive `tasks/runs/panelB.json` e `tab_v3v4.tex`.

## 5. E3 — Panel C, curva del budget di dati (≈3 h)

```bash
docker compose run --rm regen-report /app/tests/regen_report_tables.py --panels C \
  2>&1 | tee tasks/runs/lean_panelC.log
```

Default già pre-registrati: seed 1000–1004, budget 10k / 25k / 50k / 100k eventi di training (il budget
pieno, 140k, è il run di Panel A). Scrive `tasks/runs/panelC.json` con `n_star` per metodo e `adaptation`
(verdetto).

Se `n_star` risulta `undetermined` (IC troppo largo con 5 seed), estendere a 10 seed (≈ altre 3 h; i seed
1005–1009 sono già in `panelA.json`):

```bash
docker compose run --rm regen-report /app/tests/regen_report_tables.py --panels C \
  --budget-seeds 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 \
  2>&1 | tee tasks/runs/lean_panelC_10seeds.log
```

## 6. Cosa riportare e committare

`tasks/runs/panelA.json`, `panelB.json`, `panelC.json` e i `lean_*.log`. La cache
(`tasks/runs/cache/`) è ignorata da git ed è solo un lavoro in corso.

## Se qualcosa va storto

- **Un run cade a metà:** rilanciare lo stesso comando. Ogni run finito è nella cache e viene riusato
  (`[cache] ...` nel log). Dopo aver cambiato il codice cancellare `tasks/runs/cache/` oppure passare
  `--fresh`.
- **`panelA.json was not produced with seeds=...`:** B/C sono stati lanciati con seed o override diversi
  da quelli di A. Rifare A o allineare `--seeds` / `--budget-seeds`.
- **Solo un sottoinsieme di baseline:** `--baselines iforest lookup_rules` (Panel C usa comunque solo questi
  due).

## Criteri di decisione (pre-registrati, scritti prima di qualsiasi run del TGN su v5)

Implementati in `tests/regen_report_tables.py` (`_paired_vs_tgn`, `_verdict`, `_n_star`,
`_adaptation_verdict`) e fissati da `tests/test_regen_report.py`; non cambiarli dopo aver visto i dati.

- **Seed:** 1000–1009 (E1, E2), 1000–1004 (E3). Il generatore v5 è stato tarato guardando i seed
  1–9/42/7/123: per questo i numeri finali usano seed nuovi. Smoke e sviluppo: 2000 e oltre.
- **Metriche primarie, per classe (lateral, cred-theft):** AUC e AP. La recall a 1 % di FPR è secondaria:
  per le regole, con punteggi interi, è alla più piccola soglia intera con FPR benigno di validation ≤ 1 %.
- **E1:** "il TGN batte un comparatore" solo se, sull'**AUC** di lateral **e** di theft, il test di
  Wilcoxon appaiato è significativo **e** l'IC95 bootstrap della differenza media esclude 0 verso l'alto.
  Comparatore principale: `lookup_rules`. La conclusione si dichiara robusta solo se il TGN batte anche
  `lookup_rules_val`. Se non regge: riposizionamento (substrate ZTA, gate anti-poisoning con OPA, parità
  train/serve, generatore + audit).
- **Regole:** la variante primaria riceve etichette vere solo fino a `train_end`, come il TGN (le cui
  repliche di validation e test avanzano su `not signal_dirty`, mai su etichette). `lookup_rules_val` (etichette
  fino a `val_end`, il protocollo dell'audit) è un oracolo sulla validation, quindi un limite superiore per
  le regole; `lookup_rules_all` non usa etichette.
- **E2:** nodo config acceso/spento, appaiato per seed.
- **E3:** budget di dati **a ricetta fissa (15 epoche)**: i budget piccoli fanno meno passi di gradiente,
  quindi N\* è una stima per eccesso. N\* = più piccolo N il cui IC95 appaiato di AUC_N − AUC_pieno ha limite
  inferiore ≥ −0.02 **e** semilarghezza ≤ 0.02, su lateral e theft. Un IC più largo del margine è
  `undetermined`, mai una promozione.
- **Claim di adattamento:** sopravvive solo se N\*_TGN ≤ N\*_IF **e** l'AUC del TGN a N\*_TGN batte quella
  dell'IF a N\*_IF e delle regole a N\*_regole (appaiato, lateral e theft). Altrimenti E3 si riporta come
  curva di costo e il claim esce dall'abstract. Va scritto come "quanta storia benigna serve per
  inizializzare un deployment", **non** come adattamento al drift, che non è misurato.

## Cose da sapere sui confronti

- L'Isolation Forest vede **meno segnali** del TGN (solo l'attore device: niente user/source/config né
  contatori di binding). Il paper (`setup.tex`) dice ancora "stessi segnali": da correggere in S8.
- Il leak di etichette dell'IF (contatori costruiti con le etichette del test) è corretto
  (`label_horizon=val_end`). Il gate dei contatori dopo `val_end` è "aggiorna tutto", mentre quello del TGN è
  `not signal_dirty`: differiscono solo sugli eventi signal-dirty (v5, seed 2000–2001: lateral 8–14 %, theft
  3–5 %, benigni 6–7 %). Effetto piccolo, direzione non determinata.
- OCSVM, XGBoost e GNN statica hanno ancora lo stesso leak: non citarli.
- Panel A/B/C **non** sostituiscono ancora i numeri nel paper: `docs/paper/main.tex` ha `\preliminarytrue`
  e `results.tex` contiene solo numeri v4 stale. La riscrittura (S8) segue i risultati.
