# OpTC con il protocollo LMDEval: runbook

Procedura per valutare il TGN su DARPA OpTC secondo Larroche 2026 (arXiv 2607.29390, codice
`github.com/cl-anssi/LMDEval`). Stato al 2026-09-29: pipeline completa e verificata su un pilota di 25 host
(§8); nessuna run sul protocollo completo. Contesto, rischi e misure sulla release: `external_dataset_optc.md`.

## 1. Protocollo

- Eventi: FLOW START di eCAR con entrambi gli estremi interni, deduplicati sul 5-tupla, indirizzi mappati
  sugli host. Etichette per id evento da `optc_redteam.csv` (1359 id: 249 "Lateral movement", 1110 "Other").
- Split: test = 23, 24 e 25 settembre 2019 (ora locale −04:00), tutti gli host. Training sui giorni precedenti.
  Il paper considera un'inflazione rimuovere host dal test: il test non si riduce.
- Metriche: AUC e AP per evento, in due varianti. "LM only" (positivi = Lateral movement, Other contati come
  negativi, come `--lm-only`) e "all malicious" (positivi = tutti gli id).
- Riferimenti (Tab. 6, AUC/AP in %). LM only: Pikachu 50.74/0.01, Euler 52.90/0.00, Argus 43.23/0.01.
  All malicious: 86.02/0.52, 84.31/0.51, 79.75/3.36.
- Controllo di coerenza: sull'intero dataset il paper conta 1354 positivi di cui 244 LM dopo la
  deduplicazione, tutti nei giorni di test.

## 2. Scelte per una macchina non enterprise

| Voce | Scelta | Motivo |
|---|---|---|
| Sorgente | release corretta di Majorczyk (doi:10.57745/UXCWOC) | un tar per giorno e un file per host, letto via range HTTP; Drive originale soggetto a quota giornaliera |
| Test | 23–25/9, tutti i 500 host | vincolo del protocollo |
| Training | 22/9 (00:00–20:00), calibrazione 20:00–24:00 | adiacente al test; il paper allena su 6 giorni |
| Grezzo | mai salvato: estrazione in streaming | ~1 riga su 500 sopravvive al filtro |
| Replay di test | batched (1024) | il replay per evento (~66 eventi/s) su ~10 M flussi richiede ~42 h |

Volume da leggere: 22/9 124 GB, 23/9 120 GB, 24/9 112 GB, 25/9 68 GB, in tutto 424 GB. Al ritmo misurato
sul pilota (~22 MB/s con 4 processi) sono circa 5,5 h. Su disco restano solo i CSV estratti (ordine di 1 GB).

## 3. Estrazione

Prerequisito: `data/optc/meta/LMDEval/` con `optc_redteam.csv` e `optc_known_addresses.json` (li scarica
`scripts/download_optc.py` al primo download reale; nel repo di lavoro sono già presenti).

```bash
for d in 2019-09-22 2019-09-23 2019-09-24 2019-09-25; do
  .venv/bin/python scripts/optc_extract.py extract --day $d --out data/optc/flows --jobs 8 \
    2>&1 | tee -a tasks/runs/optc_extract_$d.log
done
.venv/bin/python scripts/optc_extract.py build data/optc/flows --meta data/optc/meta/LMDEval \
  --out data/optc/optc_flows.csv.gz | tee tasks/runs/optc_build.log
```

- `extract` scrive un `.csv.gz` per membro del tar, in modo atomico. Rilanciato, salta i membri già
  completi: un'interruzione costa al più i membri in corso. L'indice del tar è salvato in
  `data/optc/flows/<giorno>.tar.index.tsv`.
- Se un membro fallisce dopo 4 tentativi lo script lo segnala e prosegue; si rilancia lo stesso comando.
- `--jobs 8` non è misurato: se la velocità aggregata non sale rispetto a 4, tornare a 4.
- `build` deduplica sull'insieme dei file presenti. Va eseguito una sola volta, dopo che tutti i giorni sono
  stati estratti.

Controlli sull'output di `build` prima di allenare:

- positivi: circa 1354 (all) e 244 (LM). Un piccolo scarto è atteso perché la deduplicazione vede un giorno di
  training invece di sei; uno scarto grande indica membri mancanti;
- inizio al 22/9 00:00 −04:00;
- nei log di `extract`, `invalidi` a 0 o trascurabile per ogni membro.

## 4. Training e valutazione

```bash
docker compose --profile eval-optc run --rm eval-optc /app/tests/eval_optc.py \
  --flows /data/optc/optc_flows.csv.gz \
  --val-start 2019-09-22T20:00:00-04:00 --test-start 2019-09-23T00:00:00-04:00 \
  --epochs 5 --seed 42 --scores-out /data/optc/scores_s42.npz \
  2>&1 | tee tasks/runs/optc_enriched_s42.log
```

- Ripetere con `--seed 43` e `--seed 44`. Le run si lanciano in sequenza: due training in parallelo sulla GPU
  da 16 GB sono già andati in OOM.
- Entro 2 minuti dal lancio cercare `Traceback` e `OutOfMemory` nel log.
- Stima non misurata: ~5 M eventi di training a ~16k eventi/s (ritmo del pilota), cioè ~5 min per epoca; test
  batched in pochi minuti. La RAM necessaria per ~15 M eventi non è misurata.
- `--scores-out` salva gli score grezzi prima di qualsiasi metrica: ogni analisi successiva parte dal `.npz`.
- Il default è `--nodes enriched`: catena a 5 nodi (sorgente = IP, device = host, utente = `principal`,
  config = `image_path`, risorsa = host di destinazione; sentinelle per host sui campi vuoti). È la variante per
  cui OpTC è stato scelto e va riportata come risultato principale, dichiarando che usa più informazione dei
  detector della Tab. 6.
- `--nodes lmdeval` (solo host, come i detector del paper) dà il confronto a parità di informazione: eseguirla
  con gli stessi seed e riportarla accanto.

L'output finale (`OpTC / LMDEval SUMMARY`) riporta, per LM only, all malicious e LM contro soli benigni:
prevalenza, AUC, AP, rapporto AP/prevalenza, TPR a FPR 1% e 0,1%. Riporta poi gli allarmi al giorno alla
soglia calibrata in validazione.

## 5. Controlli da fare prima di scrivere numeri

1. **Deriva del replay batched.** Rieseguire una fetta iniziale del test con `--eval-batch-size 1` e con
   `1024`, a parità di seed (per esempio `--t-max 2019-09-23T01:00:00-04:00`). Confrontare i due `.npz`
   (max e media di |Δ|, correlazione di rango) e dichiarare il risultato.
2. **Baseline di rarità.** Calcolare sugli stessi eventi di test lo score 1/(1 + occorrenze precedenti della
   coppia sorgente→destinazione). Sul pilota ha dato LM AUC 0.993: senza questo confronto un'AUC alta del TGN
   non è interpretabile. Oggi il calcolo non è in `eval_optc.py`: va aggiunto.
3. **Release originale e corretta.** Quando la quota di Drive lo consente, scaricare
   `ecar/evaluation/23Sep19-red/AIA-201-225` con `download_optc.py`, eseguire `extract-local` e `build`, e
   confrontare l'insieme degli id e dei campi dei flussi con il pilota. Serve a dichiarare se la release
   corretta cambia gli eventi valutati.

## 6. Deviazioni da dichiarare

- Training su un giorno (22/9) invece di sei; deduplicazione calcolata sui soli giorni estratti.
- Release corretta di Majorczyk invece dell'originale, salvo l'esito del controllo §5.3.
- Filtro FLOW START esteso al JSON con spazi (unica modifica rispetto a `extract_optc.py`, equivalente sul
  pilota).
- Replay di test batched, con la deriva misurata in §5.1.
- Ordine tra eventi nello stesso secondo: per timestamp assoluto e poi per id (LMDEval: secondo intero, sort
  non stabile).
- Numero di seed (3 contro le 10 run del paper).

## 7. Codice

| File | Ruolo |
|---|---|
| `scripts/optc_extract.py` | `index`, `extract` (range HTTP), `extract-local`, `build` |
| `tests/datasets/optc.py` | flow list → `StreamData`; split per tempo; `nodes=enriched` (default) `\|lmdeval` |
| `tests/eval_optc.py` | training, replay, metriche LMDEval dagli score grezzi |
| `src/train_tgn.py` | `return_scores=True` restituisce gli score di test (default spento) |
| `docker-compose.yml` | profilo `eval-optc` |
| `tests/test_optc_extract.py`, `tests/test_optc_mapping.py` | semantica LMDEval e invarianti della mappatura |

## 8. Pilota (riferimento)

23/9, gruppo AIA-201-225 (25 host), 6,1 GB letti in streaming in ~5 min.

- **Equivalenza.** Stesso output di `extract_optc.py` originale, sia con tutte le etichette sia con
  `--lm-only`: 202.834 flussi, 301 positivi, 18 LM.
- **Copertura dei campi.** `principal` è presente nel 94,4% dei flussi, `image_path` nel 99,8%.
- **TGN.** Split interno al giorno: LM only AUC 0.997 / AP 0.265; all AUC 0.9998 / AP 0.921.
- **Baseline di rarità.** LM AUC 0.993 / AP 0.022.

Questi numeri non sono rappresentativi. Con 25 host e mezza giornata di storia le coppie del lateral movement
non compaiono mai prima dell'attacco. Log: `tasks/runs/optc_pilot_*`. Il grezzo in `data/optc/pilot/raw`
(5,7 GB) serve solo a ripetere il confronto di equivalenza.
