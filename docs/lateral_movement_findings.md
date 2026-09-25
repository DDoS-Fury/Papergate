# Movimento laterale: misure, decisioni e posizionamento

Stato al 2026-09-24, branch `fix/generator`. Tutte le misure sono sul **seed di sviluppo
2000**. I seed pre-registrati 1000–1009 non sono stati toccati e E1 non è stato eseguito:
nessuno dei numeri qui dentro può entrare in una tabella dei risultati del paper come
risultato finale.

---

## 1. Modifiche al modello

Tre difetti distinti del precursore kill-chain, tutti misurati prima di intervenire.

| # | difetto | correzione |
|---|---|---|
| 4 | `1 - sigmoid(logit)` satura a esattamente 1.0 in float32 sotto logit −16; 47 benigni erano appaiati in cima e nessun laterale | punteggio in spazio logit (`infer_logit`), conversione con `expit` in float64: la saturazione passa a logit ≈ −36.7 |
| 2 | prior moltiplicativo su una probabilità, poi clippato a 1.0: no-op sugli eventi saturi, irrilevante su quelli bassi | **shift additivo sul logit** (`precursor_shift`): è un prior sulle odds, quindi muove ogni evento della stessa evidenza in nat |
| 2a | `precursor_half_life = 600 s` contro un Δt recon→laterale mediano misurato di **8.7 h**: boost mediano esattamente 1.000, meccanismo matematicamente inerte | half-life dimensionata sulla misura |
| 3 | arming su `score >= eff_thr` con `eff_thr` = sentinella sul clean: solo Snort poteva armare il precursore | soglia di arming separata (`threshold_arm`), pari al quantile clean label-free |

Refactor di supporto: `_fit_thresholds` estratta a livello di modulo come
`train_tgn.fit_thresholds(scores, labels, v_types, v_msg, cfg)`, pura nei suoi input, così
lo sweep ricalibra senza duplicarne la logica.

**Verifica**: `tests/verify_replay_batching.py` → `OVERALL PARITY: PASS` su tutti e quattro i
percorsi (v4 5-edge e v3 legacy × gate di calibrazione e di valutazione), scarti massimi
8.5e-08 … 1.9e-07 contro una soglia di contratto di 1e-5.

---

## 2. Sweep del precursore

Un solo training, stato runtime pre-calibrazione fotografato, poi (cal A, cal B, replay test)
ripetuti per ogni configurazione: confronto **esattamente appaiato**, stessi pesi, stesso
stream, stessa memoria iniziale. Un asse per volta attorno al centro (24 h, 4 nat, arming on).

| configurazione | latAUC | latR@1% | theftAUC | aggAUC | FPR |
|---|---|---|---|---|---|
| solo spazio logit (prior inerte) | 0.7605 | 0.0569 | 0.6522 | 0.8856 | 0.0099 |
| hl=24h sh=2 arm | 0.7933 | 0.0569 | 0.7052 | 0.8991 | 0.0097 |
| hl=24h sh=4 arm | 0.8045 | 0.0664 | 0.7144 | 0.8993 | 0.0102 |
| hl=24h sh=8 arm | 0.7953 | 0.0379 | 0.7159 | 0.8868 | 0.0097 |
| hl=6h sh=4 arm | 0.7802 | 0.0474 | **0.7385** | **0.9008** | 0.0100 |
| **hl=72h sh=4 arm** (scelta) | **0.8252** | **0.0758** | 0.6741 | 0.8958 | 0.0099 |
| hl=24h sh=4 **no-arm** | 0.7992 | 0.0569 | 0.6289 | 0.8876 | 0.0097 |

Riferimento v5.3 (prima di queste modifiche): latAUC 0.760, latR@1% 0.0521, theftAUC 0.651.

**Attribuzione**: spazio logit +0.005 di recall; precursore vivo +0.044 lat e +0.062 theft;
**arming scollegato +0.086 su theft** (0.7144 contro 0.6289).

**Decisione presa**: `precursor_half_life = 259200` (72 h), `precursor_max_shift = 4.0`.
Massimizza il laterale, che è l'obiettivo. Costo dichiarato: theft a 0.6741 invece di 0.7385.

**Ipotesi non ancora verificata**: i due passaggi del kill chain non condividono una scala
temporale (recon→laterale, mediana misurata 8.7 h; laterale→esfiltrazione, **non misurata**).
Se regge, due costanti di decadimento eliminano il compromesso. Misura offline, da fare.

`thr_clean` resta la sentinella `1.0000000009` in **tutte** le righe: il routing
cost-sensitive è ancora inerte e la decisione su cosa farne resta aperta.

---

## 3. Da dove viene il limite

Composizione dei 395 falsi positivi alla soglia @1% FPR:

| mask | % dei FP | % dei benigni | FPR interna | lift |
|---|---|---|---|---|
| `cfg\|usr_new` | 0.729 | 0.173 | 0.043 | 4.2 |
| `dev\|usr_new` | 0.711 | 0.300 | 0.024 | 2.4 |
| `src\|usr_new` | 0.653 | 0.241 | 0.028 | 2.7 |
| guest | 0.549 | 0.147 | 0.038 | 3.7 |
| `ja3=0` (legacy client) | **0.000** | 0.000 | — | — |

91.4% dei FP sono spiegati da dirty | novelty | guest; 34 restano inspiegati. I legacy client
sono spariti: la correzione al generatore ha chiuso quel canale.

La soglia è dettata dalla **novità benigna**, non dai laterali. Il movimento laterale ha la
stessa firma per-evento della novità benigna, che è il 17–30% del traffico contro lo 0.5% di
laterale. Non è un difetto del modello: è la struttura del problema, costruita di proposito.

Distribuzione dei logit: benigno p50 −8.86, p99 −1.50, **max +26.50**; laterale p50 −6.65,
**max +7.64**. Nessun laterale arriva in cima alla classifica; i primi posti sono tutti FP
benigni con ~19 nat di margine.

---

## 4. Risultato negativo: l'accumulo causale peggiora

Vincolo: il sistema gira **sempre in tempo reale**, uno score per singola richiesta.

I numeri per device-settimana con bucket di **calendario** (precision 0.778, recall 0.560,
AUC 0.923) contengono eventi **successivi** al punto di decisione. Sono un look-ahead, non
una capacità consegnabile, e **non vanno riportati come risultato**.

L'analogo deployabile è una finestra **scorrevole** (solo passato). Misurato:

| variante | latAUC | latRecall | prec(agg) | campagne | ev. a TTD | h a TTD | pre-exfil |
|---|---|---|---|---|---|---|---|
| **per-evento (nessun accumulo)** | **0.8252** | 0.0758 | **0.5954** | **10/29** | **4.0** | **39.3** | **8/29** |
| trailing top-3, 3 g | 0.7649 | 0.0806 | 0.2202 | 11/29 | 7.0 | 136.9 | 1/29 |
| trailing top-3, 7 g | 0.7433 | 0.0569 | 0.1371 | 9/29 | 9.0 | 164.8 | 0/29 |
| trailing top-3, 14 g | 0.6474 | 0.0427 | 0.0833 | 6/29 | 9.0 | 183.7 | 0/29 |

Peggiore su ogni colonna. L'accumulatore tira dentro la coda di novità benigna del dispositivo
e poi **sbava**: appena una macchina ha un evento alto, il top-k resta alto per tutta la
finestra e si alzano allarmi sui suoi eventi benigni successivi. Precision 0.595 → 0.137.

**Inversione utile**: un accumulatore causale che funziona c'è già ed è il precursore, che
rende perché è **gated su un evento** (si arma su un allarme) e non **mediato** (top-k di
tutto). Non confondere "usare il contesto temporale" con "mediare sul contesto temporale".

---

## 5. Cosa riporta davvero lo stato dell'arte

Nessun lavoro riporta ">90% di precision e recall". Quel numero non esiste in questa
letteratura: il >90% che circola è l'**AUC**.

| lavoro | dataset | AUC | AP | precision | recall operativa |
|---|---|---|---|---|---|
| Euler GCN+GRU (NDSS'22) | LANL | 0.9912 | 0.0523 | **0.0054** | TPR 86.1% @ FPR 0.57% |
| Euler GCN+LSTM | LANL | 0.9913 | 0.0169 | 0.0056 | TPR 89.7% @ FPR 0.57% |
| Euler GAT+LSTM | LANL | 0.8713 | 0.0022 | 0.0002 | TPR 96.8% @ FPR 19.9% |
| PIKACHU (NOMS'22) | LANL | ~0.99 | n.d. | n.d. | TPR 95.1% |
| Argus | LANL (no dup) | 0.9821 | **0.0056** | — | Rec@10 = 0.1126 |
| UltraLMD++ (ANSSI'25) | LANL (no dup) | 0.9868 | **0.0088** | — | Rec@10 = 0.1788 |
| UltraLMD++ | OpTC | 0.9909 | 0.1510 | — | Rec@10 = 0.0623 |
| CyberGFM (arXiv 2601.05988, gen. 2026) | LANL (split casuale) | 0.9994 | **0.7600** | — | n.d. |
| CyberGFM | OpTC (split casuale) | 0.9739 | 0.8981 | — | n.d. |

Sono i numeri **auto-dichiarati**. La rivalutazione indipendente a livello di evento
(Larroche, ANUBIS'26, [2607.29390](https://arxiv.org/abs/2607.29390); 702 eventi malevoli su
369,6 M) dà su LANL: Euler AUC 0.980 / AP **0.0002** (riportato 0.0523), Argus 0.984 / **0.0009**,
Pikachu 0.783 / 0.0000. Su OpTC, solo movimento laterale, tutti al caso (AUC 0.43–0.53).
Nel paper è in `related.tex` con le macro `\LitRe*` di `results.tex`.

**CyberGFM** (King, Trindade, Bowman, Huang; letto il 2026-09-25): encoder BERT (2.57 M parametri)
preaddestrato a predire token mascherati su cammini casuali del grafo degli host, poi fine-tuning
di link prediction con negativi casuali, senza label di attacco. Nel loro protocollo Argus fa AP
0.2279 ed Euler 0.0627 su LANL. Perché lo 0.76 non è confrontabile con noi:
- split **casuale** 80/10/10 degli edge benigni, non temporale ("This split was infeasible in this
  case"): una coppia benigna di test può essere già vista in training;
- nessuna baseline "edge mai visto" sotto quello split: la quota dovuta alla novità non è misurata;
- **non induttivo** ("a major limitation of our approach is that it is non-inductive"): non valuta
  host nuovi; in una ZTA device / IP / config nuovi compaiono di continuo;
- solo AUC e AP, nessun punto operativo; non è spiegato come 1 G di eventi LANL diventino 3 M edge
  (filtro NTLM "as in prior works").
Test decisivo non ancora fatto: regola "mai visto" sul loro protocollo LANL, per vedere quanta AP
fa da sola. Idea utile per l'obiettivo di training: anche loro usano negativi casuali, il guadagno
viene dal contesto del cammino. Nel paper: `related.tex`, macro `\LitCgfm*` in `results.tex`.

Setup LANL di Euler: 17.685 nodi, 45.871.390 eventi, **518 archi anomali**, 58 giorni,
snapshot δ = 1800 s, archi (src, dst) deduplicati nella finestra. Prevalenza ~3.6e-5 contro
la nostra 5.4e-3: **150×**.

Citazioni testuali utilizzabili:
- Euler: *"this metric [AUC] is not a good indicator of model quality on data sets with
  imbalance as extreme as LANL"*.
- Larroche (ANSSI): *"Since the AUC tends to be overly optimistic in highly imbalanced
  settings (base rate fallacy), we also compute the average precision"*.

### I nostri numeri sullo stesso protocollo

| | prevalenza | AUC | AP | lift AP |
|---|---|---|---|---|
| laterale | 5.4e-3 | 0.8252 | 0.0253 | 4.7× |
| tutte le anomalie | 3.0e-2 | 0.8958 | 0.4335 | 14.4× |
| UltraLMD++ LANL | 3.6e-5 | 0.9868 | 0.0088 | ~244× |

**Non riportare Rec@B nel confronto.** Il nostro stream ha ~15 eventi per finestra da 30 min,
quindi Rec@10 significa investigare il 66% del traffico; su LANL una finestra ne ha ~5200 e il
loro Rec@10 è un budget dello 0.19%. Il nostro Rec@10 = 0.7915 contro 0.1788 è un **artefatto**.

### Il numero che giustifica il divario

Regola "Unknown Authentication" (coppia assente dal grafo di training), **protocollo identico
a Euler, grafo congelato**:

| | TPR laterale | FPR benigno | lift |
|---|---|---|---|
| noi, OR di tre coppie | 63.03% | **42.70%** | **1.48** |
| noi, device→user | 44.55% | 29.97% | 1.49 |
| **Euler su LANL** | 72.00% | **4.40%** | **16.36** |

TPR comparabile, FPR dieci volte peggiore: da noi "mai visto prima" descrive il 43% del
traffico benigno, su LANL il 4.4%. Su LANL il movimento laterale è in larga parte
memorizzazione del grafo; `benign_explore_prob` quella scorciatoia l'ha rimossa.

---

## 6. Conseguenze per il paper

- [x] Correggere la descrizione del precursore in `approach.tex`: era moltiplicativa, il
      codice ora è additivo sul logit. Fatto, il documento compila.
- [x] Aggiungere a `related.tex` cosa riportano davvero Euler / Argus / UltraLMD++, la
      base rate fallacy e la misura UA di difficoltà relativa. Fatto.
- [ ] Abbandonare il target ">90% precision e recall". I target realistici sono AP e
      precision a FPR fissato; lo stato dell'arte sta a AP 0.006–0.15.
- [ ] Riportare sempre AUC **e** AP **e** la prevalenza insieme.
- [ ] Rivendicare ciò che nessuno di loro riporta: **tempo alla rilevazione** (mediana 4
      eventi / 39 h) e **campagne fermate prima dell'esfiltrazione** (8/29). Metriche native
      dello streaming, non producibili da una valutazione retrospettiva. Scoring per evento
      in tempo reale e training self-supervised **non** sono più rivendicabili: li fa già il
      TGN su CloudTrail (Nandan et al., WoRMA'26, [2606.28923](https://arxiv.org/abs/2606.28923)),
      che però aggiorna la memoria con tutti gli eventi e non misura i falsi negativi.
- [x] Citare Larroche 2026, Jbeil, Kairos e il TGN su CloudTrail; correggere la voce `argus`
      di `refs.bib` (puntava a Bowman et al., RAID'20); togliere "and measured" dalla riga
      del gate nella tabella qualitativa. Fatto il 2026-09-25, il documento compila.
- [ ] Misurare il gate: TGN con commit-all (regime CloudTrail TGN) contro gate OPA e
      quarantena, stessi pesi, seed di sviluppo. Oggi il paper non quantifica il gate.
- [ ] Dichiarare le tre differenze che rendono incomparabili i valori assoluti: unità
      (richiesta contro arco deduplicato), prevalenza (150×), difficoltà della discriminante
      banale (11×).
- [ ] Eseguire E1 sui seed 1000–1009 e sostituire ogni numero di questo documento.

## Script

| file | cosa misura |
|---|---|
| `tasks/tmp/lateral_chain_diag.py` | struttura delle campagne, Δt recon→laterale |
| `tasks/tmp/precursor_sweep.py` | sweep del precursore, un training e N ricalibrazioni |
| `tasks/tmp/fp_breakdown.py` | composizione dei falsi positivi, recall di campagna |
| `tasks/tmp/agg_metrics.py` | recall/FPR per device-giorno e device-settimana |
| `tasks/tmp/aggregator_probe.py` | confronto degli aggregatori a livello di bucket |
| `tasks/tmp/online_probe.py` | accumulo causale, tempo alla rilevazione, pre-esfiltrazione |
| `tasks/tmp/lit_protocol.py` | AUC / AP / Rec@B sul protocollo della letteratura |
| `tasks/tmp/ua_difficulty.py` | regola Unknown Authentication, difficoltà relativa a LANL |

Fonti: [Euler, NDSS 2022](https://www.ndss-symposium.org/wp-content/uploads/2022-107A-paper.pdf) ·
[Larroche, ANSSI 2025](https://arxiv.org/abs/2504.13527) ·
[Larroche, ANUBIS 2026](https://arxiv.org/abs/2607.29390) ·
[PIKACHU, NOMS 2022](https://ieeexplore.ieee.org/document/9789921/)
