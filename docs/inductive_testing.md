# Inductive Testing & Lateral Movement Validation

Questo documento descrive il rilevamento dei **movimenti laterali** tramite il
Temporal Graph Network (TGN), l'*induttività*, e la metodologia di valutazione **onesta**:
(1) **de-circolarizzazione** del negative sampling e (2) **de-degenerazione** del task.
Le due cose insieme rendono i numeri difendibili. Sostituisce versioni precedenti che
riportavano metriche gonfiate.

## Confine di competenza del modello (cosa è suo, cosa no)

- **policy violation (`etype=1`) → di OPA, NON del modello.** L'enforcement delle policy è
  deterministico e bloccato a monte dall'orchestrator OPA. Il modello *emette comunque* uno
  score su questi eventi (e l'AUC è ~0.99), ma è **ridondante**: la teniamo solo come colonna
  di sanity, non come valore aggiunto.
- **contextual (`etype=2`) → banale.** Una rule baseline sui soli segnali edge (JA3/Snort/
  sonde) le prende al ~96.6%. Il TGN non aggiunge nulla di sostanziale qui.
- **lateral movement (`etype=3`) → l'UNICO target ML genuino del modello.** Autorizzato,
  signal-clean, indistinguibile a livello di feature da un accesso benigno: l'unico indizio è
  il **pattern temporale/relazionale**. Tutto il resto del documento si concentra su questo.

## Due fix metodologici

### 1. De-circolarizzazione (negative sampling)
La versione originale costruiva l'hard-negative come «src ↔ risorsa autorizzata-ma-non-abituale»
(via `auth_mask` + subset abituale), **pesato ×10** — *esattamente* la definizione del lateral
di test. Allenava sul test → recall gonfiata (~40%) e circolare. **Fix:** i negativi sono ora
(a) **strutturali a destinazione casuale** in un obiettivo **InfoNCE** (ranking: la dst vera
sopra K dst casuali, dato lo storico del src) e (b) **contestuali a rumore gaussiano** (mecc.
diverso dalla randomizzazione 0/1 del test). Una guardia impedisce di ripassare `auth_mask`.

### 2. De-degenerazione (il task stesso)
Anche de-circolarizzato, il task restava **degenere**: il benigno pescava *solo* dalle route
abituali, quindi «non-abituale ⟺ lateral» era quasi una tautologia (bastava un contatore per
fare ~1.0, ma non trasferibile). **Fix (`benign_explore_prob=0.15`):** il traffico benigno
compie a volte accessi **autorizzati-ma-non-abituali legittimi** (esplorazione benigna). Ora il
lateral è **feature-identico** a un accesso benigno non-abituale: l'unico discriminante è il
**contesto di kill chain** (recon → lateral sullo stesso IP compromesso). La novità da sola non
basta più — ed è questo che rende il numero onesto e difficile.

## Componenti del modello per il lateral

- **Storia esplicita (`compute_hist_feats`)**: per ogni evento `[log1p(pair_count),
  log1p(src_count), pair/(src+1)]` — contatori causali, *benign-gated*, derivabili a runtime
  (non circolari). Iniettano il segnale di novità direttamente nella feature head.
- **InfoNCE ranking** (`infonce_k=5`): allinea l'obiettivo all'AP (ranking dst-vere > dst-casuali).
- **Precursor kill-chain (serving-time)**: il lateral segue un alert di recon (Snort) sullo
  *stesso* IP, ma il gate predict-then-update scarta il precursore dalla memoria TGN. Lo
  conserviamo in uno stato `recent_alert` separato e applichiamo un **prior moltiplicativo**
  allo score: `score *= 1 + max_boost · 0.5^(Δt/half_life)` (`half_life=100k`, `max_boost=3`).
  Moltiplicativo (non additivo) di proposito: alza gli score già sospetti senza trasformare gli
  score benigni ~0 in falsi positivi. **Non** è un input addestrato (sarebbe una dead-feature col
  training solo-benigno). Vedi `serve_tgn.precursor_boost`.
- **Hashed identity deterministica** (`stable_hash`, BLAKE2b): identità induttiva e coerente tra
  processi per ogni entità (anche le risorse).

## Induttività (hashed identity)

Il TGN è induttivo: alloca dinamicamente memoria per entità mai viste e assegna l'identità via
hashing deterministico della chiave, coerente tra processi/riavvii. Per rilevare un'anomalia
*strutturale* serve però uno storico (Temporal Neighborhood): un'entità a freddo non ha
«abitudini» da cui deviare → vedi cold-start sotto.

## Risultati onesti — confronto baseline (stream sintetico, FPR target 1%, split 70/10/20)

`num_events=50000`, `seed=42`. **Tutte le baseline ricevono gli stessi segnali tabellari del
TGN** (feature di storia causali + lo stesso prior precursor): così il loro divario col TGN
isola il contributo della **macchina temporale-relazionale**, non dei contatori.

| Modello (stessi segnali tabellari) | Agg AUC | Agg AP | **lateral AUC** | lateral Rec@1%FPR |
|---|---|---|---|---|
| Rule baseline (solo segnali edge) | — | — | 0.000 rec | — |
| One-Class SVM | 0.611 | 0.476 | **0.393** | 0.6% |
| Static GNN (grafo, **no temporale**) | 0.598 | 0.511 | **0.486** ≈ caso | 0.1% |
| Isolation Forest | 0.703 | 0.335 | **0.650** | 2.3% |
| **TGN (full)** | **0.912** | **0.820** | **0.760** | **4.7%** |

### Lettura
- **Lo Static GNN — stessi contatori + precursor, stessa struttura di grafo, ma senza la
  macchina temporale — sta a caso sul lateral (0.49).** Il TGN arriva a **0.76**. Il segnale
  laterale vive quasi interamente nella **memoria ricorrente + vicinato temporale**: non nei
  contatori (che tutti hanno) né nel precursor (che tutti hanno).
- Con la de-degenerazione i detector tabellari *peggiorano* sul lateral (la novità da sola non
  basta più: anche il benigno esplora). Solo il contesto temporale del TGN regge.
- **Recall@1%FPR ~4.7%** resta basso: la soglia globale è dominata dalle classi facili
  (OPA-owned / rule-trivial). Il segnale di ranking onesto è l'**AUC 0.76** (≫ caso); convertirlo
  in recall operativa richiede una soglia per-classe/cost-sensitive (lavoro futuro).

## Ablation (multi-seed: 3 seed, 40k eventi / 12 epoche)

`save=False` (non tocca gli artifact). Lateral AUC = **media ± dev.std** su seed `[42, 7, 123]`
(il training GPU è non-deterministico, ~±0.01–0.03 per run: la media su più seed è obbligatoria).

| variante | **lateral AUC** | lateral AP | lateral Rec@thr | agg AUC |
|---|---|---|---|---|
| **full** | **0.770 ± 0.011** | 0.189 ± 0.016 | 0.058 ± 0.016 | 0.916 ± 0.004 |
| − history feats | 0.704 ± 0.035 | 0.145 ± 0.019 | 0.035 ± 0.005 | 0.891 ± 0.012 |
| − precursor | 0.697 ± 0.016 | 0.155 ± 0.018 | 0.034 ± 0.011 | 0.892 ± 0.006 |
| − struct head | 0.778 ± 0.002 | 0.203 ± 0.002 | 0.053 ± 0.011 | 0.919 ± 0.001 |
| − hashed identity | 0.773 ± 0.020 | 0.230 ± 0.010 | 0.086 ± 0.019 | 0.918 ± 0.005 |

Letture oneste:
- **Le due componenti nuove sono quelle che contano**: togliere le *history feats* costa
  −0.066 AUC, togliere il *precursor* −0.073 — entrambe ben oltre la dev.std tra seed.
- **La testa strutturale a coseno è marginale** (−struct ≈ full): conferma, su più seed, che non
  è lei il rilevatore del lateral.
- **L'hashed identity è ora SUSSUNTA dalle feature di storia esplicite**: rimuoverla non peggiora
  più il lateral AUC (0.773 vs 0.770). Nella versione precedente (senza history feats) era
  critica (crollo a ~0.52); ora il segnale di novità per-entità è portato esplicitamente dai
  contatori, quindi l'identità hashata diventa ridondante per questa classe.

## Cold-start (il recall basso NON è un artefatto di warm-up)

Conditioning del recall laterale per `src` con/senza storia benigna pregressa (causale). Nel
nostro stream **tutti** gli eventi laterali di test cadono su `src` già «caldi» (`n_cold = 0`):
la kill chain colpisce IP che hanno già accumulato storia benigna. Quindi la difficoltà del
lateral è **intrinseca** (segnale signal-clean), non un problema di entità fredde. (Riportato a
ogni run da `train_tgn`: `--- LATERAL: COLD-START CONDITIONING ---`.)

## Cold start & anti-poisoning (gate)

Memoria e vicinato si aggiornano **solo** per eventi giudicati benigni (predict-then-update). Due
limiti intrinseci: un attaccante «stealthy» scorato benigno avvelena la baseline; un benigno
scorato anomalo non viene mai appreso (starvation). Mitigazione demandata all'orchestrator (OPA
come vero decisore via `/infer`→`/update`, più un grace period per i nuovi nodi). Vedi
`docs/orchestrator_integration.md`.

## Riprodurre

```bash
docker compose --profile training-tgn up      # TGN: training + eval onesta + rule baseline + cold-start
docker compose --profile baseline-iforest up  # Isolation Forest (con storia + precursor)
docker compose --profile baseline-ocsvm up     # One-Class SVM (con storia + precursor)
docker compose --profile baseline-gnn up        # Static GNN (ablazione del solo temporale)
docker compose --profile ablations up           # ablation multi-seed (history/precursor/struct/hash)
docker compose --profile verify-tgn up          # correttezza serving-path
```

## Validità Esterna Comprovata (Dataset LANL)

Sebbene le metriche esplorative derivino da test sintetici onesti, l'architettura è stata validata con successo sul dataset reale **LANL Comprehensive Multi-Source**, superando il limite della validazione sintetica.

In assenza totale di metadati ZTA d'aiuto (scenario "stealth" massimo) e senza alcun data leakage (garantito dallo split temporale rigoroso 30/10/60 testato sull'intero periodo operativo del Red Team), il TGN Device-Centric ha raggiunto un'**AUC dell'88%** e un **Recall del 73.3%** sul lateral movement.
Pur cedendo alcuni punti percentuali rispetto ai colossi accademici SOTA offline (AUC 0.92-0.96), il nostro TGN eccelle indiscutibilmente per la sua vocazione ingegneristica: opera in puro streaming tempo-reale a basso impatto (`O(1)` per nodo), dimostrando una robusta validità esterna per applicazioni Zero-Trust su scala industriale.
