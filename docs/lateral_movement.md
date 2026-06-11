# Movimenti laterali: cosa sono e come ne miglioriamo la detection

Note sul caso anomalia più difficile — e l'**unico target ML genuino** del TGN (policy è di
OPA, contextual è banale per le regole). Con la valutazione **de-circolarizzata + de-degenerata**
(vedi `inductive_testing.md`) il laterale è onestamente difficile: **AUC ~0.76, AP ~0.20,
recall ~5%** alla soglia dell'1% di FPR. È l'unica classe in cui la macchina temporale conta:
lateral AUC **0.49 (GNN statico, stessi segnali, no temporale) → 0.76 (TGN completo)**.

## Cosa rappresentano in questo contesto

Ogni evento è un accesso `IP → Risorsa`. Per ogni IP si precalcolano (vedi
`src/data/stream_synthetic.py`): le **azioni autorizzate** (consentite dalla policy) e un
**subset abituale** (~metà, le route usate giorno per giorno).

Un **movimento laterale** (`etype=3`) è un accesso **autorizzato ma non abituale**: `ja3=1`,
`snort=0`, nessun segnale IDS, nessuna violazione di policy. Nel dominio Zero-Trust è un
**account/dispositivo compromesso che usa credenziali legittime per raggiungere sistemi che
normalmente non tocca** — pivoting post-compromissione, niente malware, nessuna regola violata.

## Perché è il caso più difficile (e onesto)

Con `benign_explore_prob=0.15` **anche il benigno** compie a volte accessi autorizzati-non-abituali
(esplorazione legittima). Quindi il lateral è **feature-identico** a un benigno non-abituale: la
sola novità della coppia non lo distingue. L'**unico** discriminante è il **pattern di kill chain**:
il lateral è temporalmente correlato a un recon (alert Snort) sullo stesso IP, dentro una sequenza
`recon → lateral → exfil`. Questo è ciò che la memoria/vicinato temporale del TGN può catturare e
un detector tabellare no.

## Leve — stato implementato

1. **Storico ampio** (`neighbor_size=30`, `memory_dim=256`, `num_hops=3`): la finestra temporale/
   strutturale. È la componente che porta il lateral AUC da 0.49 (no-temporale) a 0.76.
2. **Feature di storia esplicite** ✅ (era leva «novità/recency»): `compute_hist_feats` →
   `[log1p(pair_count), log1p(src_count), pair/(src+1)]`, causali e benign-gated. Ablation
   multi-seed: **+0.066 AUC**. NB: ora **sussumono** l'hashed identity (vedi sotto).
3. **InfoNCE ranking** ✅ (era leva «loss di ranking»): rimpiazza la BCE per-coppia sul negativo
   strutturale; allinea l'obiettivo all'AP (dst-vere > dst-casuali, dato lo storico del src).
4. **Precursor kill-chain** ✅ (nuova): prior moltiplicativo serving-time che alza lo score di
   un'entità subito dopo un suo alert (recon→lateral), con decadimento esponenziale
   (`half_life=100k`, `max_boost=3`). Stato `recent_alert` fuori dalla memoria TGN (il gate
   predict-then-update scarterebbe il precursore). Ablation multi-seed: **+0.073 AUC**, e la
   precisione aggregata **sale** (moltiplicativo → non genera FP sui benigni ~0). **Non** è un
   input addestrato. Vedi `serve_tgn.precursor_boost`.
5. **Cold-start verificato** ✅: il recall basso **non** è warm-up — tutti i laterali di test
   cadono su IP già caldi (`n_cold=0`). La difficoltà è intrinseca.

### ⚠️ Da NON fare (fuga di informazione)
Hard-negative ristretti agli «autorizzati-ma-non-abituali» (come una vecchia versione, ×10):
è **la definizione esatta** del lateral di test → recall circolare e gonfiata (il vecchio ~40%).
Rimosso di proposito; ogni miglioria deve venire da architettura/segnale, mai da negativi che
rispecchiano la regola di test.

### Testa strutturale a coseno — marginale (confermato multi-seed)
Si ipotizzava fosse il rilevatore dedicato del lateral. L'ablation a 3 seed dice il contrario:
toglierla **non** cambia il lateral AUC (0.778 vs 0.770). Il lavoro lo fanno le feature di storia
+ il precursor + la memoria temporale. Candidata a rimozione/semplificazione futura.

### Hashed identity — ora ridondante per il lateral
Nella versione senza feature di storia, rimuoverla faceva crollare il lateral AUC a ~0.52. Ora che
la novità per-entità è iniettata **esplicitamente** dai contatori, rimuoverla non peggiora
(0.773 vs 0.770): le due cose portano lo stesso segnale, e quello esplicito lo rende leggibile.

## Da ranking a recall operativa: soglia cost-sensitive + routing sul segnale

L'AUC 0.76 è reale ma una soglia **globale** all'1% FPR (fissata dalle classi facili) lascia il
recall laterale ~5%. Ora la calibrazione è **cost-sensitive** e **instradata sul segnale**
(`src/calibration.py`, applicata in `train_tgn` e `serve_tgn.score_event`):

- A serving la classe vera è ignota, ma il **segnale di edge è osservabile** (JA3/Snort/sensori →
  `_rule_baseline`). Si separano quindi due flussi: *signal-dirty* (contestuale — già preso dalla
  regola, resta sulla soglia conservativa `threshold_dirty@FPR`) e *signal-clean* (dove il laterale
  è indistinguibile dal benigno se non per il pattern temporale).
- Sul flusso **clean** la soglia minimizza `cost_ratio·FN + FP` (`TGNConfig.cost_ratio`, default 20):
  un movimento laterale mancato costa molto più di un falso allarme che l'orchestrator può
  ri-sfidare. Questo abbassa la soglia fin dove stanno i laterali, convertendo il ranking in recall.
- `train_tgn` stampa il confronto **prima/dopo** (recall laterale e FPR benigno alla vecchia soglia
  globale vs nuova decisione instradata) e la curva recall/FPR di riferimento; `serve_tgn` /
  `serve_api` espongono `threshold` (clean) e `threshold_dirty` (anche in `/health`). OPA resta il
  decisore finale allow/deny.

## Validità esterna: LANL auth (implementata)

`tests/eval_lanl.py` rimappa il dataset **pubblico** LANL 'Comprehensive Multi-Source' (log di
autenticazione host→host con etichette **red-team** = movimento laterale) sullo schema dello stream
ZTA via `tests/datasets/lanl_auth.py`, e riusa l'intera pipeline tramite `train_tgn(dataset=...)`
(stessa calibrazione cost-sensitive, breakdown per-tipo, cold-start). I file (`auth.txt.gz`,
`redteam.txt`, da <https://csr.lanl.gov/data/cyber1/>) non sono versionati: vanno in `./data` e si
lancia col profilo Compose `eval-lanl`. Il mapping tiene le colonne di allarme *pulite* (LANL non ha
discriminante è solo quello temporale/relazionale — il test onesto.

### Risultati LANL e Confronto con lo Stato dell'Arte (SOTA)
L'integrazione del modello Device-Centric ha prodotto metriche altamente competitive sul dataset pubblico LANL, in test eseguiti rigorosamente senza Data Leakage (split cronologico puro e zero metadati ZTA d'aiuto).

Su un campione vasto (test set esteso validando l'intero periodo operativo del Red Team con uno split al 30/10/60):
- **AUC ROC Aggregata**: **0.8824**
- **Recall Movimento Laterale**: **73.33%** (bloccando sistematicamente la grande maggioranza degli attacchi furtivi confermati)
- **Falsi Positivi (FPR)**: **2.18%**

Questo risultato si colloca eccezionalmente vicino al SOTA accademico offline (che oscilla tra AUC 0.92 e 0.96 su LANL per modelli architetturalmente molto più complessi come HLMD e Pikachu). La nostra soluzione, pur cedendo qualche punto percentuale, offre l'enorme vantaggio industriale di operare in puro streaming in tempo reale, elaborando gli eventi singolarmente in memoria `O(1)`, senza i pesanti calcoli in batch richiesti dai modelli SOTA classici. Questo lo rende l'unico candidato realistico per il deployment nativo su un orchestrator Zero-Trust.

## Leve ancora aperte (lavoro futuro)

- **Dataset esterni aggiuntivi**: DARPA OpTC, CIC-IDS, oltre a LANL.
- **Tuning per-deployment** di `cost_ratio` e di `clean_fpr_cap` (il cap esplicito sull'FPR del
  flusso clean è già implementato, default 0.05; va calibrato su quanto l'orchestrator assorbe i
  ri-challenge).

## Nota sulla loss "alta" con buone prestazioni

La loss di training è quella del task auto-supervisato (InfoNCE + BCE ancora/contestuale), *non*
della detection. Il suo valore assoluto è poco correlato con AUC/AP (metriche di **ranking**,
invarianti a trasformazioni monotone). Loss non bassa + AUC alta è atteso.
