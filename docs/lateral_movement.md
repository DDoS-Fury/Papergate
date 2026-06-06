# Movimenti laterali: cosa sono e come migliorarne la detection

Note di analisi sul caso anomalia più difficile del modello TGN. Con la valutazione
**de-circolarizzata** (vedi `inductive_testing.md`) il laterale è onestamente debole:
**AUC ~0.71, AP ~0.22, recall ~16%** alla soglia dell'1% di FPR — contro AUC ~0.99 di
`policy` e `contextual`. È comunque l'**unica** classe in cui la macchina temporale
conta: AUC 0.50 (non-relazionale) → 0.59 (GNN statico) → 0.71 (TGN completo).

## Cosa rappresentano in questo contesto

Nello stream sintetico ogni evento è un accesso `IP → Risorsa`. Ogni IP appartiene
a un utente con `(ruolo, clearance, device tier)`, e ogni risorsa ha regole di
accesso. Per ogni IP si precalcolano (vedi `src/data/stream_synthetic.py`):

- **azioni autorizzate** — tutte le `(risorsa, metodo)` consentite dalla policy;
- **subset abituale** — ~metà delle autorizzate, le route usate giorno per giorno.

Il traffico benigno pesca solo dal subset abituale. Un **movimento laterale**
(`etype = 3`) è un accesso **autorizzato ma non abituale**: `ja3=1`, `snort=0`,
nessun segnale IDS, nessuna violazione di policy. Le feature dell'edge e le feature
statiche dei nodi sono **identiche** a un evento benigno.

Nel dominio Zero-Trust corrisponde a un **account/dispositivo compromesso che usa
credenziali legittime per raggiungere sistemi che normalmente non tocca mai** — il
pivoting "di lato" post-compromissione. Niente malware, niente regola violata:
l'unico indizio è che *quella coppia (entità, risorsa) è nuova rispetto alla storia
di interazione dell'entità*.

## Perché è il caso più difficile

`policy` e `contextual` hanno un segnale esplicito nelle feature (statiche o
dell'edge) → AUC ~1.0. Il laterale **no**: vive solo nella memoria temporale /
strutturale. La testa preposta a catturarlo è la **`struct` head** di `tgn.py`
(cosine similarity degli embedding proiettati, `struct_proj` + `struct_scale`),
allenata con negativi strutturali a **destinazione casuale** (`train_tgn.py`,
`_sample_structural_negatives`). Diverse leve restano sottosfruttate — ma vanno
perseguite **senza** reintrodurre la fuga di informazione (vedi avviso al punto 2).

## Leve di miglioramento (ordinate per impatto/sforzo)

1. **`neighbor_size = 10` è probabilmente il collo di bottiglia.** L'abitudine è una
   proprietà di *insieme* sulla storia dell'IP, ma con 10 vicini in finestra su 20
   risorse totali e un subset abituale di ~metà delle autorizzate, il modello vede
   una fetta troppo corta della storia per distinguere abituale da non-abituale.
   È la leva più diretta (il laterale è puramente un problema di memoria): alzarlo
   a ~30–50 e salire un po' su `memory_dim`.

2. **⚠️ NON FARE: hard-negative ristretti agli autorizzati-ma-non-abituali.** Una
   versione precedente faceva esattamente questo (con peso ×10), «allenando il confine
   dove sta il laterale». Ma quel confine **è la definizione esatta** con cui il
   generatore crea il laterale di test: equivale ad addestrare sul test → recall
   gonfiata in modo **circolare** (è da qui che veniva il vecchio ~40%). È stato
   rimosso di proposito: i negativi strutturali sono ora a destinazione **casuale**,
   indipendenti da `auth_mask`/abitualità. Qualsiasi miglioria della recall laterale
   deve venire da architettura/segnale (punti 1, 3, 4, 5), **non** da negativi che
   rispecchiano la regola di test.

3. **Ripensare le due teste (l'ablation ribalta l'ipotesi iniziale).** Si supponeva che
   la `struct` head (coseno) portasse «tutto il segnale» laterale e che la `feat` head lo
   diluisse. L'ablation (vedi `inductive_testing.md`) dice il contrario: togliere la
   `struct` head **non** peggiora il lateral AUC (0.78 vs 0.74, entro la varianza), mentre
   togliere l'**hashed identity** lo fa crollare a 0.52. È la feature head *condizionata
   dall'identità hashata* a fare il lavoro. Quindi: la leva non è «far dominare la struct
   head», ma capire se la struct head serve davvero (un peso apprendibile tra le teste o un
   gating può dirlo) e investire sull'identità/storia. Da confermare su più seed.

4. **Loss di ranking invece di BCE per-coppia.** La BCE attuale spinge ogni coppia
   verso 0/1 in assoluto. Per la novità funziona meglio un obiettivo
   **contrastivo / ranking** (es. InfoNCE) che, *a parità di src*, spinga le risorse
   abituali sopra le non-abituali. Ottimizza direttamente "abituale > non-abituale"
   ed è allineato all'AP (metrica di ranking).

5. **Feature esplicita di novità/recency.** Oggi l'abitudine è solo *implicita* negli
   embedding. Il neighbor loader ha già la storia per nodo: derivarne una feature
   esplicita ("tempo dall'ultima volta che src ha toccato dst", "conteggio
   interazioni passate") inietterebbe il segnale vero in modo diretto.

6. **Verifica il cold-start.** Il modello può flaggare la novità solo *dopo* aver
   visto abbastanza storia dell'IP. Parte dell'AP basso potrebbe essere artefatto di
   nodi "freddi". Misurare il recall@thr escludendo i laterali su IP con poca storia
   dice se il problema è modellistico o solo di warm-up.

## Mosse consigliate per prime

**#1 (alzare `neighbor_size`)** e **#3 (peso apprendibile tra `feat` e `struct` head)**:
le più economiche e *non circolari*. Attaccano la causa radice — poca storia e una
feature head che diluisce il segnale strutturale — senza barare allenando sul confine
di test. Il punto #2 è esplicitamente da **evitare** (vedi avviso sopra).

## Nota sulla loss "alta" con buone prestazioni

La loss di training (~0.93) è quella del task auto-supervisato (link prediction con
negative sampling), *non* della detection. Il suo valore assoluto riflette l'entropia
irriducibile del predire il prossimo edge ed è poco correlato con AUC/AP, che sono
metriche di **ranking** (invarianti a trasformazioni monotone dello score). Loss alta
+ AUC alta è quindi atteso e non è un sintomo da inseguire.

---

### Stato implementato (de-circolarizzato)

Configurazione attuale (`src/config.py`) e relativo effetto onesto:
- **Storico ampio (`neighbor_size=30`, `memory_dim=256`, `num_hops=3`)**: amplia la
  finestra temporale/strutturale (leva #1). È la componente che porta il lateral AUC da
  0.59 (GNN statico, senza memoria) a 0.71 (TGN completo).
- **Hashed Identity deterministica (`hash_buckets=10000`, `hash_dim=16`, `stable_hash`)**:
  identità induttiva e coerente tra processi per ogni entità (anche le risorse), senza
  embedding transduttivi.
- **Negativi NON circolari**: strutturale a destinazione casuale + contestuale gaussiano,
  pesi uguali. La precedente penalizzazione ×5/×10 sull'hard-negative
  «autorizzato-non-abituale» è stata **rimossa** perché circolare (vedi punto 2): è la
  ragione per cui il recall laterale riportato è sceso dal ~40% gonfiato a ~16% onesto.
