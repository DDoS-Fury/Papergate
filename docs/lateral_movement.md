# Movimenti laterali: cosa sono e come migliorarne la detection

Note di analisi sul caso anomalia più difficile del modello TGN (al momento
AUC ~0.89 ma **AP ~0.20**, contro AUC ~1.0 di `policy` e `contextual`).

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
allenata con *hard negative* = src accoppiato a una risorsa non abituale
(`train_tgn.py`). Il design è corretto, ma diverse leve sono sottosfruttate.

## Leve di miglioramento (ordinate per impatto/sforzo)

1. **`neighbor_size = 10` è probabilmente il collo di bottiglia.** L'abitudine è una
   proprietà di *insieme* sulla storia dell'IP, ma con 10 vicini in finestra su 20
   risorse totali e un subset abituale di ~metà delle autorizzate, il modello vede
   una fetta troppo corta della storia per distinguere abituale da non-abituale.
   È la leva più diretta (il laterale è puramente un problema di memoria): alzarlo
   a ~30–50 e salire un po' su `memory_dim`.

2. **Mismatch tra hard-negative di training e laterale di test.** In training
   l'hard negative è pescato tra *tutte* le risorse non abituali — incluse quelle
   **non autorizzate**, già coperte dai negativi `policy` (facili). Il laterale di
   test è invece non-abituale **ma autorizzato**. Restringere gli hard negative alle
   sole risorse *autorizzate-ma-non-abituali* allena il confine esattamente dove sta
   il laterale, invece di sprecare gradiente su casi facili.

3. **La feature head sporca il segnale.** Lo score è `feat + struct` a peso uguale.
   Per il laterale la `feat` head vede feature identiche al benigno → contributo
   quasi costante/rumoroso che diluisce la `struct` head (che porta *tutto* il
   segnale). Un peso apprendibile tra le due teste, o un gating, lascerebbe la testa
   strutturale dominare quando le feature non dicono nulla.

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

**#1 (alzare `neighbor_size`)** e **#2 (hard negative ristretti agli
autorizzati-non-abituali)**: le più economiche, attaccano la causa radice — il
modello non ha abbastanza storia e non viene allenato esattamente sul confine che
poi deve giudicare.

## Nota sulla loss "alta" con buone prestazioni

La loss di training (~0.93) è quella del task auto-supervisato (link prediction con
negative sampling), *non* della detection. Il suo valore assoluto riflette l'entropia
irriducibile del predire il prossimo edge ed è poco correlato con AUC/AP, che sono
metriche di **ranking** (invarianti a trasformazioni monotone dello score). Loss alta
+ AUC alta è quindi atteso e non è un sintomo da inseguire.

---

### Aggiornamento Soluzione (Implementata)

Per risolvere queste criticità e abbattere il gap del Lateral Movement, sono state applicate le seguenti ottimizzazioni al modello:
- **Aumento Storico (`neighbor_size=30`, `memory_dim=128`)**: Risolve il problema #1 ampliando significativamente la finestra temporale di esplorazione, aiutato anche dall'architettura multi-hop (`num_hops=2`).
- **Hashed Identity (`hash_buckets=10000`, `hash_dim=16`)**: Anziché usare embedding transduttivi inefficaci sui nodi nuovi, l'identità viene fornita ai Layer tramite un hash dell'URI in modo totalmente induttivo. Questo permette alla rete GNN di raggruppare i nodi in base ai comportamenti storici consolidati, portando il Lateral Movement Recall dal <1% a quasi il **19%** nei test asincroni, pur mantenendo scalabilità estrema.
- **Penalizzazione Hard Negatives**: La loss per i falsi laterali (hard struct negatives) è stata pesata x5, forzando la classificazione sulle anomalie strutturali.
