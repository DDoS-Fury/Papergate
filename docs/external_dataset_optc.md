# Dataset esterno aggiuntivo: DARPA OpTC

Documento tecnico ad uso degli autori. Raccoglie l'esito della ricerca del 2026-09-19 di un
dataset pubblico da affiancare a PicoDomain (LANL escluso per scelta), le motivazioni, i rischi
noti e i controlli ancora da fare.

**Nessun risultato su OpTC esiste ancora.** Tutto ciò che segue riguarda la scelta e il
protocollo, non le prestazioni. Le informazioni sono marcate per grado di verifica (§9).

---

## 1. Perché serve un secondo dataset esterno

*   PicoDomain è l'unico dataset trovato che contiene insieme JA3, identità utente
    (Kerberos/NTLM) e lateral movement etichettato (`docs/paper/sections/external.tex`), ma è
    piccolo: 5 workstation + 1 DC, 2.67 giorni (`docs/data_validation_methodology.md` §3).
*   Un paper ANSSI del luglio 2026 (arXiv 2607.29390) lo liquida come dataset "sintetico basato
    su una rete molto piccola (solo sei host)". *Citazione da riassunto automatico: verificare
    sul testo.* È un'obiezione che un reviewer può riprendere.
*   UWF-ZeekData24 non è utilizzabile così com'è (§6).

Non è stato trovato nessun altro dataset pubblico con JA3 + utente + lateral movement etichettato.
La ricerca non è esaustiva.

## 2. Scelta: DARPA OpTC con il protocollo LMDEval

1.  **Scala e lateral movement reale.** Un periodo benigno seguito da tre giorni di red team
    (23–25 settembre). Giorno 1: staging PowerShell Empire con lateral movement e privilege
    escalation; giorno 2: exfiltration via Netcat/RDP; giorno 3: aggiornamento software
    malevolo. 292.367 eventi malevoli su ~17.4 miliardi (0.0016%).
2.  **Confronto con Euler.** Il paper 2607.29390 rivaluta Pikachu, Euler e Argus su OpTC con un
    protocollo che chiama "equo". Adottarlo dà un substrato comune con il competitor canonico
    (vedi memoria del progetto: Euler va citato e, idealmente, usato come baseline).
3.  **Protocollo pubblicato.** LMDEval (BSD-2-Clause, `github.com/cl-anssi/LMDEval`) fornisce
    estrazione e etichettatura di OpTC pensate per evitare le trappole di valutazione descritte
    dal paper (§5, punto 4).

Dati di base:

| Voce | Valore | Fonte |
|---|---|---|
| Formato / dimensione | JSON eCAR compresso, ~1 TB, su Google Drive | README OpTC-data |
| Eventi totali | ~17.4 miliardi; FLOW = 71.7% | 2103.03080 |
| Ground truth | `OpTCRedTeamGroundTruth.pdf` (documento, non etichette per evento) | README OpTC-data |
| Host | i dati raccolti coprono ~500 host su 1000 previsti (README); altre fonti dicono 1000 | *fonti discordanti* |
| Log di sistema | mancanti per circa metà delle workstation | 2607.29390 |
| Durata periodo benigno | discordante tra le fonti (19–23 sett. secondo 2103.03080; 9 giorni secondo 2607.29390) | *fonti discordanti* |

## 3. Mappatura sulla catena a 5 nodi

Campi dei FLOW (2103.03080): `start_time`, `end_time`, `src_ip`, `dest_ip`, `src_port`,
`dest_port`, `l4protocol`, `direction`, `image_path` (programma che apre il flusso), `size`.
Campi comuni eCAR (`ecar.md` del repo OpTC-data): `timestamp`, `hostname`, `objectID`, `object`,
`action`, `actorID`, `pid`, `tid`, `principal` (utente che esegue l'azione), `properties`.

| Nodo | Campo OpTC | Nota |
|---|---|---|
| source | `src_ip` | |
| device | `hostname` | |
| user | `principal` | **da verificare che non degeneri** (§5, punto 2) |
| resource | `dest_ip:dest_port` | |
| config | `image_path` | **proxy del JA3, non JA3**: dice quale programma apre la connessione, non quale stack TLS. Scelta di progetto da validare, non un fatto. |

## 4. Cosa OpTC può e non può dimostrare

*Può:* rilevamento induttivo/streaming su traffico enterprise di grande scala; abitudini
temporali principal/host → risorsa; confronto con Euler e con l'Isolation Forest corretta (§8);
tenuta fuori dal sintetico.

*Non può:*
*   **Il claim sul nodo config come JA3.** Resta verificabile solo su PicoDomain. La frase di
    `external.tex` ("un solo dataset pubblico fornisce l'intersezione richiesta") rimane vera per
    i 5 nodi completi; OpTC è una seconda istanza degradata.
*   **Il gate anti-poisoning con policy engine.** OpTC non ha policy: `roleVal`/`clrVal` neutri
    (come PicoDomain §3.2) e nessun segnale sensori, quindi `signal_dirty` non scatta mai e il
    gate di commit è commit-everything, sia per il TGN sia per le baseline (come su UWF).

## 5. Rischi e punti aperti

1.  **LMDEval scarta `principal` e `image_path`.** Letto in `extract_optc.py`: legge timestamp,
    `hostname`, `src_ip`, `dest_ip`, porte, `l4protocol`, `direction`; l'output ha colonne
    `timestamp, src, dst, src_port, dst_port, proto, label` (solo flussi host→host). Le etichette
    sono attaccate per ID evento (`optc_redteam.csv`), quindi si può estendere l'estrattore
    mantenendo le etichette.
2.  **`principal` sui FLOW potrebbe essere quasi sempre SYSTEM / account macchina.** Le fonti
    dicono che il campo esiste su ogni record eCAR, ma non è verificato che sia informativo sui
    FLOW START. Se degenera, il nodo user collassa e l'istanza è a 3 nodi: va dichiarato.
3.  **Numeri attesi modesti.** Secondo 2607.29390, sotto valutazione equa tutti i detector
    peggiorano molto (Tab. 5–6). Dal riassunto: Pikachu passa da ~99% a ~50.7% AUC sui soli
    lateral movement su OpTC — *cifra da verificare*. Va presentato con lo stesso framing onesto
    di PicoDomain.
4.  **Trappole di valutazione segnalate dal paper** (da evitare): preprocessing incoerente tra
    lavori; filtraggio irrealistico (togliere dal test gli utenti non coinvolti riduce
    meccanicamente i falsi positivi); metriche a livello arco invece che evento; **etichette OpTC
    ambigue**: nessuna ground truth per evento, e lavori precedenti hanno etichettato come lateral
    movement *tutti* i flussi uscenti da un host compromesso. LMDEval usa un'etichettatura ibrida
    (tracciamento dei processi + ispezione manuale).
5.  **Possibile via al JA3 vero (da esplorare).** 2103.03080 dice che i FLOW START rimandano ai
    record del sensore di rete Bro. Non è stato verificato se i log Bro/Zeek siano nella release
    né se contengano `ssl.log` con JA3.
6.  **Costo:** ~1 TB di JSON. Serve l'estrazione dei soli FLOW START prima di qualsiasi training.

## 6. Candidati scartati

| Dataset | Motivo |
|---|---|
| **UWF-ZeekData24** (così com'è) | Mapping degenere: `usr:`, `dev:`, `src:` sono lo stesso IP; `cfg:` è il campo `service` di Zeek, non un JA3; `ja3_valid`, `s1..s3`, `role`, `clr` costanti. Delle 45 colonne date all'IF, 32 sono identiche per ogni riga: ne restano 6 informative (method, bytes_in, dt_user, 3 contatori). Attacchi e benigni da cartelle diverse, timestamp degli attacchi riscalati nella finestra di test, benigni = primi 30k eventi. La classe "lateral" è Initial_Access + Defense_Evasion + Persistence + Privilege_Escalation, e la lista di download non contiene Lateral Movement. Non esercita nessuna delle novità del paper. |
| **LMDG** (arXiv 2508.02942) | Framework di generazione: 25 VM, 25 giorni, 944 GB, 35 attacchi multi-stadio, 22 account utente, etichettatura per albero dei processi. Il repo indicato (`github.com/WASPLab/LMTrace`) risultava **vuoto** il 2026-09-19: da ricontrollare. |
| **ATLASv2** (arXiv 2401.01341) | 2 VM Windows (Security, Sysmon, Firefox, DNS): più piccolo di PicoDomain. |
| **Multi-Source Cybersecurity Logs** (arXiv 2606.18190) | 870 sessioni da 20 minuti (2.3M eventi, 70 con attacco), CC BY 4.0; lateral movement solo nel 21% delle sessioni; nessuna baseline lunga: non adatto a un modello con memoria temporale. |

## 7. Piano operativo

1.  Copiare `extract_optc.py` di LMDEval (BSD-2: mantenere licenza e attribuzione), aggiungere
    `principal` e `image_path` all'output, tenere `optc_known_addresses.json` (IP → hostname) e la
    logica di etichettatura per ID evento.
2.  Prima di qualsiasi training: valori distinti e quota SYSTEM/account macchina di `principal`;
    valori distinti di `image_path`. Se `principal` degenera → istanza a 3 nodi, dichiararlo.
3.  Audit anti-scorciatoia, come per il sintetico (`tests/test_leakage_audit.py`): AUC per singola
    colonna (floor ≤ 0.60) e sovrapposizione delle entità (host, utenti, risorse) tra train e test.
    È la lezione di UWF.
4.  Split e metriche secondo LMDEval: metriche a livello evento. Lo split esatto per OpTC è nelle
    §§5.2–5.3 del paper (**non letto**).
5.  Baseline: Isolation Forest corretta (§8), static GNN, Euler (disponibilità del codice non
    verificata).
6.  Gate di commit: commit-everything oltre `val_end` per TGN e baseline (`label_horizon`).
7.  Riportare multi-seed e dispersione come nel resto del paper.

## 8. Questione collegata: oracolo nei contatori storici delle baseline

`causal_hist_features` (`src/eval_common.py`) faceva avanzare i contatori pair/src solo sugli
eventi con `y == 0` **anche nella slice di test**: le etichette del test entravano nelle
feature, e una coppia d'attacco restava "mai vista" per quante volte si ripetesse. Il TGN, in
valutazione, aggiorna invece su `not signal_dirty` (`src/train_tgn.py`, proxy dell'OPA-ALLOW),
che su UWF è sempre vero.

**Stato al 2026-09-19** (non eseguito su dati reali):

| Punto di chiamata | Stato |
|---|---|
| `tests/eval_uwf_baseline.py` | corretto (`label_horizon=val_end`); test in `tests/test_eval_common.py` |
| `tests/baselines/isolation_forest/isolation_forest_baseline.py` (**l'IF della tabella baseline del paper**) | da correggere |
| `tests/baselines/ocsvm/ocsvm_baseline.py` | da correggere |
| `tests/baselines/xgboost/xgboost_baseline.py` | da correggere |
| `tests/baselines/simple_gnn/simple_gnn_baseline.py` | da correggere |
| `tests/eval_picodomain_iforest.py` | da correggere |

*   `label_horizon=None` (default) mantiene il comportamento precedente, quindi i numeri già
    pubblicati non cambiano finché i punti sopra non vengono aggiornati.
*   **Sul flusso sintetico il gate di parità col TGN è "aggiorna se non `signal_dirty`", non
    "aggiorna tutto"**: serve un parametro analogo a `pred` di `causal_src_seen`. Da decidere
    prima di applicare il fix lì.
*   Effetto atteso sulla IF sintetica: uguale o peggiore (lateral AUC già 0.537, sotto il floor
    0.567), quindi il claim del paper ne esce rafforzato. È un'attesa, non una misura.
*   Finché non è sistemato, la frase di `docs/paper/sections/setup.tex` sui contatori "causal
    benign-gated" dati a tutte le baseline non è del tutto esatta. Va corretta o va corretto il
    codice, prima di rigenerare la tabella.
*   **UWF dopo il fix resta non affidabile.** È stato tolto l'oracolo sulle etichette, non il
    sospetto di entità disgiunte (IP degli attaccanti assenti nel benigno). Controlli da fare
    sulla workstation: intersezione di `src_ip_zeek` e `dest_ip:port` tra `Benign/` e le cartelle
    degli attacchi; AUC per singola colonna; IF senza i 3 contatori.

## 9. Stato delle verifiche

*   **Verificato leggendo il codice:** tutto il §6 su UWF, il §8, la struttura di
    `extract_optc.py` (campi letti/scartati, colonne d'uscita, etichettatura per ID).
*   **Da riassunti automatici di pagine web, da controllare sul testo originale:** i contenuti di
    2607.29390 (citazione su PicoDomain, cifre delle Tab. 5–6, trappole di valutazione,
    etichettatura ibrida), i campi FLOW e le percentuali di 2103.03080, il README di OpTC-data, i
    dettagli di LMDG, ATLASv2 e 2606.18190.
*   **Non verificato:** popolamento di `principal` sui FLOW; presenza di log Bro/Zeek con JA3 nella
    release OpTC; disponibilità del codice di Euler; lo split raccomandato per OpTC.
*   Le slide INRIA "A New Hope for DARPA OpTC" (Majorczyk, Pilastre, Dijoud, ACSAC CSET 2025) sono
    state viste solo nelle prime pagine: da leggere per l'etichettatura di OpTC.

## Fonti

*   On Fair and Realistic Performance Evaluations for Graph-Based Lateral Movement Detectors,
    arXiv 2607.29390 — https://arxiv.org/html/2607.29390
*   LMDEval — https://github.com/cl-anssi/LMDEval
*   OpTC-data (README, `ecar.md`) — https://github.com/FiveDirections/OpTC-data
*   Analyzing the Usefulness of the DARPA OpTC Dataset in Cyber Threat Detection Research,
    arXiv 2103.03080 — https://ar5iv.labs.arxiv.org/html/2103.03080
*   A New Hope for DARPA OpTC (ACSAC CSET 2025) —
    https://www.acsac.org/2025/workshops/cset/proceedings/Majorczyk-ANewHopeForDARPAOpTC-2025-12-08.pdf
*   LMDG, arXiv 2508.02942 — https://arxiv.org/abs/2508.02942
*   ATLASv2, arXiv 2401.01341 — https://arxiv.org/pdf/2401.01341
*   Multi-Source Cybersecurity Logs, arXiv 2606.18190 — https://arxiv.org/html/2606.18190v1
