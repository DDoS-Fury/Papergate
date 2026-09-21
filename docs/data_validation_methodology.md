# Metodologia di Validazione Scientifica dei Dati

Documento tecnico ad uso degli autori per la redazione, revisione e difesa del manoscritto.  
Sintetizza i criteri metodologici con cui sono stati convalidati i dati di addestramento e test (benchmark sintetico e traccia reale PicoDomain), specificando la portata dei risultati e la corretta interpretazione dei valori empirici.

---

## 1. Premessa e Posizionamento Epistemologico

Nella letteratura di intrusione di rete e sicurezza (*Sommer & Paxson*, IEEE S&P 2010; *Arp et al.*, USENIX Security 2022), la valutazione basata esclusivamente su simulatori proprietari è soggetta a forte scetticismo accademico per via del rischio di *Author Bias* e *Shortcut Learning* (il modello impara artefatti introdotti dal generatore piuttosto che correlazioni causali).

Al contempo, non esiste in letteratura alcun benchmark pubblico specifico per architetture Zero Trust (ZTA) che fornisca contestualmente:
1. Catena causale a 5 nodi (`source_ip -> config (JA3) -> device -> user -> resource`);
2. Fingerprint TLS del client (JA3) associati a identità di autenticazione Kerberos/NTLM;
3. Tracciamento ground-truth di lateral movement e credential theft.

I benchmark pubblici di autenticazione (es. LANL Cyber1, DARPA OpTC) omettono i fingerprint TLS; i benchmark di traffico di rete (es. CIC-IDS2017) omettono le identità utente o il lateral movement etichettato.

La strategia di validazione adotta pertanto una **struttura a due binari complementari**:
*   **Validità Interna (Benchmark Sintetico):** Ambiente controllato controfattuale per l'ablazione dei componenti, validato matematicamente contro leakage e scorciatoie univariati.
*   **Validità Esterna (PicoDomain):** Case study empirico su telemetria Zeek reale di un'infezione enterprise, a dimostrazione della fattibilità dello schema a 5 nodi su dati non simulati.

---

## 2. Validazione del Benchmark Sintetico

Il generatore primario (`src/data/stream_synthetic.py`) è formalmente vincolato per eliminare separabilità banali.

### 2.1 Audit di De-Leakage (`tests/test_leakage_audit.py`)
L'audit gira sullo stream di training (200k eventi, 3 seed). Impone sei invarianti:
1.  **Nessuna scorciatoia su feature singola ($\text{AUC} \le 0.75$):**
    nessuna colonna scalare in ingresso, cioè il messaggio o le feature statiche di tutti e
    cinque i nodi, separa da sola una classe oltre la soglia. Fanno eccezione i segnali
    allow-listati per design (sonde IDS sul recon, volumi sull'exfil, RISK della risorsa sulle
    violazioni di policy). Lateral movement e credential theft non hanno eccezioni.
    Il floor misurato sul lateral va rimisurato sul generatore v5: il valore 0.567 della
    Tabella I si riferisce al generatore v4.
2.  **Nessuna scorciatoia storica su lookup singolo ($\text{AUC} \le 0.85$, v5):**
    nessuna regola di set-membership («IP mai visto», «coppia config→utente mai vista»,
    «claim di ruolo diverso dal solito», …) separa lateral o theft. Le regole sono in
    `graphagate.data.lookup_rules`. Seguono il protocollo del paper: memoria dei benigni
    etichettati prima della finestra di test, poi dei soli eventi predetti benigni.
    Sul generatore v4 il lookup «IP mai visto» raggiungeva AUC 1.000 sul theft. La loro
    somma (baseline *stateful*) è il riferimento che i modelli appresi devono battere
    (`tasks/runs/generator_rule_audit.log`).
3.  **Claim di ruolo coerente con l'identità:** il ruolo nel messaggio è sempre quello
    reale dell'utente. Nel v4 il ruolo era falsificato nel 50% del lateral: un canale
    senza falsi positivi.
4.  **Misurabilità:** ogni classe ha almeno 30 eventi. Lateral e theft ne hanno almeno 100
    nella finestra di test, così nessun controllo passa per assenza di campioni.
5.  **Invarianza marginale delle destinazioni (KS sull'effect size, $D_{\text{KS}} \le 0.15$)**
    e coppie (route, metodo) sempre servite.
6.  **Assenza di impronte costanti:** volumi e $\Delta t$ sono estratti da distribuzioni
    continue. Il messaggio contiene solo segnali disponibili al PDP prima della risposta.

### 2.2 Modello del traffico (v5) e scelte parametriche
I valori sono scelte di modellazione, esposte come parametri in `TGNConfig` e ablabili
con `scratch/knob_ablation.py`. Non sono derivati da uno standard.
*   **Risorse:** legge di potenza sul rango di popolarità con esponente $s = 1.2$. Il rango è
    permutato rispetto all'indice della risorsa. Breslau et al. (INFOCOM 1999) riportano
    $\alpha \approx 0.64$–$0.83$ per richieste a proxy web, quindi $1.2$ è una
    concentrazione più forte, non un valore di letteratura.
*   **Inter-arrivi:** processo di Poisson a tasso costante a tratti (ore lavorative, notte,
    weekend). Paxson & Floyd (ToN 1995) sostengono il modello di Poisson per gli arrivi di
    *sessione* utente, non per le singole richieste: è un'approssimazione dichiarata.
*   **Mondo aperto (v5):** la novità è un evento benigno comune.
    *   IP mai visti nel roaming (`p_new_source`).
    *   Release di client che cambiano il JA3 della flotta (`p_config_release`).
    *   Hot-desking, cioè un utente su una macchina non sua (`p_hotdesk`).
    *   Cancellazione dei cookie (`p_cookie_wipe`).
    *   Falsi positivi IDS (`p_sensor_fp`) e client legacy senza JA3 (`p_legacy_client`).

    Attaccanti e benigni prendono i nodi nuovi dallo *stesso* pool, con lo stesso formato di
    chiave.
*   **Attaccante mimetico (theft):** usa un client comune della flotta, esce da indirizzi già
    usati dalla flotta o riusa la sessione rubata (pass-the-cookie, con il fingerprint della
    vittima).
*   **Lateral movement (v5):** la macchina compromessa usa credenziali raccolte di un altro
    utente (nuovo binding device→utente, come in Euler/LANL), oppure esegue accessi non
    abituali del proprietario.
*   **Tasso base:** intrusioni a tasso globale (`p_compromise`) con kill chain finita e
    remediation. La prevalenza degli attacchi è circa 1.3–1.5% per seed (etype 1–5), stabile lungo lo
    stream. Nel v4 era circa 28% e alla fine 76 macchine su 80 risultavano compromesse.
*   **Cosa non modelliamo:** timeout di sessione Kerberos/TLS e lease DHCP (RFC 4120, 8446,
    2131) non sono implementati nel generatore. Il TTL di 10 ore della §3 riguarda il
    binding in serving su PicoDomain.

---

## 3. Validazione Empirica su Telemetria Reale (PicoDomain)

PicoDomain (Laprade et al., 2020) è una cattura Zeek/Security Onion di 2.67 giorni su un dominio Active Directory (5 workstation, 1 Domain Controller) con attività di red team documentata.

### 3.1 Modalità di Esecuzione del Test
È fondamentale chiarire come il modello opera su PicoDomain per evitare obiezioni metodologiche:
*   **Non è un trasferimento zero-shot di pesi:** Le memorie recurrent del TGN tracciano entità specifiche; i pesi addestrati su ID sintetici non possono essere applicati direttamente a nomi di dominio reali.
*   **È la pipeline di apprendimento che viene convalidata:** La medesima architettura a 5 nodi e gli identici iperparametri di default (`TGNConfig()`) vengono istanziati ed addestrati sulla fetta iniziale di PicoDomain (primo 70% degli eventi, 100% benigno). Il test avviene in modalità one-class sulla porzione finale contenente l'intrusione.
*   **Assenza di ri-tuning:** Nessun iperparametro (learning rate, dimensione embedding, pesi della loss) è stato ottimizzato ad-hoc per PicoDomain.

### 3.2 Assenza di Policy ZTA nel Dataset Reale
PicoDomain non contiene definizioni formali di policy autorizzative (niente ruoli RBAC o livelli Bell-LaPadula).
Di conseguenza:
*   Le feature di ruolo e clearance sono impostate a valori neutri (`roleVal=0, clrVal=0`).
*   La classe *Policy Violation* non è valutabile ed è esclusa dall'analisi.
*   Il modello opera come **rilevatore di anomalie puramente strutturali e temporali**.

### 3.3 Interpretazione Rigorosa di AUC 0.640 e 0.696
A prima vista, un'AUC di 0.6402 sul lateral movement può apparire modesta se confrontata con benchmark supervisionati. Nel contesto di security anomaly detection su tracce reali, il valore possiede un significato specifico:

1.  **Condizione One-Class Pura:** Il modello non ha mai osservato alcun esempio di attacco durante il training ($n_{\text{attack}} = 0$).
2.  **Traccia Signal-Clean (Assenza di Sonde IDS):** In PicoDomain i campi di allarme sensore sono nulli ($s_1=s_2=s_3=0$). La baseline a regole ottiene:
    $$\text{Precision} = 0.0000, \quad \text{Recall} = 0.0000, \quad \text{AUC} = 0.5000$$
    I sistemi basati su firme sono del tutto ciechi.
3.  **Comandi "Living off the Land":** Il lateral movement reale impiega WMI, DCOM, PsExec e chiamate RPC/SMB. Singolarmente, ciascuna richiesta è sintatticamente identica al normale traffico di sistema Windows. Non vi è payload anomalo né spike volumetrico.
4.  **Scala Temporale Ridotta (2.67 giorni, 8 IP, 15 utenti):** L'orizzonte temporale è breve per consentire a una GNN dinamica di stabilizzare una memoria a lungo termine delle abitudini degli utenti.
5.  **Segnale del Nodo Configurazione (AUC 0.6957):** Sulla classe *Credential Theft*, l'AUC raggiunge **0.6957** (~0.70). Questo conferma empiricamente l'ipotesi centrale del paper: il disallineamento nel binding `config (JA3) -> user` fornisce un segnale discriminante quando credenziali lecite vengono utilizzate tramite tool estranei (es. Mimikatz o beacon).

**Sintesi:** AUC 0.640 non dimostra che il sistema è pronto per un blocco autonomo in produzione, ma prova che in condizioni ostili e prive di firme, il grafo temporale estrae un segnale di ranking statisticamente superiore al caso ($0.64 > 0.50$).

### 3.4 Analisi di Sensibilità del Binding TTL
Poiché in Zeek `ssl.log` e `kerberos.log` hanno identificatori di connessione disgiunti (overlap `uid` = 0), l'associazione temporale `config -> user` è governata dal parametro $\Delta t_{\text{bind}}$:
*   $\tau = 900\text{ s}$ (15 min): copertura utente $11.8\%$ (troppo restrittivo);
*   $\tau = 3600\text{ s}$ (1 ora): copertura utente $29.0\%$;
*   $\tau = 36000\text{ s}$ (10 ore, default Kerberos TGT RFC 4120): copertura utente **$90.2\%$** (con device coverage al $97.5\%$).

Dimostrare questa curva nel paper prova che il parametro discende dallo standard Kerberos e non da un tuning opportunistico. Gli eventi non associati sono isolati tramite nodi sentinella per-IP (`sentinel_user_<IP>`), impedendo la formazione di hub artificiali.

---

## 4. Linee Guida per la Stesura del Testo e la Difesa con i Reviewer

### 4.1 Cosa NON affermare
*   Non affermare che il generatore sintetico "rappresenta fedelmente la complessità di una rete enterprise".
*   Non affermare che il modello addestrato sul sintetico è stato "trasferito senza modifiche su PicoDomain ottenendo 0.64".
*   Non presentare PicoDomain come un "benchmark quantitativo di scala".

### 4.2 Cosa affermare (Formulazione Consigliata)
1.  *Sul sintetico:* «Data l'assenza di benchmark pubblici ZTA a 5 nodi, utilizziamo un generatore streaming vincolato formalmente da un leakage audit automatico, dimostrando che il task di rilevamento del lateral movement non è risolvibile tramite scorciatoie univariate, lookup storici singoli o invarianze marginali, e riportando la baseline a regole stateful come riferimento.» *(Floor da rimisurare sul generatore v5.)*
2.  *Su PicoDomain:* «PicoDomain funge da studio empirico di fattibilità: dimostra che la catena a 5 nodi si mappa direttamente su telemetria Zeek reale (JA3 + Kerberos) e che la pipeline di apprendimento temporale, istanziata senza tuning per-corpus, preserva capacità induttiva di ranking (AUC 0.640 su lateral, 0.696 su theft) laddove i rilevatori a firme falliscono interamente.»
3.  *Sulle limitazioni:* Dichiarare con trasparenza la dispersione dei seed ($\text{ddof}=1$), l'impossibilità di significatività asintotica con $N=3$ (riportando il pattern di segno di Wilcoxon), e la necessità di istanziare policy formali per calcolare metriche di recall a soglia.

---

## 5. Riferimenti Bibliografici Chiave da Includere

*   **R. Sommer, V. Paxson**, *"Outside the Closed World: On Using Machine Learning for Network Intrusion Detection"*, IEEE S&P 2010. *(Motivazione del semantic gap e della non-trivialità).*
*   **D. Arp et al.**, *"Dos and Don'ts of Machine Learning in Computer Security"*, USENIX Security 2022. *(Linee guida metodologiche contro lo shortcut learning).*
*   **S. Axelsson**, *"The Base-Rate Fallacy and the Difficulty of Intrusion Detection"*, ACM TISSEC 2000. *(Giustificazione dell'uso di AP/PR-AUC e partial-AUC rispetto a ROC-AUC).*
*   **L. Breslau et al.**, *"Web Caching and Zipf-like Distributions"*, IEEE INFOCOM 1999. *(Grounding della distribuzione delle risorse).*
*   **V. Paxson, S. Floyd**, *"Wide-Area Traffic: The Failure of Poisson Modeling"*, IEEE/ACM ToN 1995. *(Grounding del traffico non-omogeneo circadiano).*
*   **C. Laprade, B. Bowman, H. H. Huang**, *"PicoDomain: A Compact High-Fidelity Cybersecurity Dataset"*, arXiv:2008.09192, 2020. *(Riferimento del dataset reale).*
*   **IETF RFC 4120**, *"The Kerberos Network Authentication Service (V5)"*, 2005. *(Giustificazione del binding TTL a 36.000 s).*
