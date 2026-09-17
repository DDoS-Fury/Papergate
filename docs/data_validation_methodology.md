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
La suite di audit automatico impone 4 invarianti di non-trivialità:
1.  **Floor AUC su Feature Singola ($\le 0.60$ su Lateral Movement):**  
    Nessuna colonna scalare in ingresso (messaggio o attributi dei nodi) può separare da sola il lateral movement oltre $\text{AUC} \le 0.60$. Il valore post-audit misurato è **0.567** (Tabella I del paper), prossimo alla distribuzione casuale (0.50). Ciò garantisce che il task sia risolvibile solo modellando le correlazioni relazionali e temporali del grafo.
2.  **Invarianza Marginale delle Destinazioni (Test KS su Effect Size):**  
    Il traffico di lateral movement campiona le risorse bersaglio dalla stessa legge di popolarità dell'esplorazione benigna. La divergenza è valutata tramite statistica di Kolmogorov-Smirnov sull'effect size ($D_{\text{KS}} \le 0.15$), impedendo che la "novità della destinazione" costituisca un'etichetta gratuita.
3.  **Assenza di Impronte Costanti:**  
    I campi volumetrici (`bytes_in`, `bytes_out`) e temporali ($\Delta t$) sono estratti da distribuzioni continue, evitando costanti per-classe che permetterebbero la memorizzazione esatta.
4.  **Causalità Inline (Zero Response-Side Leakage):**  
    Il vettore di messaggio (10 float) include esclusivamente segnali osservabili al momento dell'arrivo della richiesta al PDP (JA3, sonde IDS, metodo, credenziali, recency utente). I campi di risposta (es. status code HTTP o byte inviati dal server) sono rigorosamente esclusi per evitare violazioni di causalità.

### 2.2 Grounding dei Parametri su Standard
I parametri del generatore non sono arbitrari:
*   **Risorse:** Legge di Zipf-Mandelbrot con esponente $s \approx 0.9$, coerente con la letteratura sugli accessi a intranet e repository aziendali (*Breslau et al.*, INFOCOM 1999).
*   **Tempi di inter-arrivo:** Processo di Poisson non-omogeneo (NHPP) con cicli circadiani diurni (8:00–18:00) e traffico batch notturno (*Paxson & Floyd*, ToN 1995).
*   **Sessioni e Timeout:** Ancorati a standard IETF: Kerberos TGT lifetime di 10 ore (RFC 4120), TLS session parameters (RFC 8446), DHCP lease churn (RFC 2131).
*   **Rumore benigno:** Inclusione di esplorazione non-abituale legittima, errori umani (richieste 403) e roaming di rete per chiudere il *semantic gap*.

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
1.  *Sul sintetico:* «Data l'assenza di benchmark pubblici ZTA a 5 nodi, utilizziamo un generatore streaming vincolato formalmente da un leakage audit automatico, dimostrando che il task di rilevamento del lateral movement non è risolvibile tramite scorciatoie univariata (floor AUC 0.567) o invarianze marginali.»
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
