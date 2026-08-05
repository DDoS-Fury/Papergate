# Dataset esterni per la valutazione di Graphagate

> Selezione di un dataset **pubblico e reale** con struttura paragonabile alla catena causale
> a 5 nodi `source (IP) → config (JA3) → device → user → resource`.
> Ricerca bibliografica: 2026-08-05. Ogni URL qui sotto è stato risolto.
> Misure su PicoDomain: eseguite, comando riportato — nessun numero è stimato.

## TL;DR

- **Non esiste** alcun dataset pubblico costruito per la Zero Trust / continuous authorization.
- **Nessun dataset pubblico consegna insieme identità utente e fingerprint TLS** come campi
  già presenti. L'unica eccezione trovata è **PicoDomain**, dove `ssl.log` porta la colonna
  `ja3` e `kerberos.log`/`ntlm.log` portano le identità di dominio — ma su connessioni
  diverse, quindi il legame va ricostruito (vedi §PicoDomain).
- **Scelta**: **PicoDomain** come caso di studio su dati reali (adapter implementato e
  verificato), **stream sintetico** come benchmark principale, **AIT Log Data Set V2.0**
  come lavoro futuro per una valutazione full-chain su scala.
- **LANL** non viene usato come benchmark alla pari ma discusso come *ablazione del nodo
  config*: lì il nodo config degenera e la classe credential-theft non è valutabile.

---

## 1. Il gap

Le rassegne sistematiche sulla ZTA (2016-2025) confermano che i lavori su trust-score e PDP
sono valutati su **simulazioni bespoke non rilasciate**. IEEE DataPort non ha dataset
zero-trust pertinenti. Questo è il motivo per cui la valutazione principale poggia su uno
stream sintetico, e va **dichiarato nel paper come motivazione**, non nascosto.

Il secondo gap è più stretto e più interessante: il nodo *configuration* esiste perché
"tool nuovo su device noto" (furto di credenziali riusate da un client diverso) non è
esprimibile in un grafo di autenticazione host→host. Ma i dataset che hanno le **label** di
lateral movement (LANL, OpTC) **non hanno alcun fingerprint del client**, e i dataset che
hanno il fingerprint (o il pcap da cui ricavarlo) in genere **non hanno le label**. La
letteratura vicina non ha questo problema perché non modella quel nodo.

## 2. Shortlist

| Dataset | 5 nodi popolabili | Label lateral | Accesso | Verdetto |
|---|---|---|---|---|
| **PicoDomain** (2020) | **5/5** (ja3 reale) | red log, ±1 min | GitHub, aperto, 16 MB | ✅ **scelto** — caso di studio |
| AIT Log Data Set V2.0 (2022) | 5/5 (ja3 da pcap) | ❌ (scan→webshell→cracking→exfil) | Zenodo, CC BY-NC-SA, 130 GB | 🔜 lavoro futuro |
| LANL 2015 | 3/5 | ✅ red team | form email → token | ⚠️ ablazione "senza config" |
| DARPA OpTC | 4/5 (no ja3) | narrativa in PDF | Drive, aperto, ~1 TB | ⏳ costo di preprocessing |
| CMU CERT r4.2 | 3/5 (no IP, no config) | ❌ (solo insider/exfil) | Figshare | ❌ classi 2/3/4 non valutabili |
| CIC-IDS2017 / 2018 | 3/5 (**no utente**) | ❌ (Infiltration ≠ lateral) | UNB, aperto + pcap | ❌ senza identità non è ZTA |
| Unraveled (2023) | 5/5 | ✅ stage APT incl. lateral | GitLab; mirror S3 **403** | ⚠️ dati bulk non ottenibili |
| CAM-LDS (2026) | 5/5 | ✅ ATT&CK, scenario 3 = lateral | Zenodo, 621 GB | ❌ **nessun traffico benigno** |
| LANL Unified 2018 | 4/5 | ❌ **nessun ground truth** | form email → token | ❌ non etichettato |
| UNSW-NB15 / CIDDS / LID-DS | ≤2/5 | ❌ | aperto | ❌ struttura incompatibile |
| LMDG (2025) | — | ✅ | repo **vuoto** | ❌ non scaricabile |
| Splunk BOTSv3 | 5/5 (`stream:tls` ha ja3) | ❌ (CTF, answer key) | S3, aperto | 📌 solo come esistenza-proof |

### Perché non le alternative "ovvie"

- **CIC-IDS2017** ha il pcap e quindi JA3 veri, ma **non ha alcun livello di identità**: il
  nodo `user` non è popolabile. In un modello ZTA è fatale.
- **CMU CERT** è l'unico che porta nativamente ruolo e unità organizzativa (i nostri
  attributi statici), ma non ha IP né fingerprint e **non modella affatto il lateral
  movement**: nessun attaccante si muove tra host.
- **CAM-LDS** ha lo scenario 3 dedicato al lateral movement con label a livello di tecnica
  ATT&CK, ma per esplicita ammissione degli autori non contiene comportamento benigno
  simulato — inutilizzabile per un training one-class. Resta il candidato migliore come
  *sorgente di iniezione* sopra il traffico benigno di AIT (stessa istituzione, stesso
  testbed, stessi formati).
- **AIT v2** è il vero obiettivo a medio termine: 8 testbed indipendenti = 8 stream
  cronologici, cioè uno studio di generalizzazione cross-testbed che nessuno tra
  Euler/Argus/LMDetect può offrire. Costa 130 GB e un join pcap→JA3 da costruire.

## 3. PicoDomain — il dataset scelto

| | |
|---|---|
| Paper | Laprade, Bowman, Huang, *PicoDomain: A Compact High-Fidelity Cybersecurity Dataset* (2020) |
| Download | https://github.com/iHeartGraph/PicoDomain (`Zeek_Logs.7z` 16 MB, `Red Log.xlsx`) |
| Licenza | aperta, nessuna registrazione |
| Periodo | 2019-07-18T23:44 → 2019-07-21T15:59 UTC (**2.67 giorni**) |
| Ambiente | dominio Windows: 5 workstation + DC + gateway, Security Onion 16.04, Zeek in JSON UTC |
| Log presenti | `conn, dce_rpc, dhcp, dns, files, http, kerberos, known_certs, known_hosts, known_services, ntlm, pe, rdp, smb_files, smb_mapping, software, ssl, weird, x509` |

### 3.1 Il punto che decide: `ja3` c'è davvero

Misurato, non assunto:

```
ssl.log:      3433 record, 3433 con ja3 (100%), 12 fingerprint JA3 distinti, 9 ja3s
http.log:     38 920 record, 12 user_agent distinti (fallback disponibile)
kerberos.log: 4686 record, 1097 con `client` (23.4%), 29 principal distinti
ntlm.log:     1686 record, 5 username, 5 hostname
conn.log:     257 540 record, 14 IP sorgente distinti
```

`kerberos.log:client` porta sia utenti (`jdoe/G.LAB`) sia **account macchina**
(`HR-WIN7-1$/G.LAB`), cioè utente **e** device da un solo campo. Il realm compare come
`G.LAB`, `g.lab` e troncato `G` nella stessa cattura: senza normalizzazione una sola entità
occuperebbe tre slot di memoria.

### 3.2 Mapping sui 5 nodi

| nodo | campo Zeek | qualità |
|---|---|---|
| `source` | `conn`/`http`/`smb_*`/`dce_rpc` → `id.orig_h` | ✅ reale, RFC1918 calcolabile |
| `config` | `ssl.log` → `ja3` | ✅ **reale**, 12 fingerprint |
| `device` | `kerberos.log` account macchina `HOST$`, `ntlm.log` → `hostname` | ✅ reale, 5 host |
| `user` | `kerberos.log` → `client` (esclusi `$`), `ntlm.log` → `username` | ✅ reale, ~7 utenti umani |
| `resource` | `smb_mapping:path`, `smb_files:path/name`, `http:host+uri`, `dce_rpc:endpoint.operation` | ✅ 569 oggetti distinti |

Attributi statici (ruolo, clearance, tier, sensibilità della risorsa): **assenti**, messi a
zero. Solo lo slot 5 (internal/external) è popolato, perché deriva dall'IP stesso.

### 3.3 Il limite vero: non esiste una chiave di join di sessione

`uid` è l'identificatore di **connessione** Zeek. Misurato: l'intersezione degli `uid` fra
`ssl` e `kerberos` è **0**, e fra `ssl` e `ntlm` è **0** — l'handshake TLS e
l'autenticazione avvengono su connessioni diverse. Quindi il legame `config → user` **non è
osservato**: va ricostruito per-IP con una finestra temporale (*last observed within TTL*).

TTL di default: **36 000 s (10 h)**, la durata predefinita di un TGT Kerberos, cioè il
periodo per cui il dominio stesso considera valida una sessione di logon. Scelto per quella
ragione, non ottimizzato su una metrica — ma la copertura dipende da esso e la dipendenza va
riportata:

| `bind_ttl` | copertura `user` | `device` | `config` |
|---|---|---|---|
| 900 s | 11.8% | 86.6% | 87.5% |
| 3 600 s | 29.0% | 89.7% | 90.6% |
| **36 000 s** | **90.2%** | **97.5%** | **94.8%** |

Gli eventi non attribuibili cadono su sentinelle **per-IP** (`usr:none:<ip>`, …), mai su un
nodo "unknown" condiviso: un evento non attribuito non deve ereditare la memoria di
un'altra entità.

### 3.4 Ground truth

`Red Log.xlsx` registra `(timestamp, victim host, victim user, azione)` con accuratezza
dichiarata **±1 minuto**. La kill chain è completa e pertinente: archivio troianizzato →
Hot Potato / bypass UAC → **Mimikatz + powerdump** → enumerazione del dominio → **esecuzione
remota via DCOM e WMIc** su HR-WIN7-2, RND-WIN10-1, RND-WIN10-2 → persistenza → Meterpreter
come `local.admin`. Cioè: **lateral movement e riuso di credenziali entrambi presenti**.

Il testo dell'azione è mappato sulla tassonomia del progetto da una tabella di regex
(`_ACTION_CLASS` in `tests/datasets/picodomain.py`); un evento di accesso è etichettato se
cade entro `label_window` (default ±90 s) da una riga del red log **e** ne condivide host o
utente. È un'etichettatura **approssimata di un log scritto a mano** e va riportata come tale.

### 3.5 Stream risultante (misurato)

```
$ python -m tests.datasets.picodomain --log-dir data/logs --red-log "data/Red Log.xlsx"
[picodomain] events=55436 nodes=625 (users=15 devices=13 sources=8 configs=20 resources=569)
[picodomain] binding coverage: user=90.2% device=97.5% config=94.8%  ttl=36000s
[picodomain] labels: benign=54566 contextual=412 lateral=356 theft=102  window=±90s
StreamData ready: (55436, 10) msg, span=230374s (2.67 d), anomalous fraction = 0.0157
```

### 3.6 Limitazioni da dichiarare nel paper

1. **Scala**: 8 IP sorgente, 5 device, ~7 utenti, 2.67 giorni. È un **caso di studio**, non
   un benchmark titolare. La varianza su 356 eventi laterali è alta.
2. Il binding `config → user` è **ricostruito**, non osservato (§3.3). Ogni risultato va
   accompagnato dalla copertura misurata e dal TTL usato.
3. Le label vengono da un foglio Excel compilato a mano, ±1 min.
4. Nessun pcap: se `ja3` non fosse stato presente in `ssl.log`, non sarebbe stato
   ricalcolabile. (È presente — verificato.)
5. Nessun attributo statico ZTA: ruolo, clearance e sensibilità della risorsa sono a zero,
   quindi il modello si appoggia solo a memoria + storia delle interazioni. La classe
   *policy* non è valutabile per costruzione.
6. Le colonne di allarme del messaggio sono tenute **pulite** (nessun IDS nel rilascio) e
   `bytes_out` è tenuto a 0: la dimensione della risposta non è disponibile al decision
   time, stessa ragione per cui `http_status` è stato rimosso dal messaggio sintetico.

## 4. LANL come ablazione, non come benchmark alla pari

LANL 2015 è ciò su cui girano Euler (NDSS'22 / ACM TOPS'23), Argus (RAID'20), LMDetect
(arXiv 2411.10279) e il graph foundation model (arXiv 2504.13527), con convenzioni di split
precise (LANL: prime 41 ore in training; OpTC: 6 giorni / 3 giorni). Omettere LANL invita la
domanda "perché no?" da ogni revisore.

Ma la degradazione va detta in una frase chiara: `source` e `device` collassano entrambi
sull'id del computer; il flag internal/external diventa costante; `config` degenera nella
coppia `(auth type, logon type)` con 55% / 14% di valori nulli; `resource` è a granularità
host senza metadati di sensibilità; **tutti** gli attributi statici sono sintetizzati.
Conseguenza: **su LANL la classe credential-theft — la ragione per cui il nodo config esiste
— non è valutabile.** Il modo onesto di usarlo è proprio questo: LANL *è* il braccio "senza
nodo config", e la sua degradazione è l'argomento empirico a favore del nodo.

Harness già presente: `tests/eval_lanl.py` + `tests/datasets/lanl_auth.py` (nota: il
messaggio è ancora a 7 dimensioni contro `msg_dim=10`, va allineato prima di eseguirlo).

## 5. Riproduzione

```bash
# 1. dataset (16 MB, aperto)
git clone --depth 1 https://github.com/iHeartGraph/PicoDomain.git data/pico
7zz x -odata/logs data/pico/Zeek_Logs.7z
cp "data/pico/Red Log.xlsx" data/

# 2. ispezione del mapping (CPU, pochi secondi) — stampa copertura e conteggi
python -m tests.datasets.picodomain --log-dir data/logs --red-log "data/Red Log.xlsx"

# 3. valutazione completa (GPU box)
docker compose --profile eval-picodomain up
```
