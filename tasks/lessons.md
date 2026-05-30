# Lessons

## Esecuzione: usare Docker Compose, non venv locali
**Correzione utente:** non creare/usare virtualenv locali su questo progetto.
L'ambiente (torch 2.12 + PyG + CUDA 13) vive nell'immagine `graphagate` ed è
orchestrato da `docker-compose.yml` (bind-mount `./src` e `./public`).
**Regola:** per eseguire qualsiasi cosa qui (training, script, introspezione) usare
`docker compose --profile <p> run --rm <service> [comando]`, eventualmente con
`--entrypoint python` per snippet ad-hoc. L'entrypoint dell'immagine è `python -m`.

## Verificare le affermazioni dei subagent prima di trattarle come bug
Durante questa analisi alcune affermazioni di esplorazione erano errate e andavano
verificate leggendo il codice e introspezionando le librerie:
- "dtype skew dei timestamp": falso — `TGNMemory.last_update` è `int64`, e sia il
  generator sia `score_event` usavano già `long` (coerenti).
- "split random = leakage": falso — lo split by-index su stream ordinato nel tempo è
  cronologico.
- "memoria non resettata prima del test = leakage": falso — per uno stream continuo è
  il comportamento corretto.
**Regola:** confermare i punti critici (ordine memoria, dtype dei buffer, nomi attributi
come `msg_s_store`/`msg_d_store`) con `Read` + introspezione runtime nel container prima
di pianificare correzioni.

## Un check di "determinismo al reload" che confronta due reload può mascherare bug
Il check "reload determinism" di `verify_tgn` confrontava due `load_model` tra loro: se
il salvataggio/ricaricamento è rotto in modo *identico* per entrambi, il check passa lo
stesso. Nascondeva un bug reale: `TGNMemory.train(False)` (cioè `model.eval()`) fa il
*flush* del message store sul buffer `memory`. `load_model` ricostruiva il modello in
training mode, faceva `load_state_dict` (buffer `memory` già completo) **e** ripristinava
i message store, poi `model.eval()` → ri-applicava i messaggi sul buffer = **doppio
conteggio**. Fix: in `build_model` chiamare `model.eval()` *prima* di `load_state_dict`,
così il flush avviene a store vuoto e il ripristino è esatto.
**Regola:** un test di idempotenza deve confrontare lo stato ricaricato con lo stato
*originale*, non due ricaricamenti tra loro. Verificare il round-trip con
`torch.equal(buffer_originale, buffer_ricaricato)`.

## Il neighbor loader rileva il lateral movement solo se l'obiettivo lo esercita
Cablare il `MessageNeighborLoader` (corretto, verificato, round-trip esatto) **non basta**
a rilevare il lateral movement. Con negativi strutturali "facili" (dst casuale su tutti i
nodi, per lo più utenti/IP, non risorse) il modello impara solo "IP→risorsa = normale" e
il lateral resta a livello di caso (AUC ~0.49). Aggiungere negativi *hard* (dst = risorsa
casuale) + 10 epoche ha migliorato la policy (AUC 0.857→0.903) ma il lateral è rimasto a
caso (AUC ~0.53): la risorsa casuale collide spesso con quella abituale (solo 20 risorse)
e le risorse non hanno feature statiche distintive (`node_feat` risorse = 0).
**Regola:** prima di concludere che un meccanismo "non serve", verificare che
l'obiettivo di training e i dati lo *esercitino* davvero (negativi informativi,
identificabilità delle entità). Misurare il valore di un componente richiede un setup che
gli dia modo di esprimerlo.
