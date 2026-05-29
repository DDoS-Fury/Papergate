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
