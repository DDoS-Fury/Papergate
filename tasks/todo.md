# TGN — Correttezza & Streaming-Readiness (ZTA)

Verifica della rete TGN (`src/model/tgn.py`, `src/train_tgn.py`) per l'uso real-time
su Zero Trust Architecture, e correzione dei punti critici trovati.

## Esito verifica (cosa era corretto)
- Ordine memoria read → use → update: **corretto** (come l'esempio ufficiale PyG).
- `rel_t = last_update - t`: **corretto** (combacia con PyG).
- Timestamp `long` end-to-end: **corretto** — `TGNMemory.last_update` è `int64`, quindi
  generator/score_event coerenti. (Smentita la presunta "skew di dtype".)
- Split 80/20 by-index su stream ordinato nel tempo: **è cronologico**, non è leakage.

## Correzioni implementate
- [x] **Anti-poisoning gate.** Update di memoria solo su eventi benigni: per-label in
      calibrazione, per-predizione (`score < threshold`) in serving/eval. Train/eval/serve
      ora coerenti. (`serve_tgn.score_event`, `train_tgn._replay`)
- [x] **Valutazione = serving.** Loop di test riscritto per-evento sullo stesso codepath
      del serving (`infer_score`/`update_memory`), niente più snapshot stantio su batch da 200.
- [x] **Nodi dinamici.** `NodeRegistry` (`src/model/registry.py`) mappa entità esterne →
      slot interni con admission lazy ed eviction LRU; memoria dimensionata a `capacity`
      (= entità note + headroom). Entità mai viste in training vengono ammesse a runtime.
- [x] **Threshold calibrato.** Quantile a `target_fpr` (da config) su slice benigno di
      validazione; persistito.
- [x] **Persistenza deployabile.** `public/tgn_checkpoint.pt` (pesi + buffer memoria +
      message store `msg_s_store`/`msg_d_store`) e `public/tgn_stats.json` (threshold +
      registry + capacity). `load_model`/`save_model` in `serve_tgn.py`.
- [x] **Config dedicata.** `TGNConfig` + path artifact in `src/config.py`.
- [x] **Harness di verifica.** `src/verify_tgn.py` (profilo Compose `verify-tgn`).

## Verifica eseguita (in container, GPU)
- `docker compose --profile training-tgn run --rm train-tgn`
  → Train Loss 0.91→0.17; threshold@FPR0.05 = 0.817; **AUC/AP 1.0** (atteso: il
  generatore sintetico rende le anomalie banalmente separabili via feature `ja3`/`snort`);
  artifact scritti in `public/`.
- `docker compose --profile training-tgn run --rm train-tgn graphagate.verify_tgn`
  → **4/4 PASS**: reload determinism; benign aggiorna memoria; anomalia **non** avvelena
  la memoria; nodo dinamico ammesso (idx 170, registry 170→171).

## Limitazione nota (FUORI scope per scelta)
Il modulo di attention fa message-passing **sull'arco target stesso** invece che sui
vicini temporali storici (un TGN canonico usa un neighbor loader). Quindi: (1) il segnale
relazionale che dovrebbe cogliere il *lateral movement* è in gran parte assente, e (2) il
`msg` dell'evento entra nell'embedding ed è anche passato grezzo al predittore. Le feature
statiche dei nodi (schema 14-D: privilege/role) non sono usate. Da affrontare in un
secondo step ("full canonical TGN") se si vuole rilevare anomalie strutturali reali oltre
a quelle contestuali.

## Review
Tutte le modifiche sono additive o localizzate; il loop di training conserva la logica
originale (filtro benigno + negativi struttura/contesto). Il valore AUC perfetto dipende
dal dataset sintetico, non dalla pipeline: con dati realistici servirà il neighbor loader
(vedi limitazione) per il segnale strutturale.
