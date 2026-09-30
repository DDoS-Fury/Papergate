# Proposte di refactoring non implementate

Stato al 2026-09-30, branch `refactor/readability`. Le proposte vengono dall'analisi del codice
fatta insieme alla pulizia dei commenti e al refactor di `src/data/stream_synthetic.py`. Quel refactor
lascia lo stream identico bit per bit a master.

Le proposte qui sotto **non** sono state applicate perché toccano la numerica del training o il
percorso di serving. Si possono controllare sul Mac solo in parte (test di parità su CPU); la prova
definitiva richiede un training sulla workstation, confrontato con master.

---

## 1. Tabella unica della catena di archi

**Problema.** La topologia della catena (quali archi esistono dati `has_src` / `has_bind` /
`has_config`) è scritta a mano in 8 punti:

| file | punto |
|---|---|
| `train_tgn._replay` | 3 punti: gruppi di scoring, commit in batch, contatori `_bump` |
| `train_tgn.train_tgn` | 5 punti: negativi, `parts` dell'espansione, loss di binding, commit, contatori |
| `serve_tgn` | `score_event` (gruppi + commit), `commit_event` (commit) |

**Proposta.** Una tabella dichiarativa in `serve_tgn`, usata anche dal training:

```python
# (kind, src_role, dst_role), in ordine di COMMIT
CHAIN = [
    (EDGE_SRC_CFG,  "source", "config"),
    (EDGE_CFG_DEV,  "config", "device"),
    (EDGE_CFG_USER, "config", "user"),
    (EDGE_DEV_USER, "device", "user"),
    (EDGE_ACCESS,   "user",   "dst"),
]
```

La tabella va filtrata in base ai ruoli presenti. `EDGE_SRC_DEV` compare solo nell'ablazione senza
nodo config.

**Insidie.**
- L'ordine di commit **cambia la memoria**, perché ogni `update_state` legge lo stato lasciato dalla
  chiamata precedente. Va mantenuto esattamente: source→config, config→device, config→user,
  device→user, user→resource.
- L'ordine dei gruppi di scoring è diverso da quello di commit. Qui l'ordine non conta, perché le
  calibrazioni sono indicizzate per `kind` e poi si prende il max. Servono comunque due viste
  distinte della tabella.
- I gruppi **non** vanno fusi in una sola chiamata `model.score`: cambierebbe l'arrotondamento
  float32 del time encoder (vedi il docstring di `chain_edge_logits`).
- Contratto di parità: `tests/verify_replay_batching.py` e `tests/test_fusion_parity.py` devono
  restare verdi.

## 2. Tipi di anomalia e colonne delle feature: un solo vocabolario

**Problema.** I codici `etype` 0-6 e gli indici delle colonne di `node_feat` (2, 4, 5, 14) e del
messaggio (0-4) compaiono come numeri magici in:
- `train_tgn`: metriche per tipo, `fit_thresholds`, cold start, scenari, baseline a regole;
- `calibration.operating_point`;
- `serve_tgn`: `_reset_slot`, `_set_node_features`, `signal_dirty`, `sensor_alarm`;
- `lookup_rules` ed `eval_common`.

**Proposta.** Riusare `stream_synthetic.EventType`, già introdotto, e rendere pubbliche le colonne
`_NF_*` di `stream_synthetic`, per esempio come `NodeFeat(IntEnum)` in un modulo neutro. Non va creato
un secondo enum con gli stessi valori.

**Insidia.** Nelle metriche per tipo compare anche il codice 5 (exfiltration, dataset esterni).
`EventType` lo riserva ma non lo definisce: va aggiunto oppure gestito a parte.

## 3. Serving: blocchi duplicati

- **`serve_tgn`**: `score_event` e `commit_event` hanno due blocchi identici.
  - Ammissione delle chiavi + feature statiche → `_admit_chain(...) -> ChainIds`, un piccolo
    dataclass con user, device, dst, source e config.
  - Sequenza di commit → `_commit_chain(model, ids, t, features, device)`.
  - Anche la regola "attore = device se presente, altrimenti user" è ripetuta: in `score_event`,
    `commit_event`, `deny_event` e `train_tgn._replay`.
  - `"conf:guest"` è scritto due volte: andrebbe in una costante `GUEST_CONFIG_KEY` in `netclass`,
    accanto a `GUEST_DEVICE`.
- **`serve_api`**:
  - `/infer` e `/score` differiscono solo per `update=` e per l'azione trasmessa sulla websocket:
    si possono ridurre a un unico `_score(ev, update, action, bg)`;
  - il payload trasmesso sulla websocket è duplicato;
  - `save_model(...)` è chiamato due volte (shutdown e `/persist`): serve un `_persist()`;
  - `bool(STATE.hp.get("guest_device_fallback", False))` compare 4 volte: basta una proprietà di
    `_State`.

**Insidie.**
- Contratto HTTP: alias `flagged`, default di `ScoreOut.alarm`, chiavi e 503 di `/health`, nomi delle
  `action` websocket (li legge la dashboard).
- Le chiamate `model.eval()` in `score_event` e `commit_event` devono restare dove sono.

## 4. Split di `train_tgn()` (~770 righe)

Funzioni candidate:
- `_seed_everything`, `_select_device`, `_build_model`;
- `_train_epoch`, con dentro `_sample_negatives`, `_batch_loss` e `_commit_benign_batch`;
- `_calibrate`, che restituisce una NamedTuple `Thresholds(clean, dirty, clean_unsup)`;
- `_eval_test`, `_cold_start_metrics`, `_scenario_metrics`, `_rule_baseline_metrics`, `_persist`.

`_replay` (~200 righe) si può dividere in `_edge_groups`, `_decide_event` e `_commit_batch`.

**Insidie** (tutte cambiano i numeri in silenzio):
- **Ordine delle estrazioni casuali per i negativi.** Oggi è `neg_res`, `neg_usr`, `neg_cusr`,
  `neg_cdev`, `neg_scfg`, `neg_dev`, e poi `randn_like`. Inoltre
  `_sample_structural_negatives` fa un secondo `randint` solo in caso di collisione.
- **Inizializzazione dei pesi.** Consuma l'RNG di torch, quindi deve seguire subito il seeding.
- **Somma della loss.** È accumulata da sinistra a destra: l'ordine dei termini va mantenuto.
- **Calibrazione.** Serve la stessa sequenza: pass A, ricalcolo dei punteggi, restore, pass B;
  `recent_alert.clear()` va prima del replay di test.
- **`anomaly_score`.** Resta in float64.

## 5. Stato di runtime in un solo posto (Memento)

**Problema.** L'elenco dei campi di runtime è ripetuto in tre punti: `serve_tgn.save_model`,
`serve_tgn.load_model` e `_snapshot_runtime` / `_restore_runtime` in `train_tgn`. I campi sono:
- `msg_s_store`, `msg_d_store`;
- `last_contact`, `pair_count`, `src_count`, `recent_alert`;
- lo stato del neighbour loader.

**Proposta.** `ZTATemporalGraphNetwork.runtime_state()` / `load_runtime_state()`, così aggiungere un
campo diventa una modifica in un solo punto. Il formato del checkpoint va tenuto compatibile, cioè
con le stesse chiavi.

## 6. Fallback hard-coded in `serve_tgn.build_model`

I default usati quando manca una chiave in `hp` non coincidono con `TGNConfig`:

| chiave | fallback in `build_model` | `TGNConfig` |
|---|---|---|
| `neighbor_size` | 10 | 30 |
| `hash_buckets` | 10000 | 100000 |

Nei checkpoint prodotti da `train_tgn` le chiavi ci sono sempre, quindi oggi non ha effetti. Conviene
però leggere tutti i fallback da `TGNConfig`, come già si fa per i parametri del precursor.

---

## Da non fare

- **Non unificare `train_tgn._rule_baseline` e `serve_tgn.signal_dirty`.** Il primo usa i confronti
  esatti `== 0.0` / `== 1.0`, il secondo le soglie `<= 0.5` / `> 0.5`. Sui dataset esterni con colonne
  non strettamente 0/1 i risultati cambierebbero.
- **`GeneratorParams`** (oggetto parametri per le ~40 manopole del generatore): romperebbe tutte le
  chiamate `dataclasses.replace(cfg, p_x=...)` e `**V4_KNOBS` in test e ablation. La divergenza dei
  default è già risolta, perché `generate_streaming_data` passa gli argomenti al simulatore così
  come sono.
- **RNG per istanza nel generatore** (`np.random.Generator` al posto dello stato globale): sarebbe
  più pulito, ma cambia gli stream e quindi tutti i numeri sintetici.
- **Strategy per `gate_by_label`, template method per il loop di training, registry delle metriche:**
  due casi o un solo chiamante, sarebbe troppo per il beneficio.

## Piccole cose rimaste

- Tre intestazioni stampate in `train_tgn` sono ancora in italiano o con la versione: "SCENARI v2",
  "BASELINE A REGOLE", "lateral su device condivisi". Si possono rinominare: nessuno script le
  legge.
- "Inferenza (replay test)" invece **va lasciata così**: `tasks/tmp/diag_edgecal/diag_scores.py` la
  usa per riconoscere la fase del replay.
