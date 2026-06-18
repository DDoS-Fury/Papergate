# TODO — Aggiunta nodo "configuration" (JA3) al TGN (schema v3 → v4)

## Contesto
Promuovere la configurazione del client (fingerprint JA3, fallback `conf:guest`) a 5° ruolo
di nodo. Catena causale: `source → config → device → user → resource` (config sostituisce
l'edge diretto `source → device`) + binding `config → user`. JA3 bit `features[0]` resta
come validità TLS (msg_dim=7, node_feat_dim=16 invariati). Architettura invariata per ora.
Piano completo: /home/gabs/.claude/plans/obiettivo-si-vuole-aggiungere-toasty-rossum.md

## Edge set per richiesta (5 edge; config sempre presente lato serving)
- `user → resource` (access, msg) — esistente
- `device → user` (binding) — esistente
- `source → config` (binding) — NUOVO
- `config → device` (binding) — NUOVO
- `config → user` (binding) — NUOVO
Ordine commit memoria: source→config, config→device, config→user, device→user, user→resource.

## Checklist
- [x] 1. `src/config.py`: `num_configs=40`, slot config in `total_nodes`, `schema_version=4`
- [x] 2. `src/data/stream_synthetic.py`: range config, config abituali per macchina, conf:guest,
         config attaccante (theft/lateral new-tool), key_config in eventi, SyntheticStream + layout
- [x] 3. `src/serve_tgn.py`: SCHEMA_VERSION=4, key_config (default conf:guest), 3 nuovi edge
- [x] 4. `src/serve_api.py`: `EventIn.key_config`, pass in /infer, /update, /score
- [x] 5. `src/train_tgn.py`: StreamData config, training loop (3 objective + reroute source→config→device),
         _replay (config edges), chiamate calibrazione/test
- [x] 6. `tests/generator.py` + `tests/test_client.py`: key_config + prefix prod_; `tests/test_serve_v2.py` aggiornato a v4
- [x] 7. Verifica: train breve (checkpoint v4), serve /infer con key_config, test_client

## Review

### Cosa è stato fatto
- Nodo `configuration` aggiunto come 5° ruolo. Catena: `source → config → device → user → resource`
  + binding `config → user`. Il vecchio edge diretto `source → device` è stato SOSTITUITO da
  `source → config` + `config → device` (gated `has_config`; il path legacy resta per dataset
  senza config, es. LANL). `msg_dim=7` e `node_feat_dim=16` invariati; nessuna modifica a `tgn.py`.
- Generatore: ogni macchina ha 1-2 config abituali da un pool condiviso; `conf:guest` per client
  non fingerprinted; furto credenziali alloca `conf:atk-NNN` (mai visto); movimento laterale
  presenta con prob 0.5 un config "new tool" mai usato da quel device.

### Verifiche eseguite (Docker, immagine graphagate:latest)
- `py_compile` pulito su tutti i file modificati.
- Primitive di serving (snippet diretto): schema gate rigetta v1/v2/v3; chain a 5 edge committa
  i pair_count corretti (source→config, config→device, config→user, device→user, user→resource +
  aux); default `conf:guest`; fallback `key_source=None` (salta solo source→config). TUTTO OK.
- Pipeline training completa (20k eventi / 2 epoch / batched): gira end-to-end senza errori,
  dimensioni coerenti, metriche per-tipo sane (policy AUC 0.95, lateral 0.85). NB: run smoke,
  NON comparabile ai numeri pubblicati (200k/15ep/bs=1).
- Live HTTP: serve_api carica un checkpoint v4 (schema_version=4), `/infer` con/senza key_config,
  `/update`, no-device, dirty-signal → tutti 200 + score validi. `test_client.py` 15s: 1016 eventi,
  P50 ~10ms, benign specificity 99.7%, nessun errore (recall 0% atteso: modello sotto-addestrato +
  tutti gli attori cold-start `prod_`).

### Retrain full v4 eseguito (seed 42, 200k/15ep/bs=1, salvato in public/, capacity 50547)
Confronto con i numeri v3 di riferimento (Run B, stesso seed/config — vedi sopra):

| Metrica | v3 (Run B) | v4 | Δ |
|---|---|---|---|
| agg AUC | 0.952 | **0.964** | +0.012 |
| agg AP | 0.896 | **0.905** | +0.009 |
| agg recall (routed) | 0.709 | **0.835** | +0.126 |
| lateral AUC | 0.894 | **0.936** | +0.042 |
| lateral AP | 0.464 | **0.614** | +0.150 |
| lateral recall (routed) | 0.255 @4.8%FPR | **0.599** @5.6%FPR | **+0.344** |
| lateral recall (global @1%FPR) | 0.141 | 0.152 | +0.011 |
| policy AUC / recall@thr | ~0.95 | 0.967 / 0.941 | ~ |
| contextual AUC / recall@thr | ~0.99 | 0.990 / 0.970 | ~ |

**Esito:** il nodo config migliora nettamente il movimento laterale (recall routed 0.255→0.599,
AP +0.150, AUC +0.042) a fronte di un FPR benigno quasi invariato (~5.6% vs 4.8%). Il segnale
"tool nuovo su device noto" (config→device) + "client che l'utente non usa" (config→user) sono
esattamente i tell del laterale/furto.

**Caveat (onestà):** (1) single-run — lat-AUC ha rumore GPU ±0.01–0.03, quindi il +0.04 AUC è
reale ma il segnale robusto è il +0.34 di recall routed; per claim pubblicabili serve multi-seed.
(2) cred-theft e wiped-cookie hanno **n=0** nello split di test (ultimi 20%): artefatto di split
(gli slot theft si esauriscono presto nello stream), identico in v3 → non misurabile questa run.
Il beneficio atteso di config→user sul furto credenziali resta da quantificare con uno split/eval
che porti gli incidenti theft nel test.

### Follow-up possibili
- Eval mirata cred-theft (aumentare num_theft_slots o split dedicato) per quantificare config→user.
- Layer extra al `LinkPredictor` (concordato): valutare solo se serve altra capacità — le metriche
  attuali sono già migliorate senza, quindi non urgente.
