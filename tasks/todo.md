# TODO — Doc + validazione mirata + tuning architetturale nodo config (v4)

## Contesto
Il nodo `configuration` (JA3) è già in v4 (deployable in public/). Ora: documentare in
LaTeX, quantificare il cred-theft (oggi n=0 nel test split), e valutare miglioramenti
architetturali (layer MLP, memory, heads). Decisioni: eval theft `save=False`,
multi-seed [42,7,123] protocollo ridotto, re-save deployable solo se migliora.
Piano: /home/gabs/.claude/plans/obiettivo-si-vuole-aggiungere-toasty-rossum.md

## Checklist
- [x] 1. Infra: `use_config_node` toggle (≈v3) in train_tgn; knob `gnn_heads`,
        `link_pred_hidden_layers` in config/tgn/serve_tgn/train_tgn (default = attuale)
- [x] 2. Verifica infra: py_compile + load checkpoint v4 esistente (back-compat OK)
- [x] 3. Deliverable A: `run_config_eval.py` + profilo `config-eval` → cred-theft n=452
        (era 0); Δ config node: theft recall +0.108, lateral recall +0.135
- [x] 4-5. Deliverable B: `run_arch_sweep.py` + profilo `arch-sweep` → nessuna variante
        migliora (layer MLP neutro; memory/heads peggiorano) → architettura invariata
- [x] 6. Deliverable C: capitolo `docs/latex/report.tex` — architettura v4 + 3 tabelle
        (v3→v4, cred-theft, sweep) + sintesi; PDF compila (12 pp., exit 0)
- [x] 7. Deliverable D: deployable INVARIATO (nessun miglioramento da adottare);
        lessons.md aggiornato coi numeri di riferimento

## Review

### Cosa è stato fatto
- **Infra**: toggle `use_config_node` (ablazione nodo-config ≈v3 a parità di dati,
  riusa il gating `has_config` già presente) + knob architetturali parametrici
  (`gnn_heads`, `link_pred_hidden_layers`) con default = architettura storica. La
  back-compat è garantita: `LinkPredictor.lin_extra` è una ModuleList vuota a default,
  quindi i checkpoint v4 esistenti caricano senza modifiche (verificato in Docker).
- **Deliverable A (validazione mirata)**: nuovo driver theft-rich; cred-theft passa da
  n=0 a n=452 nel test. Il nodo config aumenta recall su furto credenziali (+0.108) e
  laterale (+0.135) vs ablazione; AUC theft +0.053 (std ampia → recall è il segnale
  robusto). `save=False`: deployable mai toccato.
- **Deliverable B (sweep)**: layer MLP extra neutro (AUC +0.002 entro rumore, recall ↓);
  memory_dim=384 e gnn_heads=8 peggiorano e aumentano la varianza. Il laterale è
  signal-bound, non capacity-bound → architettura invariata.
- **Deliverable C (LaTeX)**: nuovo capitolo "Evoluzione del modello: il nodo
  Configuration (v4)" con motivazione, architettura (doppio ruolo JA3, catena a 5 archi,
  TikZ + tabella edge), 3 tabelle risultati (v3→v4 deployable, validazione cred-theft,
  sweep) e sintesi. Vecchio capitolo v3 lasciato intatto come confronto.

### Verifiche
- py_compile pulito su tutti i .py modificati; `docker compose config -q` valido.
- Smoke Docker: percorsi `use_config_node=False` e knob (layer/memory/heads) girano
  senza errori di shape; cred-theft n>0 confermato.
- Back-compat: load del checkpoint v4 esistente + score_event OK coi nuovi default.
- LaTeX: 2 passate, 12 pagine, exit 0, nessun riferimento irrisolto.

### Decisioni
- Deployable in public/ INVARIATO (decisione "solo se migliora": nessun guadagno).
- Default theft/wipe slots invariati (64/16): la config theft-rich è solo per l'eval.

### Possibili follow-up (non richiesti)
- Multi-seed full 200k/15ep per i numeri headline v3→v4 (oggi single-run + caveat).
- Verificare lo sweep a scala piena (improbabile cambi: degrado consistente a 40k).
