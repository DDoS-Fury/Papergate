# TODO — Colmare il gap: v4 + dev:guest sullo stream standard (2026-06-24)

## Contesto
Il deployable (`guest_device_fallback=True`, `dev:guest`) è valutato SOLO su stream
theft-rich (Pannello C, Tab. `tab:guestdev`). Sullo stream standard (Pannello B,
200k ev / 15 ep, decisione cost-sensitive instradata) tutti i numeri pubblicati sono
**per-cookie**, prodotti prima dell'adozione del guest. Manca la riga v4+guest standard:
è ciò che gira in produzione e dovrebbe stare nel quadro riepilogativo.

## Piano
- [x] Indagine: confermato che v4+guest non è mai misurato su stream standard
- [x] Driver `tests/ablations/run_guest_standard_eval.py` (sibling di run_guest_device_eval,
      ma stream standard + metriche Pannello B; v4 per-cookie vs v4+dev:guest; 3 seed; save=False)
- [x] Profilo Compose `guest-standard-eval`
- [x] Eseguire via docker compose (GPU) — 2 varianti × 3 seed × 200k/15ep
- [x] Riportato in 05.tex: NIENTE sezione nuova — il risultato è folded in coda
      all'Interpretazione di §sec:guestdev (un paragrafo + Tab. tab:gueststd, dati appaiati
      dello stesso run). Niente rigenerazione: per-cookie e guest freschi vengono dallo
      stesso run. Pannello B pubblicato non toccato. Tesi compila (latexmk exit 0, ref OK).
- [x] Sezione review

## Note
- Sullo stream standard il furto si esaurisce in train → metriche theft NaN nel test
  (atteso, come Pannello B). Focus su lateral + aggregate + FPR instradato.

## Review — risultato (3 seed [42,7,123], 200k/15ep, instradata, save=False)
| metrica (test) | v4 per-cookie | v4 + dev:guest | Δ (guest−cookie) |
|---|---|---|---|
| AUC aggregata | 0.979±0.004 | 0.965±0.005 | **−0.014** |
| AP aggregata | 0.955±0.007 | 0.926±0.021 | −0.029 |
| Recall aggregata (instradata) | 0.851±0.003 | 0.811±0.000 | −0.039 |
| Lateral AUC | 0.952±0.007 | 0.922±0.009 | **−0.029** |
| Lateral AP | 0.749±0.027 | 0.652±0.078 | −0.097 |
| Lateral Recall (instradata) | 0.603±0.006 | 0.532±0.028 | **−0.071** |
| Lateral Recall @1%FPR | 0.458±0.041 | 0.372±0.065 | −0.086 |
| FPR benigno (instradato) | 0.043±0.008 | 0.042±0.007 | −0.001 |

**Finding (onesto, controintuitivo rispetto alla premessa).** Sullo stream STANDARD il
collasso `dev:guest` NON è il miglioramento di Pareto visto su theft-rich: è una **lieve
regressione** sul laterale (lat AUC −0.029, recall instradata −0.071), agg AUC −0.014, FPR
**piatto** (−0.001). Motivo: il test split standard ha **zero furto e ~zero cookie-wipe**,
quindi i due vantaggi del guest (detection del furto + eliminazione dei FP da wipe + crollo
di varianza) sono **assenti per costruzione**; resta solo la perdita della novelty-cue
per-macchina, che qui costa un po' di ranking laterale.
- Cross-check: guest 3-seed (lat AUC 0.922, agg 0.965) ≈ retrain deployable single-seed in
  README (0.918 / 0.959). Coerente.
- **Caveat code-drift**: il v4 per-cookie FRESCO (lat AUC 0.952) è > del Pannello B
  pubblicato (0.930): il codice è migliorato dopo la generazione delle tabelle. Il confronto
  VALIDO è appaiato fresh-vs-fresh (stesso run), cioè il Δ qui sopra. Per coerenza assoluta
  andrebbe rigenerato anche il Pannello B (profilo `regen-report`).
- Implicazione tesi: il modello **deployato** (guest) ha lat AUC ~0.922 sullo stream
  standard, leggermente **sotto** il 0.930 per-cookie pubblicato; il guest si giustifica coi
  benefici theft-rich/operativi (Tab. guestdev), non come win sullo stream standard.

---

# TODO — Esperimento: device senza TPM → nodo `dev:guest` condiviso (2026-06-18)

## Contesto
Testare come si comporta il modello se TUTTI i device senza TPM collassano su un unico
nodo `dev:guest` condiviso (mirror di `conf:guest`), invece di un nodo cookie per-macchina.
Flag `guest_device_fallback` (default off) → A/B reversibile.
Piano: /home/gabs/.claude/plans/obiettivo-vorrei-testare-il-whimsical-hejlsberg.md

## Checklist
- [x] 1. `netclass.py`: `GUEST_DEVICE = "dev:guest"` + helper `to_guest_device`
- [x] 2. `config.py`: flag `guest_device_fallback: bool = False` in SyntheticConfig
- [x] 3. `stream_synthetic.py`: collasso REALE via `machine_slot` su un unico slot guest
        (non basta rietichettare le keys: in training i nodi sono slot id), wipe no-op,
        attacker device cred-theft anch'esso collassato, wiring in generate_streaming_data
- [x] 4. `serve_tgn.py`: collasso `to_guest_device(key_device)` in score_event/commit_event
- [x] 5. `train_tgn.py`/`serve_api.py`: flag salvato in hp e riapplicato al serving
- [x] 6. Test `tests/test_serve_v2.py` + verifica generatore (collasso, baseline, wipe)

## Review
- **Errore di impianto corretto in corsa:** il piano iniziale assumeva che rietichettare
  `self.keys` collassasse i device. FALSO: in training i nodi sono indici di slot
  (`device=dev_slot`), e `registry.preregister` esige chiavi UNICHE per slot. Vero collasso
  = instradare `machine_slot` di ogni macchina non-TPM su un unico slot guest; gli slot
  inerti tengono chiavi placeholder uniche. Vedi lessons.md.
- **Verifiche (Docker `graphagate:latest`):** 12/12 pytest verdi (nuovo test incluso);
  generatore con flag on → 1 sola key `dev:guest`, tutti gli eventi non-TPM su quello slot,
  `preregister` identity su tutti gli slot, wipe neutralizzato (off=179→on=0 ad alto rate),
  tier feature guest=0. Baseline (flag off) strutturalmente invariata (path guardato).
- **Conseguenza sperimentale documentata:** l'attacker device del cred-theft, essendo
  TPM-less, collassa anch'esso su guest → il tell device-identity sparisce; restano IP/JA3.

## Run A/B (Docker GPU, 3 seed, theft-rich 80k/12ep, save=False) — `tests/ablations/run_guest_device_eval.py`
- Profilo compose `guest-device-eval`. Esito **controintuitivo**: collassare i device
  non-TPM su `dev:guest` NON degrada, anzi è un miglioramento di Pareto sulle metriche:
  - lateral AUC 0.915→0.911 (piatto, −0.003 entro rumore); lateral recall +0.035, AP +0.049
  - **furto cred. AUC 0.701→0.804 (+0.103)**, recall +0.043 (ma std baseline ±0.078 alta)
  - agg AUC +0.007, **FPR benigno 0.045→0.037 (−0.007)**, **varianza tra seed crollata**
  - **cookie-wipe FP eliminati**: n_wiped 651→0 (neutralizzazione confermata)
- Interpretazione: l'identità per-cookie era rumore (nodi sparsi/freddi); il segnale degli
  attacchi signal-clean vive su config/source/user, non sulla novità del device.
- Costo non misurato: perdita di attribuzione per-macchina (forense). Deployable resta a
  default cookie (`guest_device_fallback=False`).
- Sezione `\section{...dev:guest}` (label `sec:guestdev`, Tab. `tab:guestdev`) aggiunta al
  Cap.2 di `docs/latex/report.tex`; compila (pdflatex, 14 pagine, ref risolte).

## Deployable → dev:guest (branch 5-nodes-alpaca)
- `config.py`: `guest_device_fallback` default ribaltato a **True** (è ora la scelta deployable).
- Riaddestrato l'artefatto `public/tgn_checkpoint.pt` (200k/15ep/seed42, save=True): agg AUC
  0.959, lateral AUC 0.918, `wiped-cookie n=0` (flag attivo). Backup del v4-cookie precedente
  in `backups/public_v4_cookie_20260618/`.
- Verificato end-to-end: `hp.guest_device_fallback=True`; serving collassa `ck:`→`dev:guest`,
  mantiene `tpm:` distinto. serve_api applica il flag leggendolo da `hp`.

---

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
