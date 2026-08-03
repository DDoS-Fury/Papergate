# Provenienza dei numeri del report

> Mappa **tabella → sorgente numerica → script → profilo Docker** per `report.tex`.
> Obiettivo (tasks/report-improvements.md, P2): nessun numero titolare copiato a mano
> senza una sorgente rigenerabile. Ogni run usa `save=False` — l'artefatto deployable in
> `public/tgn_checkpoint.pt` non viene mai toccato.

## Come rigenerare

Tutto gira sulla GPU box via Docker Compose (mai venv CPU locale — cfr. `tasks/lessons.md`):

```bash
cd infra/ai-inference
docker compose --profile regen-report      up   # Pannelli A e B (tab:baselines, tab:v3v4)
docker compose --profile config-eval       up   # tab:theft
docker compose --profile arch-sweep        up   # tab:archsweep
docker compose --profile guest-device-eval up   # tab:guestdev
docker compose --profile ablations         up   # ablation componenti (Cap. 1)
```

I numeri multi-seed (media ± dev.std su 3 seed `[42, 7, 123]`) sono lo standard. Il
Pannello A/B scrive metriche **machine-readable** in `tasks/runs/panel{A,B}.json` e
frammenti LaTeX in `docs/latex/generated/`; gli altri driver stampano su stdout (catturato
nei `tasks/runs/*.log`).

## Mappa tabelle

| Tabella (label) | Sezione | Sorgente numerica | Script generatore | Profilo Compose |
|---|---|---|---|---|
| `tab:baselines` (Pannello A) | §eval | `tasks/runs/panelA.json` + `docs/latex/generated/tab_baselines.tex` | `tests/regen_report_tables.py` | `regen-report` |
| `tab:v3v4` (Pannello B) | §v4results | `tasks/runs/panelB.json` + `docs/latex/generated/tab_v3v4.tex` | `tests/regen_report_tables.py` | `regen-report` |
| `tab:theft` | §v4results | `tasks/runs/config_eval.log` | `tests/ablations/run_config_eval.py` | `config-eval` |
| `tab:archsweep` | §archsweep | `tasks/runs/arch_sweep.log` | `tests/ablations/run_arch_sweep.py` | `arch-sweep` |
| `tab:guestdev` | §guestdev | `tasks/runs/*guest*device*.log` | `tests/ablations/run_guest_device_eval.py` | `guest-device-eval` |
| Ablation componenti (Cap. 1) | §architettura/limiti | stdout | `tests/ablations/run_ablations.py` | `ablations` |
| `tab:summary` | §summary | aggrega A (`tab:baselines`), B (`tab:v3v4`), C (`tab:theft`, `tab:guestdev`) | — (collage) | — |
| `tab:v4edges` | §v4 topology | statico (descrizione catena a 5 archi) | — | — |

## Protocollo dei Pannelli A/B (importante)

- **Pannello B** (`tab:v3v4`): TGN, keying **per-cookie** (`guest_device_fallback=False`),
  stream standard 200k eventi / 15 epoche, decisione cost-sensitive **instradata**.
  Variante `use_config_node` ON = v4 (nodo config), OFF = ≈v3 (catena legacy a 4 nodi).
- **Pannello A** (`tab:baselines`): stesso stream standard, **protocollo unico per tutte le
  righe** — TGN e baseline (GNN non temp., One-Class SVM, Isolation Forest, XGBoost) girano
  tutte sotto la configurazione **deployable** (`dev:guest`, `guest_device_fallback=True`,
  v4) e tutte alla **soglia globale 1% FPR**. Quelle di Pannello B sono invece **instradate**
  (operative).
- ⚠️ Il protocollo misto precedente (riga TGN = v3 per-cookie riusata dal Pannello B, baseline
  = deployable) è stato **rimosso**: `run_panel_a_tgn` esegue ora il TGN sotto lo stesso
  protocollo delle baseline. **Le tabelle A attualmente versionate provengono dal protocollo
  vecchio e da un generatore con label leakage: vanno rigenerate.**

## Semantica delle metriche (chiavi JSON → celle)

Il dict restituito da `graphagate.train_tgn.train_tgn` (e dai baseline) contiene:

- `agg_auc`, `agg_ap` — AUC / AP aggregate sul test.
- `agg_recall` — recall aggregata alla decisione **instradata** (Pannello B).
- `agg_recall_global` — recall aggregata alla **soglia globale 1% FPR** (Pannello A).
- `per_type[c].auc|ap|recall|n` — breakdown per classe (`policy`/`contextual`/`lateral`
  e, per il TGN, `cred-theft`); `recall` = decisione instradata per il TGN, soglia globale
  per i baseline.
- `lateral_recall_before` — recall laterale alla soglia globale 1% FPR (Pannello A/B "@1%FPR").
- `fpr_after` — FPR benigno alla decisione instradata (Pannello B).

Il marcatore `$^{\dagger}$` nei frammenti generati segnala un Δ il cui valore assoluto è
**entro** la banda di rumore multi-seed (max delle due dev.std) → non significativo.
