# Provenance of every number in the paper

Rule (`tasks/lessons.md` L5): a number enters a document only with (a) a file in
`tasks/runs/` and (b) the commit hash that produced it, recorded here. A number that
cannot be regenerated is **withdrawn**, not softened.

Mechanically enforced: no literal three-decimal figure may appear in `sections/*.tex`.
`make check` fails the build if one does. Every value lives in `results.tex`.

---

## Status

| Block in `results.tex` | Status | Blocking on |
|---|---|---|
| **Block 1** — synthetic-stream results | 🔴 **PRELIMINARY** | GPU box regeneration (`regen-report`/`config-eval`, see below) |
| **Block 2** — PicoDomain descriptors | 🟢 **MEASURED** | `tests.datasets.picodomain` |
| **Block 3** — PicoDomain model evaluation | 🟢 **MEASURED** (single run) | `tests/eval_picodomain.py`, see below |

`main.tex` carries `\preliminarytrue`, which stamps a banner on page 1 and on the
affected tables/figures. **Flip it to `\preliminaryfalse` only when Block 1 has been
regenerated and this table has commit hashes in it.**

> 2026-08-30: this row was briefly (commit `a11ed74`) marked "🟢 SYNCED" and the flag
> flipped to `\preliminaryfalse` without any accompanying regeneration — no new
> `tasks/runs/` artifact exists for that commit, and `results.tex`/`sections/*.tex`
> still carry the same `\prelim{}`-flagged, pre-de-leakage, mixed-protocol numbers
> described below. Reverted; see `tasks/todo.md` 2026-08-30 audit section.

---

## Block 1 — why it is preliminary

Two independent problems, both documented in `tasks/todo.md` §6 and §7:

1. **Not reproducible from HEAD.** The values were produced on 2026-06-24, before the
   generator de-leakage of 2026-08-03. That generator no longer exists. The runs that
   produced them were clean *on the lateral class* — `AUC(node_feat[dst,3])` on lateral
   was 0.4899 on the pre-leak generator, i.e. chance, so the reported lateral AUC was
   genuinely earned — but the de-leaked task is measurably harder (single-feature floor
   on lateral: 0.920 → 0.603, per `tasks/runs/leakage_audit_floor.log`), so the numbers
   will move.
2. **Panel A mixes protocols.** The TGN row comes from a v3 per-cookie run; every
   baseline row comes from a v4 deployable run. The driver has been corrected
   (`run_panel_a_tgn` now runs the TGN under the baseline protocol), but the versioned
   numbers predate that fix.

### Regeneration

All on the GPU box via Compose — never a local CPU venv (`tasks/lessons.md`):

```bash
docker compose --profile regen-report      up   # Panels A and B
docker compose --profile config-eval       up   # credential-theft deltas
docker compose --profile ablations         up   # per-component ablations (withdrawn)
docker compose --profile arch-sweep        up
docker compose --profile guest-device-eval up
```

Then update `results.tex`, fill the hashes below, and set `\preliminaryfalse`.

### Macro → source map

| Macros | Table | Source of record | Generator script | Compose profile | Commit |
|---|---|---|---|---|---|
| `\AggAuc*`, `\AggAp*`, `\LatAuc*`, `\LatRec*`, `\AggRec*` | III (`tab:baselines`) | `tasks/runs/panelA.json` | `tests/regen_report_tables.py` | `regen-report` | ⬜ TBD |
| `\Bagg*`, `\Blat*`, `\Bfpr*` | IV (`tab:panelb`) | `tasks/runs/panelB.json` | `tests/regen_report_tables.py` | `regen-report` | ⬜ TBD |
| `\TheftRecallDelta`, `\TheftLateralDelta`, `\TheftRecOn/Off`, `\TheftAucOn/Off`, `\TheftN` | §VI-C prose | `tasks/runs/config_eval.log` | `tests/ablations/run_config_eval.py` | `config-eval` | ⬜ TBD |
| `\Floor*` | I (`tab:floor`) | `tasks/runs/leakage_audit_floor.log` | `tests/test_leakage_audit.py` | CPU, seconds | 🟢 Verified |
| `\RunToRun*`, `\PublishedSd*` | §VIII-A | `tasks/runs/panelB.json` vs `tasks/runs/tgn_v*_percookie.log` | — (comparison of two logs) | n/a | 🟢 static (both logs exist and are read directly; no regeneration is owed — the "commit" column is n/a, not pending) |
| `\LatencyPFifty`, `\LatencyPNinetyNine` | §IV-F, §VIII-D | `tasks/runs/serving_client.log` | `tests/test_client.py` | `serve-tgn` | 🟢 Measured |
| `\ParityDelta` | §IV-F | `tests/verify_replay_batching.py` | CPU | — | 🟢 Measured |

### Withdrawn — do not reinstate without a log

Per-component ablation deltas (hashed identity, history features, precursor prior,
structural head). Two mutually contradictory series existed in the repository for the
same quantities — history `+0.163` vs `+0.066`, hashed identity `+0.046` vs `−0.003`,
precursor `+0.013` vs `+0.073` — and neither had a supporting run. The hypothesis that
explained the older series (a Δt shortcut) has since been removed, so the result may not
reproduce. Section VI-D says this explicitly. When regenerated, report as **per-seed
paired** deltas with a Wilcoxon signed-rank test and bootstrap CI
(`report_metrics.paired_delta`), not as a difference of means.

---

## Block 2 — PicoDomain descriptors

These are properties of a public dataset, not model results. Produced 2026-08-05, CPU,
seconds, reproducible without a GPU:

```bash
git clone --depth 1 https://github.com/iHeartGraph/PicoDomain.git data/pico
7zz x -odata/logs data/pico/Zeek_Logs.7z
python -m tests.datasets.picodomain --log-dir data/logs --red-log "data/pico/Red Log.xlsx"
```

Output backing `\Pico*`:

```
[picodomain] events=55436 nodes=625 (users=15 devices=13 sources=8 configs=20 resources=569)
[picodomain] binding coverage: user=90.2% device=97.5% config=94.8%  ttl=36000s
[picodomain] labels: benign=54566 contextual=412 lateral=356 theft=102  window=±90s
StreamData ready: (55436, 10) msg, span=230374s (2.67 d), anomalous fraction = 0.0157
```

`\PicoCovUserShort` (29.0%) and `\PicoCovUserShortest` (11.8%) come from the same
command with `--bind-ttl 3600` and `--bind-ttl 900`. `\PicoSSLrecords`,
`\PicoJAthree`, `\PicoKrbCoverage` and `\PicoUidOverlap` come from the schema
inspection recorded in `docs/datasets.md` §3.1 and §3.3.

These values stand independently of the Block 1 regeneration.

---

## Block 3 — PicoDomain model evaluation

Produced 2026-08-30, GPU (`docker compose --profile eval-picodomain up`), commit
`a11ed74`. Log: `tasks/runs/picodomain_eval_docker.log`. Single run — no seed sweep, no
hyperparameter search on this corpus (same config as the synthetic Panel A/B run).

```
aggregate AUC=0.6658 AP=0.1370
lateral    AUC=0.6402 AP=0.0597 n=356
theft      AUC=0.6957 AP=0.0222 n=102
contextual AUC=0.6810 AP=0.0729 n=397
```

**Recall-at-threshold is withdrawn from this block, deliberately.** The chronological
70/10/20 split is by event count; PicoDomain's red-team campaign is concentrated in the
last ~16% of the stream (first lateral event at index fraction 0.843, first theft at
0.842), so the validation window (indices 70-80%) contains zero lateral and zero theft
events — training is 100% benign in the first 70%, which is correct behaviour for
one-class training, not a defect. Any threshold fit on that validation window is
calibrated against zero positive examples and is uninformative by construction; the
`recall@thr=0.0000` figures in the raw log are an artifact of this, not a measurement of
the model. AUC/AP are computed directly on the test-set ranking and do not depend on the
threshold, so they are the only PicoDomain-model numbers reported in the paper. Do not
add a recall figure for this block without first fixing the split (e.g. time-stratified
rather than count-stratified) — a decision explicitly deferred, since the current split
is the correct training protocol, only the wrong protocol for measuring recall on a
single tail-concentrated campaign.
