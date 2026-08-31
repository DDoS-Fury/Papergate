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
| **Block 1** — synthetic-stream results | 🟢 **MEASURED** (2026-08-31) | — (see "2026-08-31 regeneration" below) |
| **Block 2** — PicoDomain descriptors | 🟢 **MEASURED** | `tests.datasets.picodomain` |
| **Block 3** — PicoDomain model evaluation | 🟢 **MEASURED** (single run) | `tests/eval_picodomain.py`, see below |

`main.tex` carried `\preliminarytrue`, which stamps a banner on page 1 and on the
affected tables/figures, since Block 1 was blocked on GPU regeneration. As of
2026-08-31 all three pieces of Block 1 (Panel A, Panel B, credential-theft deltas) are
regenerated and reproducible from HEAD — see below for the specific evidence — so the
flag was flipped to `\preliminaryfalse` in the same session as this update.

> 2026-08-30: this row was briefly (commit `a11ed74`) marked "🟢 SYNCED" and the flag
> flipped to `\preliminaryfalse` without any accompanying regeneration — no new
> `tasks/runs/` artifact exists for that commit, and `results.tex`/`sections/*.tex`
> still carried the same `\prelim{}`-flagged, pre-de-leakage, mixed-protocol numbers
> described below. Reverted; see `tasks/todo.md` 2026-08-30 audit section. **The
> 2026-08-31 flip below is not a repeat of that mistake**: unlike `a11ed74`, this flip
> is accompanied by (a) a new `tasks/runs/panelA.json` with 3-seed data for all 6
> models, (b) `results.tex` macros updated to match it (verified against
> `tab_baselines.tex` and recomputed independently via `report_metrics.mean_std`,
> ddof=1), (c) `git log -1 --format=%ci -- src tests` = `2026-08-19 15:55:58 +0200`,
> before the run's `2026-08-31T09:19:22+00:00` timestamp, confirming HEAD's code
> produced this data, (d) a green `make && make check`, and (e) the log-capture gap
> below disclosed rather than hidden.

### 2026-08-31 regeneration — Panel A (Table III)

`docker compose --profile regen-report up`, commit `031b442`. Output:
`tasks/runs/panelA.json` (`meta.generated` = `2026-08-31T09:19:22+00:00`, 3 seeds ×
6 models — TGN, TGN-2node, static GNN, One-Class SVM, Isolation Forest, XGBoost — all
present), `docs/latex/generated/tab_baselines.tex`.

**Known gap, disclosed rather than hidden**: `tasks/runs/regen_report.log`'s capture is
truncated mid-run, at Panel A / TGN / seed=42, before that seed's inference phase even
finishes printing — it does **not** cover the full run that produced `panelA.json`
(the JSON's `meta.generated` timestamp is ~14h after the log's last line). The JSON
itself is not in doubt: all 3 seeds are present for all 6 models, `src/train_tgn.py`
fully seeds `torch`/`numpy`/`random` and calls
`torch.use_deterministic_algorithms(True, ...)` (verified in-session:
`tasks/runs/panelB.json`'s raw floating-point values are bit-identical between a
2026-08-18 and a 2026-08-30 rerun of the same seeds), and `git log` confirms `src`/
`tests` were untouched between the code that would have produced this data and HEAD.
But the log transcript itself cannot be cited as evidence past line 408 — if this run
is ever disputed, re-run `docker compose --profile regen-report up` and diff against
the JSON checked in here (determinism means it should reproduce bit-for-bit).

**Protocol fix realized**: `run_panel_a_tgn` in `tests/regen_report_tables.py` builds
the TGN's config identically to every baseline row (`dataclasses.replace(TGNConfig(),
seed=seed)`, deployable dev:guest, v4) — the "TGN row = v3 per-cookie, baselines = v4
deployable" mismatch described in older drafts of this document no longer exists. (The
LaTeX-emission code had stale labels claiming otherwise — `"TGN (v3, per-cookie)"` and
a `"Protocollo MISTO (storico)"` comment — even though the underlying run was already
unified; fixed in the same commit as this note, cosmetic only, did not affect any
number.)

**Not incorporated — flagged, not lost**: `panelA.json`/`tab_baselines.tex` also
measured a `tgn_2node` baseline (`tests/baselines/tgn_2node`) not currently in
`tab:baselines`. It is competitive with the full 5-node TGN under the *same* protocol
and seeds: it **beats** the TGN on aggregate recall (0.625 vs 0.550), lateral recall
(0.170 vs 0.161) and aggregate AP (0.822 vs 0.801), and **ties** on aggregate AUC
(0.854 vs 0.853); the 5-node TGN wins only on lateral AUC (0.721 vs 0.659). Values are
recorded as `\AggAucTGNii` etc. in `results.tex` but not wired into a table row —
deciding whether/how to present a same-family baseline that outperforms the proposed
method on 3 of 5 metrics is a narrative decision for the paper's authors, not a data-
sync task. Do not drop this without addressing it in the text.

---

## Block 1 — why it *was* preliminary (resolved 2026-08-31)

Two independent problems, both documented in `tasks/todo.md` §6 and §7 — kept here as
history, since a future regeneration needs to know what was originally wrong:

1. **Not reproducible from HEAD.** The values were produced on 2026-06-24, before the
   generator de-leakage of 2026-08-03. That generator no longer exists. The runs that
   produced them were clean *on the lateral class* — `AUC(node_feat[dst,3])` on lateral
   was 0.4899 on the pre-leak generator, i.e. chance, so the reported lateral AUC was
   genuinely earned — but the de-leaked task is measurably harder (single-feature floor
   on lateral: 0.920 → 0.603, per `tasks/runs/leakage_audit_floor.log`), so the numbers
   moved. **Resolved**: superseded by the 2026-08-31 `panelA.json` run, reproducible
   from HEAD (see status table above).
2. **Panel A mixed protocols.** The TGN row came from a v3 per-cookie run; every
   baseline row came from a v4 deployable run. **Resolved**: `run_panel_a_tgn` now runs
   the TGN under the baseline protocol, and the 2026-08-31 run reflects that fix (see
   "Protocol fix realized" above).

### Regeneration

All on the GPU box via Compose — never a local CPU venv (`tasks/lessons.md`). Panel A/B
and config-eval are what gate Block 1's status and are now done (see status table
above); `ablations` is permanently withdrawn (§VI-D); `arch-sweep` and
`guest-device-eval` are not currently cited by any macro in `results.tex` and do not
gate anything here — re-run them only if new sections start depending on their output.

```bash
docker compose --profile regen-report      up   # Panels A and B — done 2026-08-31/30
docker compose --profile config-eval       up   # credential-theft deltas — done 2026-08-18
docker compose --profile ablations         up   # per-component ablations (withdrawn, do not re-run for Block 1)
docker compose --profile arch-sweep        up   # not currently cited in the paper
docker compose --profile guest-device-eval up   # not currently cited in the paper
```

### Macro → source map

| Macros | Table | Source of record | Generator script | Compose profile | Commit |
|---|---|---|---|---|---|
| `\AggAuc*`, `\AggAp*`, `\LatAuc*`, `\LatRec*`, `\AggRec*` | III (`tab:baselines`) | `tasks/runs/panelA.json` (generated 2026-08-31T09:19:22Z) | `tests/regen_report_tables.py` | `regen-report` | 🟢 `031b442` |
| `\Bagg*`, `\Blat*`, `\Bfpr*` | IV (`tab:panelb`) | `tasks/runs/panelB.json` (generated 2026-08-30T19:24:30Z) | `tests/regen_report_tables.py` | `regen-report` | 🟢 `031b442` |
| `\TheftRecallDelta`, `\TheftLateralDelta`, `\TheftRecOn/Off`, `\TheftAucOn/Off`, `\TheftN` | §VI-C prose | `tasks/runs/config_eval.log` (2026-08-18) | `tests/ablations/run_config_eval.py` | `config-eval` | 🟢 verified against log in-session |
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
