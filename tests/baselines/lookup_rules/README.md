# Baseline: stateful lookup rules

No learning. The score is the number of history-dependent binding flags that fire on an
event (`cfg|dev_new`, `cfg|usr_new`, `dev|usr_new`, `src|usr_new`, `role_changed`; see
`src/data/lookup_rules.py`) — what a SIEM correlation rule does with a dictionary of
"seen so far" pairs. It is the baseline a reviewer runs first against a synthetic
benchmark: if it separates a class, the class measures the generator, not the model.

Same stream, seed, chronological split and test window as the TGN and the Isolation
Forest. Reported **per class** (lateral, cred-theft) as benign-vs-class AUC / AP in the
test window, plus recall / FPR at the smallest integer threshold whose benign
*validation* FPR is ≤ `target_fpr` (a quantile is degenerate on integer scores). No
aggregate: policy / contextual / exfil belong to the OPA and the sensor layer.

- `lookup_rules_baseline(cfg, stream=None)` — **primary**. `proto-self` gate with
  ground-truth benign labels only through the *training* window; validation and test are
  self-gated (only events no binding rule fired on commit). This is what the TGN gets — its
  validation / test replays commit on `not signal_dirty`, never on labels — and what a
  deployment can have.
- `lookup_rules_val_baseline(cfg, stream=None)` — **sensitivity**. As above, but the state
  also receives ground-truth labels through the *validation* window (the rule audit's
  protocol). An oracle over validation, hence an upper bound for the rules: on v5 it lifts
  lateral AUC by 0.04–0.08 and the validation-calibrated threshold lands at ~10 % benign
  FPR on the test window (target 1 %).
- `lookup_rules_all_baseline(cfg, stream=None)` — `gate="all"`: commits everything, uses
  no label at all.

Run (CPU, seconds; no torch training):

```bash
python tests/baselines/lookup_rules/lookup_rules_baseline.py
```

Tests: `tests/test_lookup_rules_baseline.py`.
