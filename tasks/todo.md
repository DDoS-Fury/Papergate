# Task: Align TGN Synthetic Training Data with ZTA Rego Policies

## Objective
Make the synthetic training data generated for the TGN model reflect the real
Zero Trust Architecture policies defined in `docs/policies.txt` (Rego).

## Constraints
- No data-flow changes — only change the *content* of generated examples
- No regressions in existing functionality
- Any non-explicitly-requested modification requires user approval

## Status: 🔍 Research Phase

### Phase 1 — Research (in progress)
- [ ] Analyze current synthetic data generation (`src/data/synthetic.py`)
- [ ] Analyze TGN training pipeline (`src/train_tgn.py`, `src/model/tgn.py`)
- [ ] Identify all hardcoded values that diverge from Rego policies
- [ ] Create detailed gap analysis: current values vs. policy values

### Phase 2 — Plan
- [ ] Create implementation plan with precise file/line changes
- [ ] Get user approval on plan

### Phase 3 — Implementation
- [ ] (TBD after research)

### Phase 4 — Verification
- [ ] Run training pipeline to confirm no regressions
- [ ] Verify generated data matches policy constraints
