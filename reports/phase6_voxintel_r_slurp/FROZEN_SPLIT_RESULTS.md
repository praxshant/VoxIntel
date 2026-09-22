# Frozen held-out split validation (SLURP) — removes the "provisional" caveat on H1

**Script:** [`src/analysis/frozen_split_eval.py`](../../src/analysis/frozen_split_eval.py)
· **Inputs:** `voxintel_r_features.csv` (cached, no re-decode)
· **Outputs:** `frozen_split_model_comparison.csv`, `frozen_split_summary.json`, `models/voxintel_r_intent_rf.joblib`

## Why
Notebook 16 reported SLURP ROC-AUCs from 5-fold CV over the *whole* evaluation
population, with no test set held aside. Feature families (A/B/C) and the model
were chosen while looking at those same numbers, so downstream the headline
"intent-only beats combined" (H1 not supported) was flagged as **provisional**.
This designates one stratified test split that is never used for any modelling
decision.

## Setup
- 8,688 SLURP-dev utterances → stratified **70/30** split (seed 42), test *n* = 2,607, failure rate 21.3%.
- Same models as NB16: `RandomForest(n_estimators=300, class_weight="balanced")`, `LogisticRegression` + `StandardScaler`.
- Feature families: **A** = 9 ASR-native, **B** = 3 intent-native (`intent_confidence`, `intent_entropy`, `intent_margin`), **C** = 12 combined.
- Leakage guard asserts no reference/ground-truth column enters any family.

## Held-out results (RandomForest)

| Family | Features | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|
| **B — intent-only** | 3 | **0.912** | 0.789 | 0.087 |
| C — combined | 12 | 0.875 | 0.688 | 0.105 |
| A — ASR-only | 9 | 0.615 | 0.311 | 0.165 |

The held-out B-intent AUC (0.912) matches the CV number (0.911) almost exactly —
the estimate was stable, not lucky.

## H1 — does ASR uncertainty add beyond intent uncertainty?

- **Paired bootstrap on the test set** (2,000 resamples): C−B AUC gap = **−0.037**, 95% CI **[−0.051, −0.025]** — entirely negative. C beats B in **0%** of resamples.
- **Repeated-split stability** (20 independent stratified splits): mean gap **−0.030 ± 0.006**; C beats B in **0/20** splits.

**Verdict: H1 NOT SUPPORTED — now on a frozen held-out split, not just CV.** Adding
ASR-native confidence features does not help and slightly hurts. Intent-native
uncertainty is a near-sufficient statistic for downstream intent failure. This
promotes the SLURP result from *provisional* to a confirmed negative result.

The persisted intent-only RF (`models/voxintel_r_intent_rf.joblib`) is what the
serving endpoint (`src/serving/app.py`) loads.
