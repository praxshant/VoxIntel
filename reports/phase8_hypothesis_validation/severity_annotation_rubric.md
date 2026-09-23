# Severity annotation rubric — H4 action-risk tiers (Appendix B)

**Purpose.** H4 asks whether risk-aware *deferral* lowers expected **action**
cost. It is currently INCONCLUSIVE because SLURP/FSC label no action severity,
so every misfire is weighted equally. These annotations supply the missing
ground truth: a severity tier per intent, which
[`h4_severity_cost.py`](../../src/analysis/h4_severity_cost.py) turns into the
H4 verdict. **Do not guess** — an unlabelled or invented tier is worse than a
missing one (Appendix B: guessed tiers manufacture the ground truth H4 needs).

## The three tiers

| Tier | Definition | SLURP examples |
|---|---|---|
| **benign** | Read-only or trivially reversible. A wrong action wastes a moment, nothing more. | `qa_*`, `play_music`, `weather_query`, `general_quirky`, `lists_query`, `news_query` |
| **moderate** | Reversible state change. A wrong action is annoying and needs a manual undo. | `alarm_set`, `calendar_set`, `lists_createoradd`, `audio_volume_up`, `music_likeness` |
| **critical** | Hard to reverse, or money-/safety-/privacy-relevant. A wrong action has real external consequences. | `email_sendemail`, `social_post`, `transport_taxi`, `iot_cleaning`, payments, `iot_hue`/lock control |

## Boundary cases (decide these the same way every time)

- **Sends a message to a person / posts publicly** → **critical**, even if "just a text". The recipient sees a wrong message; it cannot be unsent.
- **Books / orders / pays** → **critical** (money leaves; cancellation is friction at best).
- **Controls a physical device** (`iot_*`) → **critical** if it affects safety or cannot be trivially reverted (lock, appliance, heating); **moderate** for a lamp toggle.
- **Sets vs. queries the same domain**: `calendar_set` = moderate, `calendar_query` = benign. Split the intent by verb, not domain.
- **Alarm / reminder / list edits** → **moderate** (reversible, local, no external party).
- If an intent genuinely spans tiers across its utterances, mark it in `notes` and refine per-utterance for that intent only (see below).

## Procedure

1. **Annotate at the intent level first.** Each row of
   [`severity_labels_TEMPLATE.csv`](severity_labels_TEMPLATE.csv) is one of the
   70 SLURP intents, with an example transcript and utterance count. Fill
   `tier_annotator_1..3` with one of `benign` / `moderate` / `critical`.
2. **2–3 independent annotators.** No conferring on the first pass.
3. **Agreement gate.** Compute Cohen's κ (2 annotators) or Fleiss' κ (3). If
   **κ < 0.7**, revise the rubric's boundary examples and re-annotate the
   disagreements — do not proceed to the verdict below κ 0.7.
4. **Adjudicate.** Resolve disagreements by discussion; write the agreed tier
   into `tier_final`. Every row must have a `tier_final` in the tier vocab.
5. **Per-utterance refinement (optional).** Only for an intent flagged in step
   1 as spanning tiers.

## From labels to the H4 verdict

```python
from src.analysis.h4_severity_cost import load_labels, run_h4, TEMPLATE
res = run_h4(load_labels(TEMPLATE))          # or a copy with tier_final filled
print(res["H4_supported"], res["cost_risk_defer"], res["cost_always_execute"])
```

`run_h4` refits the frozen-split intent risk model (same split/model as H1/H3),
then prices three policies on the frozen **test** set: (a) always-execute,
(b) confidence-threshold deferral, (c) VoxIntel-R risk-threshold (Bayes)
deferral. **H4 is supported iff (c) is cheaper than both (a) and (b) with a
paired-bootstrap CI excluding 0.**

## Cost units

`DEFAULT_COST = {benign: 1, moderate: 5, critical: 25}`, `DEFER_COST = 2` are
**illustrative** — they only let the harness run. Replace them with the costs
your deployment context justifies (Appendix B allows a continuous cost per
tier); pass them as `run_h4(labels, cost=..., defer_cost=...)` and record the
choice. Report the verdict's sensitivity to these weights, as NB22 does.

## Cleaner substrate

SLURP smart-speaker intents skew low-severity (most utterances are benign
queries). A corpus with genuinely irreversible actions (in-car / home-automation
commands, or a severity-stratified benchmark) is the sharper test; record the
dataset choice and its licence if you switch.
