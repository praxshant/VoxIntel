# Paper appendices — deferred experiments, fully specified

These two experiments are named in the paper's Future Work. Appendix A has
since been **run on its LM-free feature subset** (result below); Appendix B
still needs human annotation. Rather than approximate or fabricate what remains,
this document specifies each precisely enough to run and to judge — including
the decision rule that would change the paper's conclusions. Both build on the
frozen 70/30 SLURP split (seed 42) used in
[`src/analysis/frozen_split_eval.py`](../src/analysis/frozen_split_eval.py) and
reported in [`FROZEN_SPLIT_RESULTS.md`](phase6_voxintel_r_slurp/FROZEN_SPLIT_RESULTS.md).

---

## Appendix A — N-best / decoder-LM uncertainty features (confirmatory for H1)

**Question it settles.** H1 asks whether ASR uncertainty adds predictive signal
beyond intent uncertainty. With 9 CTC frame-level ASR features (family A) the
answer was *no*: held-out ΔAUC(C−B) = −0.037, 95% CI [−0.051, −0.025], combined
beats intent-only in 0/20 splits. A reviewer may reasonably ask whether *richer*
ASR uncertainty — from an N-best list and an external language model — changes
this. This decode produces those features so the result is confirmed, not assumed.

**Decode configuration.** Wav2Vec2 CTC logits → `pyctcdecode` beam search with a
KenLM n-gram LM (4-gram, trained on SLURP-train transcripts or a general corpus).
beam_width = 100, N-best = 10, LM weight α and word-insertion β tuned on train
only (never on the frozen test set).

**Features (family A′, all reference-free, inference-time only):**
- N-best posterior entropy over softmax-normalised hypothesis scores.
- Hypothesis score gap: top1 − top2 (normalised).
- LM score of the top hypothesis; acoustic − LM score disagreement.
- Mean token-level posterior of the top hypothesis.
- N-best dispersion: mean pairwise Levenshtein distance among the 10 hypotheses.

**Protocol.** Refit B vs (B ∪ A′) on the *same* frozen split and the same RF /
LogReg configs; report held-out ΔAUC with the identical paired bootstrap (2,000
resamples) and 20-split stability check already in the script.

**Decision rule.** H1 flips to *supported* only if the whole 95% CI of
ΔAUC(B∪A′ − B) is > 0 on the frozen split **and** it replicates in ≥ 11/20
splits. Prior: intent-native uncertainty is already a near-sufficient statistic
for downstream failure (B AUC 0.912 ≈ CV 0.911), so A′ is expected to add ≤ noise.

**Compute.** One beam-search decode over the 8,688 SLURP-dev clips plus a KenLM
build. No ASR/intent retraining. On the order of GPU-hours, not GPU-days.

**Result (LM-free subset, run 2026-09-23, RTX 3050).** Beam-search decode
(`pyctcdecode`, beam_width = 100, N-best = 10, **no LM**) over all 8,688
SLURP-dev clips on the fine-tuned checkpoint (`checkpoint-41500`, WER 0.199),
computing the 4 reference-free A′ features — N-best posterior entropy, top1−top2
gap, mean token posterior, N-best Levenshtein dispersion. The 2 LM features
(LM score, acoustic−LM disagreement) are omitted: KenLM has no Windows wheel and
was not built. On the frozen split, **B ∪ A′ *underperforms* intent-only B**:
ROC-AUC **0.890 vs 0.912**, ΔAUC(B∪A′ − B) = **−0.022, 95% CI [−0.033, −0.010]**
(0 / 2,000 bootstrap resamples positive), and B∪A′ beats B in **0 / 20** splits.
The decision rule (CI > 0 **and** ≥ 11/20) fails by a wide margin, so **H1
remains NOT SUPPORTED — now with beam-search N-best evidence, not just
frame-level CTC.** This matches the stated prior: intent-native uncertainty is a
near-sufficient statistic (B AUC 0.912 reproduces the paper's headline 0.911),
so richer ASR uncertainty adds noise, not signal. The consistent sign across
family A (−0.037), tuned NB20 (−0.017) and N-best (−0.022) makes a reversal from
the 2 still-untested LM features unlikely; a KenLM build is the one remaining
step. Harness [`src/analysis/appendix_a_nbest.py`](../src/analysis/appendix_a_nbest.py);
features `appendix_a_nbest_features.csv`; verdict `appendix_a_nbest_h1.json`
(both in [`phase8_hypothesis_validation/`](phase8_hypothesis_validation/)).

---

## Appendix B — Severity-label annotation protocol (makes H4 a real test)

**Question it settles.** H4 asks whether risk-aware deferral lowers expected
*action* cost. It is currently untestable: SLURP and FSC label no action severity,
so every misfire is weighted equally and H4 is only a sensitivity analysis
(marked INCONCLUSIVE on both corpora). This protocol supplies the missing ground
truth.

**Label schema — 3 action-risk tiers:**
- **Benign** — read-only / trivially reversible (query weather, play music).
- **Moderate** — reversible state change (set alarm, add a list item).
- **Critical** — hard-to-reverse, safety- or money-relevant (send message, make
  payment, unlock door, control an appliance).

Optionally a continuous cost per tier in a common unit.

**Unit.** Annotate at the intent-class level first (SLURP ≈ 60 intents — cheap,
covers every utterance); refine per-utterance only where one intent spans tiers.

**Reliability.** 2–3 annotators, a written rubric with boundary examples,
inter-annotator agreement by Cohen's/Fleiss' κ, adjudication of disagreements;
require κ ≥ 0.7 before the labels are used.

**How H4 becomes a test.** Define a cost matrix `C(tier, error_type)`. Expected
cost of a policy = Σ over the test set of P(fail)·execute-cost + defer-cost.
Compare three policies on the frozen test set: (a) always-execute, (b)
confidence-threshold deferral, (c) VoxIntel-R risk-threshold deferral. H4 is
**supported** iff (c) has lower expected cost than both (a) and (b) with a paired
bootstrap CI excluding 0.

**Dataset note.** SLURP smart-speaker intents skew low-severity; a corpus with
genuinely high-severity actions (in-car / home-automation commands with
irreversible effects, or a severity-stratified benchmark) is the cleaner
substrate. Record the dataset choice and its licence.

**Why it is not approximated.** Assigning severities by guesswork would
manufacture the very ground truth H4 needs. The honest status is *protocol
specified, labels not yet collected.*

**Harness (this repo).** The machinery is built and self-checked:
[`src/analysis/h4_severity_cost.py`](../src/analysis/h4_severity_cost.py)
generates the 70-intent annotation task
([`reports/phase8_hypothesis_validation/severity_labels_TEMPLATE.csv`](phase8_hypothesis_validation/severity_labels_TEMPLATE.csv)),
and `run_h4(load_labels(...))` refits the frozen-split risk model and prices the
three policies on the frozen test set with paired-bootstrap CIs. It runs today
on illustrative tiers only; the real verdict waits on the κ ≥ 0.7 annotation
above. Rubric with boundary examples:
[`severity_annotation_rubric.md`](phase8_hypothesis_validation/severity_annotation_rubric.md).
