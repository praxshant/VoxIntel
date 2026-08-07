# SLURP Dataset Report

Generated from `notebooks/01_dataset_audit.ipynb`.

## Number of samples

| Split | Count |
|---|---|
| train | 50,628 |
| train_synthetic | 69,253 |
| validation | 8,690 |
| test | 13,078 |
| **Total** | **141,649** |

Note: `train_synthetic` (TTS-generated audio) is larger than the real `train`
split. Whether to merge it into `train` for modeling is a decision to make
explicitly in Phase 3/4, not something to leave implicit.

## Number of intents

- **101** unique intent classes
- **18** unique scenarios
- **54** unique actions
(intent = `scenario_action`)

## Average duration

- Minimum: 0.50 sec
- Maximum: 23.36 sec
- **Average: 2.59 sec**
- Median: 2.37 sec
- 96.3% of clips are under 5 seconds
- 1,397 clips fall above the 99th percentile (6.5 sec)

## Average transcript length

Transcript length (words) : min=1, median=6.0, mean=6.65, max=61

## Interesting findings

- The dataset is **very clean**: 0 missing transcripts, 0 missing intent
  labels, 0 missing/unreachable audio files, and 0 corrupted files in a
  2,000-file sample check.
- Audio clips are short and consistent — the vast majority (96.3%) are under
  5 seconds, which keeps ASR training/inference cheap.
- `train_synthetic` outnumbers real `train` audio (69,253 vs 50,628),
  meaning close to 60% of available training audio is TTS-generated rather
  than real human speech.
- The five most common intents (`email_query`, `calendar_set`,
  `qa_factoid`, `email_sendemail`, `play_music`) each have 5,000+ samples,
  while the rarest (`addcontact`, `wemo_on`, `convert`, `settings`,
  `likeness`) have between 3 and 5 samples total.

## Potential issues

- **Severe class imbalance** — imbalance ratio of **2415x** between the
  largest and smallest intent class. `addcontact` has only 3 samples across
  the entire dataset. This affects several downstream decisions:
  - Accuracy alone will be a misleading metric — macro-F1 (or per-class F1)
    is needed to see how the model does on rare intents.
  - Some intent classes may have too few examples to learn reliably at all,
    and may be worth flagging separately in the evaluation report rather
    than silently averaged in.
  - Class weighting or oversampling may be needed during DistilBERT
    fine-tuning.
- **Synthetic vs. real audio split** — since most training audio is
  synthetic, ASR fine-tuned on this data may not generalize as well to real
  human speech at inference time (accents, noise, natural disfluencies).
  Worth evaluating ASR performance separately on real vs. synthetic-derived
  validation/test audio if possible.
- **Duration outliers** — a small number of clips run up to 23.36 seconds
  vs. a 2.37s median. These may be worth inspecting individually (mislabeled
  files, recording errors) before training, or capping/filtering during
  preprocessing.
- **Duplicate / leakage check incomplete** — the notebook's duplicate-check
  cell (duplicate audio paths, duplicate transcripts, and train/test
  transcript overlap) hasn't been executed yet. This should be run and
  reviewed before training, since transcript overlap between train and test
  would inflate reported test performance.

---
