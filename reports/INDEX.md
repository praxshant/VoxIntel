# Reports Index

Artifacts are grouped into phase folders that mirror the notebook pipeline
(`notebooks/01…18`). Notebooks historically wrote to a **flat** `reports/`
directory, so if you re-run an old notebook it will recreate the file at the
old flat path; move the new output into the matching phase folder afterwards
(or update the notebook's output path).

Headline numbers are quoted from the JSON/CSV summaries in each folder.

| Phase | Folder | Notebooks | Theme |
|---|---|---|---|
| 1 | `phase1_dataset_audit/` | 01–02 | SLURP dataset + audio audit |
| 2 | `phase2_asr/` | 03–06 | Baseline ASR + Wav2Vec2 fine-tuning |
| 3 | `phase3_intent_propagation/` | 07–08 | Intent upper bound + ASR→intent propagation |
| 4 | `phase4_taxonomy/` | 09–13 | Error taxonomy: broken → forensic audit → fixed → consolidated → validated |
| 5 | `phase5_semantic_risk/` | 14–15 | High-impact error analysis + (reference-aware) semantic risk prediction |
| 6 | `phase6_voxintel_r_slurp/` | 16 | Reference-free VoxIntel-R risk model (SLURP) |
| 7 | `phase7_reliability_cross_dataset/` | 17–18 | Cross-dataset (FSC) validation, calibration, selective prediction, cost-sensitivity |

`voxintel_research_audit.md` (repo-root of `reports/`) is the August-2026
external peer-review audit covering notebooks 01–12.

---

## Phase 1 — Dataset & audio audit (NB 01–02)
- `dataset_report.md` — SLURP split sizes (≈50.6k train / 8.7k dev / 13.1k test), intent/scenario counts, duration stats, class imbalance.

## Phase 2 — ASR baseline + fine-tuning (NB 03–06)
Headline: corpus **WER 58.4% → 38.4%**, **CER 30.9% → 18.7%** after fine-tuning.
- `baseline_predictions.csv` — 100-sample sanity check (NB03).
- `full_validation_predictions.csv` — baseline Wav2Vec2 over full dev set (NB04).
- `finetuned_validation_predictions.csv`, `finetuned_train_predictions.csv` — fine-tuned ASR outputs (NB05/06).
- `master_transcripts.csv` — ground-truth + baseline-ASR + fine-tuned-ASR transcripts aligned per sample (feeds NB08).

## Phase 3 — Intent upper bound + propagation (NB 07–08)
Headline intent accuracy: **ground-truth 0.858 / baseline-ASR 0.385 / fine-tuned-ASR 0.734** (macro-F1 0.703 / 0.308 / 0.555).
- `classification_report.json`, `confusion_matrix.png`, `calibration_ground_truth.png`, `tsne_intent_embeddings.png` — DistilBERT upper-bound diagnostics (NB07).
- `intent_metric_comparison.png`, `metric_comparison.csv`, `confidence_*` , `per_intent_f1_comparison.csv`, `per_scenario_f1_comparison.csv` — 3-way transcript-source comparison (NB08).
- `error_propagation_results.csv` — per-sample cascade results (basis for phases 4–7).
- `intent_predictions_{baseline,finetuned,ground_truth}.csv`, `*_intent_probabilities.npy` — cached predictions/probabilities.

## Phase 4 — Error taxonomy (NB 09–13)
Story: NB09 taxonomy was **silently broken** (100% `no_error`) → NB10 forensic audit caught it → NB11 fixed it → NB12 archived broken + promoted fixed to canonical → NB13 validated/refined.
- `debugging_summary_07_08_09.json`, `debugging_verdict_07_08_09.csv` — NB10 forensic audit verdicts.
- `error_taxonomy_dataset.csv`, `error_type_distribution.csv`, `error_impact_matrix.csv`, `intent_fragility.csv`, `failure_gallery.csv`, `recovery_summary.csv` — **canonical (fixed)** taxonomy outputs (NB11, promoted by NB12).
- `*_fixed.png` — correct NB11 charts; the non-suffixed `.png` siblings are the earlier broken-run charts kept for history.
- `taxonomy_fix_summary.json`, `canonical_reports_manifest.csv`, `consolidation_summary_12.json`, `post_consolidation_verdict.csv`, `research_continuation_plan.csv` — NB12 consolidation records.
- `taxonomy_validation_*`, `proper_noun_subtype_*`, `taxonomy_validation_summary_13.json` — NB13 validation. `proper_noun_error` dominates (baseline ≈87%); flagged as still too coarse.
- `archive_broken_taxonomy_20260807_215056/` — original broken NB09 CSVs, archived not deleted.

## Phase 5 — Semantic impact + risk prediction (NB 14–15)
NB14 — semantic impact by category (baseline intent-failure rate): `number_error` 0.68, `proper_noun_error` 0.65 (most harmful); `deletion`/`entity_error` benign. Fine-tuning reduces both frequency and impact.
NB15 — taxonomy features add predictive signal over proxy metrics: combined-vs-proxy AUC **+0.046** (baseline target) / **+0.134** (fine-tuned target), both bootstrap-significant. **Caveat: features are post-hoc / ground-truth-aware, not deployable.**
- `high_impact_error_*`, `category_recovery_analysis.*`, `category_conditioned_fragility.csv`, `fragile_*`, `persistent_/recovered_high_impact_errors.csv`, `recovery_by_error_type.csv`, `semantic_impact_matrix.csv`, `semantic_case_studies.csv`, `taxonomy_vs_editcount_signal.csv` — NB14.
- `semantic_error_risk_*` (dataset, model_comparison, cv_summary, bootstrap_significance, feature_importance ×4, score_bins, case_studies, summary), `intent_risk_{baseline,finetuned}.csv` — NB15.

## Phase 6 — Reference-free VoxIntel-R, SLURP (NB 16)
First leakage-free risk model (ASR-native + intent-native inference-time features only; reference used only for the target label). Best model = **intent-confidence-only RF ROC-AUC 0.911**; combined-C RF 0.883; ASR-only ≈0.62. **Combined did NOT beat intent-confidence alone.**
- `voxintel_r_features.csv`, `voxintel_r_model_comparison.{csv,png}`, `voxintel_r_feature_importance.{csv,png}`, `voxintel_r_predictions.csv`, `voxintel_r_summary.json`.
- ⚠️ These SLURP numbers use 5-fold CV over the full evaluation population (no frozen held-out split) — treated as **provisional** downstream.

## Phase 7 — Cross-dataset reliability & decisions (NB 17–18)
Cross-dataset external validation on **Fluent Speech Commands (FSC)** plus calibration / selective prediction / cost-sensitivity, evaluated per dataset under its own contract (never pooled).

Verdicts (`voxintel_r_final_verdicts_combined.csv`, `*_final_summary.json`):

| Hypothesis | SLURP | FSC |
|---|---|---|
| H1 — ASR uncertainty adds beyond intent uncertainty | **NOT SUPPORTED** | **NOT SUPPORTED** |
| H2 — calibration | SUPPORTED (ECE 0.019) | PARTIALLY (ECE 0.017, MCE 0.66) |
| H3 — selective prediction (AURC) | SUPPORTED (risk 0.059 < intent 0.083 < asr 0.187 < always 0.213) | PARTIALLY (beats always-execute; not intent-confidence) |
| H4 — cost-sensitive | INCONCLUSIVE (no split) | INCONCLUSIVE (simulated severity) |

- `fsc_*` — FSC dataset audit, ASR/intent predictions, VoxIntel-R features, model comparison, calibration, reliability, risk-coverage, selective, cost-sensitivity, ablation (NB17 v2).
- `slurp_voxintel_r_*` — SLURP final reliability evaluation (NB18).
- `notebook18_*` — NB18 consolidated per-stage tables + artifact manifest + audit log.
- `voxintel_r_cross_dataset_{summary.csv,synthesis.json}` — cross-dataset synthesis.
