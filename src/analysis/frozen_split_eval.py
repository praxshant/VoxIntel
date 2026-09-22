"""Frozen held-out split validation of the VoxIntel-R risk models.

Why this exists
---------------
Notebook 16 reported SLURP ROC-AUCs from 5-fold cross-validation over the *whole*
evaluation population, with no frozen test set held aside. Because the feature
families (A/B/C) and the model were also chosen while looking at those same
numbers, the headline "intent-only beats combined" (H1 not supported) was flagged
downstream as *provisional*. This script removes that caveat: it designates one
stratified test split that is never used for any modelling decision, fits the same
models NB16 used, and reports held-out metrics plus a paired bootstrap CI for H1.

It reuses the cached inference-time features (reports/.../voxintel_r_features.csv),
so it needs no audio, no GPU, and no ASR/NLU re-decode. The reference transcript is
used only to build the `intent_failed` target, never as a feature (leakage guard
below asserts this).

Run: python src/analysis/frozen_split_eval.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "reports" / "phase6_voxintel_r_slurp" / "voxintel_r_features.csv"
OUT_DIR = ROOT / "reports" / "phase6_voxintel_r_slurp"
MODEL_OUT = ROOT / "models" / "voxintel_r_intent_rf.joblib"

TARGET = "intent_failed"
ASR_FEATURES = ["asr_mean_confidence", "asr_min_confidence", "asr_std_confidence",
                "asr_median_confidence", "asr_mean_entropy", "asr_max_entropy",
                "asr_std_entropy", "asr_num_frames", "audio_duration"]
INTENT_FEATURES = ["intent_confidence", "intent_entropy", "intent_margin"]
FAMILIES = {"A_asr": ASR_FEATURES, "B_intent": INTENT_FEATURES,
            "C_combined": ASR_FEATURES + INTENT_FEATURES}
SEED = 42
TEST_SIZE = 0.30
N_BOOT = 2000
N_REPEATS = 20  # repeated-split stability of the C-B gap


def _rf():
    # Same config as notebook 16.
    return RandomForestClassifier(n_estimators=300, max_depth=None,
                                  class_weight="balanced", random_state=SEED, n_jobs=-1)


def _logreg():
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])


def _metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "f1": f1_score(y, pred, zero_division=0),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0)}


def _fit_predict(model_fn, feats, Xtr, ytr, Xte):
    m = model_fn()
    m.fit(Xtr[feats], ytr)
    return m, m.predict_proba(Xte[feats])[:, 1]


def _paired_bootstrap_auc_gap(y, p_c, p_b, n=N_BOOT, seed=SEED):
    """H1: is combined-C's held-out AUC above intent-only-B's? Same resampled
    indices for both models each iteration (paired)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); n_obs = len(y)
    gaps = []
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        yb = y[idx]
        if yb.min() == yb.max():  # need both classes for AUC
            continue
        gaps.append(roc_auc_score(yb, p_c[idx]) - roc_auc_score(yb, p_b[idx]))
    gaps = np.array(gaps)
    return {"mean_gap_C_minus_B": float(gaps.mean()),
            "ci95_low": float(np.percentile(gaps, 2.5)),
            "ci95_high": float(np.percentile(gaps, 97.5)),
            "frac_C_beats_B": float((gaps > 0).mean()), "n_boot": len(gaps)}


def main():
    if not FEATURES.exists():
        raise SystemExit(f"missing cached features: {FEATURES}\nRun notebook 16 first.")
    df = pd.read_csv(FEATURES)
    y = df[TARGET].astype(int)

    # Leakage guard: no reference/ground-truth column may enter the feature sets.
    banned = {"ground_truth_intent", "asr_transcript", "reference", "intent_failed"}
    assert not (set(sum(FAMILIES.values(), [])) & banned), "leakage: reference feature in family"

    Xtr, Xte, ytr, yte = train_test_split(df, y, test_size=TEST_SIZE,
                                          stratify=y, random_state=SEED)
    print(f"train {len(Xtr)}  test {len(Xte)}  test failure-rate {yte.mean():.3f}")

    rows, proba = [], {}
    for fam, feats in FAMILIES.items():
        for name, fn in [("RandomForest", _rf), ("LogisticRegression", _logreg)]:
            _, p = _fit_predict(fn, feats, Xtr, ytr, Xte)
            proba[(fam, name)] = p
            rows.append({"experiment": fam, "model": name, "n_features": len(feats),
                         **_metrics(yte, p)})
    comp = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    comp.to_csv(OUT_DIR / "frozen_split_model_comparison.csv", index=False)

    # H1 on the frozen test set (RandomForest, the stronger model).
    h1 = _paired_bootstrap_auc_gap(yte.values, proba[("C_combined", "RandomForest")],
                                   proba[("B_intent", "RandomForest")])

    # Repeated-split stability of the C-B gap (independent of the frozen seed).
    gaps = []
    for s in range(N_REPEATS):
        a, b, ya, yb = train_test_split(df, y, test_size=TEST_SIZE, stratify=y, random_state=s)
        _, pc = _fit_predict(_rf, FAMILIES["C_combined"], a, ya, b)
        _, pb = _fit_predict(_rf, FAMILIES["B_intent"], a, ya, b)
        gaps.append(roc_auc_score(yb, pc) - roc_auc_score(yb, pb))
    gaps = np.array(gaps)
    stability = {"mean_gap_C_minus_B": float(gaps.mean()), "std": float(gaps.std()),
                 "n_splits": N_REPEATS, "n_splits_C_beats_B": int((gaps > 0).sum())}

    best = comp.iloc[0]
    h1_supported = h1["ci95_low"] > 0  # C beats B only if the whole CI is positive
    summary = {"n_total": len(df), "n_train": len(Xtr), "n_test": len(Xte),
               "test_failure_rate": float(yte.mean()), "seed": SEED, "test_size": TEST_SIZE,
               "best_model": {"experiment": best.experiment, "model": best.model,
                              "roc_auc": float(best.roc_auc), "pr_auc": float(best.pr_auc)},
               "H1_asr_adds_beyond_intent": {"supported": bool(h1_supported), **h1},
               "H1_repeated_split_stability": stability}
    (OUT_DIR / "frozen_split_summary.json").write_text(json.dumps(summary, indent=2))

    # Persist the headline model (intent-only RF) fit on ALL data, for serving.
    # Fit on a plain array (no column names) so the serving layer can score a
    # bare feature list without sklearn's feature-name warning.
    final = _rf(); final.fit(df[INTENT_FEATURES].values, y)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    dump({"model": final, "features": INTENT_FEATURES, "threshold": 0.5,
          "trained_on": "SLURP dev (8688)", "target": TARGET}, MODEL_OUT)

    print(comp.to_string(index=False))
    print(f"\nH1 (C-B AUC gap): mean {h1['mean_gap_C_minus_B']:+.4f} "
          f"CI95 [{h1['ci95_low']:+.4f}, {h1['ci95_high']:+.4f}]  "
          f"-> {'SUPPORTED' if h1_supported else 'NOT SUPPORTED'}")
    print(f"repeated splits: C beats B in {stability['n_splits_C_beats_B']}/{N_REPEATS} "
          f"(mean gap {stability['mean_gap_C_minus_B']:+.4f})")
    print(f"saved model -> {MODEL_OUT}")
    return summary, comp


def demo():
    """Self-check: runs the full analysis and asserts the core invariants."""
    summary, comp = main()
    assert len(FAMILIES["A_asr"]) == 9 and len(FAMILIES["B_intent"]) == 3
    assert len(FAMILIES["C_combined"]) == 12
    b = comp[(comp.experiment == "B_intent") & (comp.model == "RandomForest")].iloc[0]
    assert b.roc_auc > 0.80, f"intent-only RF AUC unexpectedly low: {b.roc_auc}"
    # H1 verdict must be internally consistent with the CI.
    h1 = summary["H1_asr_adds_beyond_intent"]
    assert h1["supported"] == (h1["ci95_low"] > 0)
    print("demo OK")


if __name__ == "__main__":
    demo()
