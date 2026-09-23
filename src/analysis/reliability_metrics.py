"""Shared reliability metrics + data loaders for the VoxIntel-R hypothesis suite
(notebooks 19-22). Reference-free: everything runs off the cached feature CSVs,
no audio / GPU / re-decode.

Metrics follow the definitions used in the selective-prediction and ASR-
confidence literature so numbers are comparable to prior work:
  ece / adaptive_ece    - calibration error (fixed-width and equal-mass bins)
  fpr_at_tpr            - FPR at a target TPR (failure = positive) [OOD standard]
  risk_coverage / aurc / e_aurc - selective prediction (risk-coverage curve)
  nce                   - normalized cross entropy of a probabilistic estimate
  paired_bootstrap_gap  - paired bootstrap CI on a metric gap between two scores

All (y, p) functions take y in {0,1} and p = predicted P(y=1); for a risk model
y = intent_failed and p = risk, so the same code covers calibration of the
failure probability and of a correctness confidence (pass 1-risk / 1-fail).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
SLURP_FEATURES = ROOT / "reports" / "phase6_voxintel_r_slurp" / "voxintel_r_features.csv"
FSC_FEATURES = ROOT / "reports" / "phase7_reliability_cross_dataset" / "fsc_voxintel_r_features.csv"
OUT_DIR = ROOT / "reports" / "phase8_hypothesis_validation"

ASR_FEATURES = ["asr_mean_confidence", "asr_min_confidence", "asr_std_confidence",
                "asr_median_confidence", "asr_mean_entropy", "asr_max_entropy",
                "asr_std_entropy", "asr_num_frames", "audio_duration"]
INTENT_FEATURES = ["intent_confidence", "intent_entropy", "intent_margin"]
FAMILIES = {"A_asr": ASR_FEATURES, "B_intent": INTENT_FEATURES,
            "C_combined": ASR_FEATURES + INTENT_FEATURES}
TARGET = "intent_failed"
SEED = 42
TEST_SIZE = 0.30  # same frozen split as src/analysis/frozen_split_eval.py


def load_slurp() -> pd.DataFrame:
    return pd.read_csv(SLURP_FEATURES)


def load_fsc() -> pd.DataFrame:
    return pd.read_csv(FSC_FEATURES)


def frozen_split(df: pd.DataFrame):
    """Stratified 70/30 frozen split (seed 42) — identical to the phase-6 script."""
    y = df[TARGET].astype(int)
    return train_test_split(df, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)


# --- calibration -----------------------------------------------------------
def ece(y, p, n_bins: int = 15) -> float:
    """Expected calibration error, fixed-width bins over [0,1]."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def adaptive_ece(y, p, n_bins: int = 15) -> float:
    """ECE with equal-mass bins (robust to the fixed-bin artefact)."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    order = np.argsort(p)
    e = 0.0
    for chunk in np.array_split(order, n_bins):
        if len(chunk):
            e += len(chunk) / len(p) * abs(y[chunk].mean() - p[chunk].mean())
    return float(e)


# --- OOD-style operating point --------------------------------------------
def fpr_at_tpr(y_pos, score, tpr: float = 0.95) -> float:
    """FPR at the threshold achieving >= `tpr` recall of the positive (failure)
    class. Standard misclassification/OOD-detection metric."""
    y_pos = np.asarray(y_pos, int); score = np.asarray(score, float)
    pos, neg = score[y_pos == 1], score[y_pos == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    thr = np.quantile(pos, 1 - tpr)  # >= tpr of positives score above thr
    return float((neg >= thr).mean())


# --- selective prediction --------------------------------------------------
def risk_coverage(y_fail, risk):
    """Accept the lowest-risk samples first. Returns (coverage, selective_risk)
    arrays where selective_risk[k] = failure rate among the accepted fraction."""
    y_fail = np.asarray(y_fail, float); risk = np.asarray(risk, float)
    order = np.argsort(risk, kind="mergesort")  # low risk accepted first
    accepted = y_fail[order]
    n = len(accepted)
    cov = np.arange(1, n + 1) / n
    sel_risk = np.cumsum(accepted) / np.arange(1, n + 1)
    return cov, sel_risk


def aurc(y_fail, risk) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    _, sr = risk_coverage(y_fail, risk)
    return float(sr.mean())


def e_aurc(y_fail, risk) -> float:
    """Excess AURC = AURC - AURC of the oracle ranking (backbone-agnostic)."""
    return float(aurc(y_fail, risk) - aurc(y_fail, np.asarray(y_fail, float)))


# --- normalized cross entropy ---------------------------------------------
def nce(y, p, eps: float = 1e-12) -> float:
    """Normalized cross entropy: fraction of the base-rate uncertainty about y
    that the estimate p removes. 1 = perfect, 0 = no better than the base rate,
    <0 = worse. (Standard ASR-confidence metric.)"""
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), eps, 1 - eps)
    q = y.mean()
    if q in (0.0, 1.0):
        return float("nan")
    h_base = -(q * np.log(q) + (1 - q) * np.log(1 - q))
    h_model = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float((h_base - h_model) / h_base)


def paired_bootstrap_gap(y, s1, s2, scorer, n: int = 2000, seed: int = SEED):
    """Paired bootstrap CI on scorer(y, s1) - scorer(y, s2) (same resampled
    indices for both). scorer(y, s) -> float, e.g. roc_auc_score."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); s1 = np.asarray(s1); s2 = np.asarray(s2)
    nn = len(y); gaps = []
    for _ in range(n):
        idx = rng.integers(0, nn, nn)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        gaps.append(scorer(yb, s1[idx]) - scorer(yb, s2[idx]))
    gaps = np.asarray(gaps)
    return {"mean_gap": float(gaps.mean()), "ci95_low": float(np.percentile(gaps, 2.5)),
            "ci95_high": float(np.percentile(gaps, 97.5)),
            "frac_positive": float((gaps > 0).mean()), "n_boot": len(gaps)}


def demo():
    """Self-check on synthetic data: perfect vs random estimators bracket every
    metric the way their definitions require."""
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.25).astype(int)          # 25% failures
    perfect = y.astype(float)                            # risk == label
    random_p = np.full(len(y), y.mean())                # constant base rate
    # calibration: a constant base-rate predictor is calibrated overall (all mass
    # in one fixed bin); equal-mass bins add tie-ordering noise, so allow more.
    assert ece(y, random_p) < 0.02
    assert adaptive_ece(y, random_p) < 0.05
    # a perfect risk ranking has ~zero excess AURC; random has more
    assert e_aurc(y, perfect) < 1e-9
    assert e_aurc(y, random_p) > e_aurc(y, perfect)
    # NCE: perfect ~1, base-rate ~0
    assert nce(y, np.clip(perfect, 1e-6, 1 - 1e-6)) > 0.99
    assert abs(nce(y, random_p)) < 1e-6
    # FPR@95TPR: perfect separation -> 0
    assert fpr_at_tpr(y, perfect, 0.95) < 1e-9
    # paired bootstrap: perfect beats random on AUC, CI strictly positive
    from sklearn.metrics import roc_auc_score
    g = paired_bootstrap_gap(y, perfect + rng.normal(0, .01, len(y)), random_p, roc_auc_score, n=300)
    assert g["ci95_low"] > 0
    print("reliability_metrics demo OK")


if __name__ == "__main__":
    demo()
