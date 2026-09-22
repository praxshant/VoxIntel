"""Guards for the VoxIntel-R reliability path and the frozen-split invariants.

Dependency-free: run directly (`python tests/test_reliability.py`) or under
pytest. The model-dependent test self-skips when the persisted joblib bundle is
absent (it is gitignored), so this still runs on a fresh clone / in CI.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.serving.app import (DEFAULT_RISK_MODEL, intent_features_from_probs,
                             load_risk_model, score_reliability)
from src.analysis.frozen_split_eval import (ASR_FEATURES, FAMILIES,
                                            INTENT_FEATURES,
                                            _paired_bootstrap_auc_gap)


def test_intent_features_from_probs():
    # Hand-computed on [0.7, 0.2, 0.1]: margin = 0.7-0.2, entropy = -Σ p ln p.
    f = intent_features_from_probs([0.7, 0.2, 0.1])
    assert abs(f["intent_confidence"] - 0.7) < 1e-9
    assert abs(f["intent_margin"] - 0.5) < 1e-9
    expected_H = -(0.7 * math.log(0.7) + 0.2 * math.log(0.2) + 0.1 * math.log(0.1))
    assert abs(f["intent_entropy"] - expected_H) < 1e-4
    # Order must not matter (function sorts internally).
    assert intent_features_from_probs([0.1, 0.7, 0.2])["intent_margin"] == f["intent_margin"]
    # Empty vector is a caller error.
    try:
        intent_features_from_probs([])
        raise AssertionError("empty probs should raise ValueError")
    except ValueError:
        pass


def test_feature_family_invariants():
    assert len(FAMILIES["A_asr"]) == 9
    assert len(FAMILIES["B_intent"]) == 3
    assert len(FAMILIES["C_combined"]) == 12
    assert set(FAMILIES["C_combined"]) == set(ASR_FEATURES) | set(INTENT_FEATURES)
    # Leakage guard: no reference/ground-truth column may sit in any family.
    banned = {"ground_truth_intent", "asr_transcript", "reference", "intent_failed"}
    for feats in FAMILIES.values():
        assert not (set(feats) & banned)


def test_paired_bootstrap_gap_sign():
    # p_c separates y perfectly (AUC=1); p_b is noise (~0.5). Gap must be positive.
    y = np.array([0, 1] * 50)
    p_c = np.where(y == 1, 0.9, 0.1).astype(float)
    p_b = np.random.default_rng(0).random(len(y))
    out = _paired_bootstrap_auc_gap(y, p_c, p_b, n=200, seed=0)
    assert set(out) == {"mean_gap_C_minus_B", "ci95_low", "ci95_high",
                        "frac_C_beats_B", "n_boot"}
    assert out["n_boot"] > 0
    assert out["mean_gap_C_minus_B"] > 0.2
    assert out["frac_C_beats_B"] > 0.9


def test_score_reliability_monotonic():
    if not DEFAULT_RISK_MODEL.exists():
        print("  (skipped test_score_reliability_monotonic — model artifact absent)")
        return
    rm = load_risk_model()
    confident = score_reliability(rm, intent_probs=[0.97, 0.02, 0.01])
    uncertain = score_reliability(rm, intent_probs=[0.40, 0.35, 0.25])
    for r in (confident, uncertain):
        assert 0.0 <= r["risk"] <= 1.0
    assert uncertain["risk"] > confident["risk"], "uncertain must be riskier"
    assert confident["decision"] == "execute"
    # Passing features directly must match deriving them from probs.
    feats = intent_features_from_probs([0.97, 0.02, 0.01])
    assert abs(score_reliability(rm, features=feats)["risk"] - confident["risk"]) < 1e-9


if __name__ == "__main__":
    tests = [test_intent_features_from_probs, test_feature_family_invariants,
             test_paired_bootstrap_gap_sign, test_score_reliability_monotonic]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print("all tests passed")
