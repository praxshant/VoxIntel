"""VoxIntel Reliability Dashboard — real held-out evaluation of the risk model.

No game, no points. This reproduces the paper's honest frozen-split protocol at
startup (stratified 70/30, seed 42: fit the intent-only RandomForest on train,
score the untouched test split) and exposes the numbers a reviewer cares about:
ROC-AUC, Brier, ECE, the risk-coverage curve and AURC for the model vs the MSP
(max-softmax) baseline vs the oracle, and coverage at fixed risk budgets. It also
scores a single typed command on the same model (the inspector).

All real and CPU-only: cached SLURP features + a RandomForest refit here. Numbers
match reports/phase6_voxintel_r_slurp/FROZEN_SPLIT_RESULTS.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

_RISK_FEATURES = ["intent_confidence", "intent_entropy", "intent_margin"]
DEFAULT_LOW, DEFAULT_HIGH = 0.33, 0.66


def recommend(risk: float, low: float = DEFAULT_LOW, high: float = DEFAULT_HIGH) -> str:
    return "execute" if risk < low else ("clarify" if risk < high else "abstain")


def _downsample(cov, risk, n: int = 120) -> List[List[float]]:
    import numpy as np
    idx = np.unique(np.linspace(0, len(cov) - 1, min(n, len(cov))).astype(int))
    return [[round(float(cov[i]), 4), round(float(risk[i]), 4)] for i in idx]


class EvalReport:
    """Fit-once held-out evaluation; serves cached metrics, curves, and an inspector."""

    def __init__(self, features_csv: Path, low: float = DEFAULT_LOW, high: float = DEFAULT_HIGH):
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
        from src.analysis.reliability_metrics import (frozen_split, ece, aurc, e_aurc,
                                                      risk_coverage, TARGET, SEED)
        df = pd.read_csv(features_csv)
        missing = [c for c in _RISK_FEATURES + [TARGET] if c not in df.columns]
        if missing:
            raise ValueError(f"features csv missing columns: {missing}")
        self.low, self.high = float(low), float(high)

        Xtr, Xte, ytr, yte = frozen_split(df)
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                    random_state=SEED, n_jobs=-1)
        rf.fit(Xtr[_RISK_FEATURES].values, ytr)          # bare array -> no feature-name warning
        self._rf = rf
        y = yte.to_numpy().astype(int)
        p = rf.predict_proba(Xte[_RISK_FEATURES].values)[:, 1]
        msp = 1.0 - Xte["intent_confidence"].to_numpy()      # the field's weak baseline
        rng = np.random.default_rng(SEED)

        self._summary = {
            "n_test": int(len(y)), "failure_rate": round(float(y.mean()), 4),
            "roc_auc": round(float(roc_auc_score(y, p)), 4),
            "pr_auc": round(float(average_precision_score(y, p)), 4),
            "brier": round(float(brier_score_loss(y, p)), 4),
            "ece": round(float(ece(y, p)), 4),
            "aurc_model": round(float(aurc(y, p)), 4),
            "aurc_msp": round(float(aurc(y, msp)), 4),
            "aurc_oracle": round(float(aurc(y, y.astype(float))), 4),
            "e_aurc_model": round(float(e_aurc(y, p)), 4),
            "seed": SEED, "split": "stratified 70/30 held-out",
        }
        cov_m, sr_m = risk_coverage(y, p)
        cov_s, sr_s = risk_coverage(y, msp)
        cov_o, sr_o = risk_coverage(y, y.astype(float))
        cov_r, sr_r = risk_coverage(y, rng.random(len(y)))
        self._curves = {"model": _downsample(cov_m, sr_m), "msp": _downsample(cov_s, sr_s),
                        "oracle": _downsample(cov_o, sr_o), "random": _downsample(cov_r, sr_r)}
        self._budgets = []
        for b in (0.01, 0.05, 0.10):
            ok = cov_m[sr_m <= b]
            self._budgets.append({"risk_budget": b,
                                  "max_coverage": round(float(ok.max()), 4) if len(ok) else 0.0})

    def summary(self) -> Dict[str, Any]:
        return dict(self._summary)

    def curves(self) -> Dict[str, Any]:
        return {"curves": self._curves, "budgets": self._budgets}

    def score(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Risk + recommended action for one prediction's intent features."""
        row = [[float(features[f]) for f in _RISK_FEATURES]]
        risk = float(self._rf.predict_proba(row)[0, 1])
        return {"risk": risk, "recommended_action": recommend(risk, self.low, self.high),
                "threshold_low": self.low, "threshold_high": self.high}
