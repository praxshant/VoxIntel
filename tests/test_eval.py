"""Tests for the VoxIntel Reliability Dashboard (evaluation module + endpoints).

Covers: held-out metric sanity (model beats the MSP baseline, AUC reproduces the
paper), the risk score's monotonicity, dashboard endpoint shapes, the single-command
inspector (demo + error paths), and bad input. Runs in base (fastapi + httpx +
sklearn), no torch; the inspector is "live" only when an intent model is loaded.

Run: python tests/test_eval.py   (or under pytest)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.serving.evaluation import EvalReport, recommend
from src.serving.app import intent_features_from_probs

FEATURES = ROOT / "reports" / "phase6_voxintel_r_slurp" / "voxintel_r_features.csv"


def test_recommend_thresholds():
    assert recommend(0.10) == "execute"
    assert recommend(0.50) == "clarify"
    assert recommend(0.90) == "abstain"


def test_report_metrics_and_monotonicity():
    if not FEATURES.exists():
        print("  (skipped — features csv absent)"); return
    rep = EvalReport(FEATURES)
    s = rep.summary()
    # reproduces the frozen-split headline and beats the MSP baseline on AURC.
    assert s["roc_auc"] > 0.85, s
    assert s["aurc_model"] < s["aurc_msp"], s          # selective prediction works
    assert s["aurc_oracle"] <= s["aurc_model"]          # oracle is the floor
    assert 0.0 <= s["ece"] < 0.2
    c = rep.curves()
    assert set(c["curves"]) == {"model", "msp", "oracle", "random"}
    assert len(c["budgets"]) == 3
    # a confident prediction is lower-risk than an uncertain one.
    conf = rep.score(intent_features_from_probs([0.97, 0.02, 0.01]))
    unc = rep.score(intent_features_from_probs([0.40, 0.35, 0.25]))
    assert 0.0 <= conf["risk"] <= 1.0 and 0.0 <= unc["risk"] <= 1.0
    assert unc["risk"] > conf["risk"]
    assert conf["recommended_action"] == "execute"


def test_missing_columns_raises(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bad.csv"
        p.write_text("a,b\n1,2\n", encoding="utf-8")
        try:
            EvalReport(p)
            raise AssertionError("missing columns should raise")
        except ValueError:
            pass


def _e2e():
    try:
        from fastapi.testclient import TestClient
    except Exception as e:
        print(f"  (skipped e2e — fastapi/httpx absent: {e})"); return
    if not FEATURES.exists():
        print("  (skipped e2e — features csv absent)"); return
    from src.serving.app import create_app
    c = TestClient(create_app())

    h = c.get("/health").json()
    assert h["status"] == "ok" and h["dashboard"] == "ready", h

    s = c.get("/eval/summary").json()
    assert {"roc_auc", "aurc_model", "aurc_msp", "ece", "n_test"} <= set(s)
    assert s["aurc_model"] < s["aurc_msp"]

    rc = c.get("/eval/risk_coverage").json()
    assert "model" in rc["curves"] and len(rc["budgets"]) == 3

    ins = c.post("/inspect", json={"text": "send a message to mom"}).json()
    assert 0.0 <= ins["risk"] <= 1.0 and ins["mode"] in ("demo", "live")
    assert ins["recommended_action"] in ("execute", "clarify", "abstain")
    if ins["mode"] == "demo":
        assert ins["simulated_intent"] is True

    assert "error" in c.post("/inspect", json={"text": "   "}).json()
    print("  e2e OK (health, summary, risk_coverage, inspect demo/live, bad input)")


if __name__ == "__main__":
    tests = [test_recommend_thresholds, test_report_metrics_and_monotonicity,
             test_missing_columns_raises, _e2e]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print("all dashboard tests passed")
