"""End-to-end + unit tests for the VoxIntel Command Arena.

Covers: scoring correctness, recommended-action thresholds, bad-input handling,
a successful round, an uncertain/deferred round, invalid inputs, and the
model/arena-unavailable path. Runs in the base env (fastapi + httpx + sklearn),
no torch. Model-dependent e2e self-skips if the risk model / features are absent.

Run: python tests/test_arena.py   (or under pytest)
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.serving.arena import ArenaStore, recommend, SCORE, ACTIONS, _NEEDED

FEATURES = ROOT / "reports" / "phase6_voxintel_r_slurp" / "voxintel_r_features.csv"


def _one_row_csv(tmp: Path, would_fail: int, confidence: float = 0.9) -> Path:
    """A single-row feature table so new_round() is deterministic for scoring tests."""
    p = tmp / "f.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(_NEEDED)
        w.writerow(["play some music", "play_music", "play_music",
                    confidence, 0.4, 0.6, would_fail])
    return p


def test_recommend_thresholds():
    assert recommend(0.10) == "execute"
    assert recommend(0.50) == "clarify"
    assert recommend(0.90) == "abstain"


def test_score_table_is_applied():
    for would_fail in (0, 1):
        with tempfile.TemporaryDirectory() as d:
            store = ArenaStore(_one_row_csv(Path(d), would_fail), risk_model=None)
            for action in ACTIONS:
                r = store.new_round()
                assert "would_fail" not in r and "ground_truth_intent" not in r  # no leak
                out = store.answer(r["round_id"], action)
                assert out["points"] == SCORE[(action, bool(would_fail))]
                assert out["would_fail"] == bool(would_fail)


def test_bad_inputs():
    with tempfile.TemporaryDirectory() as d:
        store = ArenaStore(_one_row_csv(Path(d), 0), risk_model=None)
        r = store.new_round()
        try:
            store.answer(r["round_id"], "launch_rockets")
            raise AssertionError("invalid action should raise")
        except ValueError:
            pass
        try:
            store.answer("nope", "execute")
            raise AssertionError("unknown round should raise")
        except KeyError:
            pass


def test_missing_columns_raises():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bad.csv"
        p.write_text("a,b\n1,2\n", encoding="utf-8")
        try:
            ArenaStore(p, risk_model=None)
            raise AssertionError("missing columns should raise")
        except ValueError:
            pass


def test_metrics_shape():
    with tempfile.TemporaryDirectory() as d:
        store = ArenaStore(_one_row_csv(Path(d), 1), risk_model=None)
        store.answer(store.new_round()["round_id"], "abstain")
        m = store.metrics()
        assert m["rounds"] == 1 and m["score"] == SCORE[("abstain", True)]
        assert 0.0 <= m["coverage"] <= 1.0


def _e2e():
    """Full FastAPI stack via TestClient against the real artifacts."""
    try:
        from fastapi.testclient import TestClient
    except Exception as e:
        print(f"  (skipped e2e — fastapi/httpx absent: {e})"); return
    if not FEATURES.exists():
        print("  (skipped e2e — features csv absent)"); return
    from src.serving.app import create_app
    c = TestClient(create_app())

    h = c.get("/health").json()
    assert h["status"] == "ok" and h["arena"] == "ready", h

    rnd = c.get("/arena/new").json()
    assert {"round_id", "transcript", "risk", "recommended_action"} <= set(rnd)
    assert "would_fail" not in rnd  # truth must not leak to the client

    ans = c.post("/arena/answer", json={"round_id": rnd["round_id"], "action": "execute"}).json()
    assert ans["points"] in set(SCORE.values()) and "running_score" in ans

    bad = c.post("/arena/answer", json={"round_id": rnd["round_id"], "action": "zzz"}).json()
    assert "error" in bad  # already-answered OR invalid action

    st = c.post("/arena/score_text", json={"text": "turn off the lights"}).json()
    assert 0.0 <= st["risk"] <= 1.0 and st["mode"] in ("demo", "live")
    if st["mode"] == "demo":
        assert st["simulated_intent"] is True

    empty = c.post("/arena/score_text", json={"text": "   "}).json()
    assert "error" in empty

    assert c.get("/arena/metrics").json()["rounds"] >= 1
    print("  e2e OK (health, new, answer, invalid, score_text demo/live, metrics)")


if __name__ == "__main__":
    tests = [test_recommend_thresholds, test_score_table_is_applied, test_bad_inputs,
             test_missing_columns_raises, test_metrics_shape, _e2e]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print("all arena tests passed")
