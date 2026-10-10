"""VoxIntel Command Arena — the reliability game logic (pure, testable).

The arena turns the VoxIntel-R reliability model into a usable product: each round
shows a real SLURP voice command (its ASR transcript), the intent model's predicted
intent, and the calibrated failure risk; the player (or the model's own policy)
chooses EXECUTE / CLARIFY / ABSTAIN, and the score rewards acting on commands the
system got right and deferring the ones it got wrong.

Everything here is real and CPU-only: rounds are drawn from the cached SLURP
feature table (real transcripts, real intent-model outputs, real `intent_failed`
labels) and risk is the deployed RandomForest's `predict_proba`. No audio, no GPU,
no fabricated outcomes — the hidden truth each round is scored against is the
dataset's actual label.

Scoring (points reflect measurable reliability outcomes, not arbitrary gamification):
  EXECUTE  a command that would succeed  -> +10   (good automation)
  EXECUTE  a command that would fail     -> -15   (the costly unsafe action)
  ABSTAIN  on a command that would fail  ->  +8   (caught a failure)
  ABSTAIN  on a command that would succeed -> -5  (lost a good command)
  CLARIFY  on a would-fail command       ->  +3   (safe, cheap fallback)
  CLARIFY  on a would-succeed command    ->  -2   (unnecessary friction)
"""
from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

ACTIONS = ("execute", "clarify", "abstain")
SCORE = {("execute", False): 10, ("execute", True): -15,
         ("abstain", True): 8, ("abstain", False): -5,
         ("clarify", True): 3, ("clarify", False): -2}
# default risk thresholds -> recommended action (overridable via env)
DEFAULT_LOW, DEFAULT_HIGH = 0.33, 0.66
_NEEDED = ["asr_transcript", "ground_truth_intent", "predicted_intent",
           "intent_confidence", "intent_entropy", "intent_margin", "intent_failed"]
_RISK_FEATURES = ["intent_confidence", "intent_entropy", "intent_margin"]


def recommend(risk: float, low: float = DEFAULT_LOW, high: float = DEFAULT_HIGH) -> str:
    """Map a calibrated risk to the system's recommended action."""
    return "execute" if risk < low else ("clarify" if risk < high else "abstain")

class ArenaStore:
    """Thread-safe in-memory arena backed by the cached SLURP feature table.

    ponytail: in-memory history (one process). Swap for SQLite only if you need
    cross-process persistence — a game session does not.
    """

    def __init__(self, features_csv: Path, risk_model: Optional[Dict[str, Any]] = None,
                 low: float = DEFAULT_LOW, high: float = DEFAULT_HIGH,
                 seed: int = 42, max_history: int = 500):
        import pandas as pd
        df = pd.read_csv(features_csv)
        missing = [c for c in _NEEDED if c not in df.columns]
        if missing:
            raise ValueError(f"features csv missing columns: {missing}")
        self._df = df[_NEEDED].reset_index(drop=True)
        self._risk_model = risk_model
        self.low, self.high = float(low), float(high)
        self.risk_source = "model" if risk_model else "proxy(1-confidence)"
        self._rng = __import__("random").Random(seed)
        self._rounds: Dict[str, Dict[str, Any]] = {}
        self._history: List[Dict[str, Any]] = []
        self._max_history = max_history
        self._lock = threading.Lock()

    def _risk(self, row) -> float:
        if self._risk_model is not None:
            x = [[float(row[f]) for f in _RISK_FEATURES]]
            return float(self._risk_model["model"].predict_proba(x)[0, 1])
        return float(1.0 - row["intent_confidence"])  # proxy when model absent

    def new_round(self) -> Dict[str, Any]:
        with self._lock:
            i = self._rng.randrange(len(self._df))
            row = self._df.iloc[i]
            risk = self._risk(row)
            rid = uuid.uuid4().hex[:12]
            self._rounds[rid] = {"risk": risk,
                                 "would_fail": bool(int(row["intent_failed"])),
                                 "ground_truth_intent": str(row["ground_truth_intent"]),
                                 "predicted_intent": str(row["predicted_intent"])}
            return {"round_id": rid, "transcript": str(row["asr_transcript"]),
                    "predicted_intent": str(row["predicted_intent"]),
                    "risk": round(risk, 4), "risk_source": self.risk_source,
                    "recommended_action": recommend(risk, self.low, self.high)}

    def answer(self, round_id: str, action: str) -> Dict[str, Any]:
        action = (action or "").strip().lower()
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}")
        with self._lock:
            r = self._rounds.pop(round_id, None)
            if r is None:
                raise KeyError("unknown or already-answered round_id")
            points = SCORE[(action, r["would_fail"])]
            rec = {"round_id": round_id, "action": action, "points": points,
                   "would_fail": r["would_fail"], "risk": round(r["risk"], 4),
                   "recommended_action": recommend(r["risk"], self.low, self.high),
                   "predicted_intent": r["predicted_intent"],
                   "ground_truth_intent": r["ground_truth_intent"],
                   "followed_recommendation": action == recommend(r["risk"], self.low, self.high)}
            self._history.append(rec)
            del self._history[:-self._max_history]
            rec["running_score"] = sum(h["points"] for h in self._history)
            return rec

    def history(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return list(reversed(self._history[-limit:]))

    def metrics(self) -> Dict[str, Any]:
        with self._lock:
            h = self._history
            n = len(h)
            if not n:
                return {"rounds": 0, "score": 0}
            execs = [x for x in h if x["action"] == "execute"]
            fails = [x for x in h if x["would_fail"]]
            caught = [x for x in fails if x["action"] in ("abstain", "clarify")]
            return {"rounds": n, "score": sum(x["points"] for x in h),
                    "coverage": round(len(execs) / n, 3),
                    "execute_accuracy": round(sum(not x["would_fail"] for x in execs)
                                              / len(execs), 3) if execs else None,
                    "abstain_rate": round(sum(x["action"] == "abstain" for x in h) / n, 3),
                    "failure_catch_rate": round(len(caught) / len(fails), 3) if fails else None,
                    "followed_recommendation_rate":
                        round(sum(x["followed_recommendation"] for x in h) / n, 3),
                    "risk_source": self.risk_source}

