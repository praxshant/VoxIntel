"""FastAPI app for the VoxIntel speech-to-intent layer + VoxIntel-R reliability.

Two concerns live here:
  * ``/predict``      — intent classification over a transcript (pluggable model).
  * ``/reliability``  — reference-free VoxIntel-R risk score for a prediction:
                        given the intent model's own probability vector, how
                        likely is this prediction to be *wrong*, and should the
                        assistant execute it or defer/ask for confirmation?

The reliability model is the intent-confidence RandomForest validated on a
frozen SLURP split (see ``src/analysis/frozen_split_eval.py`` and
``reports/phase6_voxintel_r_slurp/FROZEN_SPLIT_RESULTS.md``). It uses only
inference-time signals — no reference transcript — so it is deployable.

SECURITY: these endpoints have NO authentication or rate limiting. That is fine
for local evaluation only. Put an auth layer / gateway in front before exposing
this beyond localhost.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from fastapi import FastAPI
except Exception:  # pragma: no cover - fastapi optional at import time
    FastAPI = None

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RISK_MODEL = _ROOT / "models" / "voxintel_r_intent_rf.joblib"
DEFAULT_FEATURES = _ROOT / "reports" / "phase6_voxintel_r_slurp" / "voxintel_r_features.csv"
STATIC_DIR = Path(__file__).resolve().parent / "static"


def load_risk_model(path: Path = DEFAULT_RISK_MODEL) -> Dict[str, Any]:
    """Load the persisted VoxIntel-R bundle {model, features, threshold, ...}."""
    from joblib import load
    if not Path(path).exists():
        raise FileNotFoundError(
            f"risk model not found: {path}\nRun: python src/analysis/frozen_split_eval.py")
    return load(path)


def intent_features_from_probs(probs: Sequence[float]) -> Dict[str, float]:
    """Derive the 3 intent-native features from a softmax vector.

    Mirrors notebook 16's ``extract_intent_features`` exactly so the served
    features match the ones the model was trained on.
    """
    p = [float(x) for x in probs]
    if not p:
        raise ValueError("empty probability vector")
    ordered = sorted(p, reverse=True)
    top1 = ordered[0]
    top2 = ordered[1] if len(ordered) > 1 else 0.0
    eps = 1e-10
    entropy = -sum(x * math.log(x + eps) for x in p)
    return {"intent_confidence": top1, "intent_entropy": entropy, "intent_margin": top1 - top2}


def score_reliability(risk_model: Dict[str, Any],
                      intent_probs: Optional[Sequence[float]] = None,
                      features: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Return {risk, decision, features} for one prediction. Pure (no FastAPI)."""
    if features is None:
        if intent_probs is None:
            raise ValueError("provide either intent_probs or features")
        features = intent_features_from_probs(intent_probs)
    order = risk_model["features"]
    row = [[features[f] for f in order]]
    risk = float(risk_model["model"].predict_proba(row)[0, 1])
    threshold = float(risk_model.get("threshold", 0.5))
    decision = "defer" if risk >= threshold else "execute"
    return {"risk": risk, "decision": decision, "threshold": threshold, "features": features}


def _demo_intent_probs(text: str) -> List[float]:
    """Deterministic SIMULATED 3-class probability vector seeded by the text.
    Used only in demo mode (no intent model loaded). The result is clearly
    flagged simulated by callers — it is NOT a real model prediction."""
    import hashlib
    h = int(hashlib.sha256((text or "").encode("utf-8")).hexdigest(), 16)
    a = 0.5 + (h % 1000) / 2000.0           # top prob in [0.5, 1.0)
    b = (1 - a) * ((h >> 10) % 1000) / 1000.0
    return [a, b, max(0.0, 1 - a - b)]


def create_app(
    model: Any = None,
    tokenizer: Any = None,
    label_map: Optional[Dict[str, int]] = None,
    prediction_fn: Optional[Any] = None,
    risk_model: Optional[Dict[str, Any]] = None,
    load_risk: bool = True,
) -> Any:
    """Create the FastAPI app. Intent inference stays pluggable; the reliability
    route is wired to the persisted VoxIntel-R model when available."""
    if FastAPI is None:
        raise ImportError("fastapi is required to create the serving app.")

    app = FastAPI(title="VoxIntel Speech-to-Intent API")

    if risk_model is None and load_risk:
        try:
            risk_model = load_risk_model()
        except FileNotFoundError:
            risk_model = None  # /reliability will report it's unavailable

    # --- VoxIntel Command Arena (reference-free reliability game) ---
    # Loads the cached SLURP feature table; real rounds, real labels, real risk.
    arena = None
    arena_error = None
    try:
        from src.serving.arena import ArenaStore
        feats = Path(os.environ.get("VOXINTEL_FEATURES", DEFAULT_FEATURES))
        low = float(os.environ.get("VOXINTEL_RISK_LOW", 0.33))
        high = float(os.environ.get("VOXINTEL_RISK_HIGH", 0.66))
        arena = ArenaStore(feats, risk_model=risk_model, low=low, high=high)
    except Exception as e:  # missing csv / bad columns -> arena disabled, app still serves
        arena_error = str(e)

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "service": "VoxIntel",
                "reliability_model": "loaded" if risk_model else "unavailable",
                "intent_inference": "live" if (prediction_fn or model) else "demo",
                "arena": "ready" if arena else f"unavailable: {arena_error}"}

    @app.post("/predict")
    def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
        text = payload.get("text") or payload.get("transcript") or ""
        if not isinstance(text, str) or not text.strip():
            return {"error": "A non-empty transcript is required.", "prediction": None}
        if prediction_fn is not None:
            return prediction_fn(text)
        if model is None or tokenizer is None:
            return {"error": "Model and tokenizer must be attached before inference.", "prediction": None}
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        with __import__("torch").no_grad():
            logits = model(**encoded).logits
        predicted_id = int(logits.argmax(dim=-1).item())
        label = next((k for k, v in (label_map or {}).items() if v == predicted_id), str(predicted_id))
        return {"prediction": label, "confidence": float(logits.softmax(dim=-1).max().item())}

    @app.post("/reliability")
    def reliability(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Reference-free risk for a prediction. Body: {"intent_probs": [...]} or
        {"features": {intent_confidence, intent_entropy, intent_margin}}."""
        if risk_model is None:
            return {"error": "reliability model unavailable — run "
                             "src/analysis/frozen_split_eval.py", "risk": None}
        try:
            return score_reliability(risk_model,
                                     intent_probs=payload.get("intent_probs"),
                                     features=payload.get("features"))
        except (ValueError, KeyError) as e:
            return {"error": str(e), "risk": None}

    def _require_arena():
        if arena is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=503, detail=f"arena unavailable: {arena_error}")
        return arena

    @app.get("/arena/new")
    def arena_new() -> Dict[str, Any]:
        return _require_arena().new_round()

    @app.post("/arena/answer")
    def arena_answer(payload: Dict[str, Any]) -> Dict[str, Any]:
        a = _require_arena()
        try:
            return a.answer(payload.get("round_id", ""), payload.get("action", ""))
        except (ValueError, KeyError) as e:
            return {"error": str(e)}

    @app.get("/arena/metrics")
    def arena_metrics() -> Dict[str, Any]:
        return _require_arena().metrics()

    @app.get("/arena/history")
    def arena_history(limit: int = 50) -> Dict[str, Any]:
        return {"history": _require_arena().history(limit=limit)}

    @app.post("/arena/score_text")
    def arena_score_text(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Score a free-typed command. Real intent inference when a model is
        attached; otherwise a clearly-flagged demo (simulated intent probs,
        real risk model on those simulated inputs)."""
        text = (payload.get("text") or "").strip()
        if not text:
            return {"error": "A non-empty command is required."}
        mode = "demo"
        probs = _demo_intent_probs(text)
        predicted = None
        if prediction_fn is not None or (model is not None and tokenizer is not None):
            try:
                pred = predict({"text": text})
                if "error" not in pred:
                    mode, predicted = "live", pred.get("prediction")
                    if "probs" in pred:
                        probs = pred["probs"]
                    elif "confidence" in pred:
                        c = float(pred["confidence"]); probs = [c, (1 - c) * 0.6, (1 - c) * 0.4]
            except Exception:
                mode = "demo"
        if risk_model is None:
            return {"mode": mode, "simulated_intent": mode == "demo",
                    "error": "reliability model unavailable"}
        scored = score_reliability(risk_model, intent_probs=probs)
        from src.serving.arena import recommend
        return {"mode": mode, "simulated_intent": mode == "demo",
                "note": "intent is SIMULATED — not a real model prediction" if mode == "demo"
                        else "live intent model",
                "text": text, "predicted_intent": predicted,
                "risk": round(scored["risk"], 4),
                "recommended_action": recommend(scored["risk"], arena.low if arena else 0.33,
                                                arena.high if arena else 0.66)}

    if STATIC_DIR.exists():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    return app


def demo():
    """Self-check: score a confident and an uncertain prediction (no FastAPI needed)."""
    rm = load_risk_model()
    confident = score_reliability(rm, intent_probs=[0.97, 0.02, 0.01])
    uncertain = score_reliability(rm, intent_probs=[0.40, 0.35, 0.25])
    assert 0.0 <= confident["risk"] <= 1.0 and 0.0 <= uncertain["risk"] <= 1.0
    assert uncertain["risk"] > confident["risk"], "uncertain prediction should be riskier"
    assert confident["decision"] == "execute"
    print(f"confident -> risk {confident['risk']:.3f} ({confident['decision']}); "
          f"uncertain -> risk {uncertain['risk']:.3f} ({uncertain['decision']})")
    print("demo OK")


__all__ = ["create_app", "load_risk_model", "score_reliability", "intent_features_from_probs"]

if __name__ == "__main__":
    demo()
