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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from fastapi import FastAPI
except Exception:  # pragma: no cover - fastapi optional at import time
    FastAPI = None

DEFAULT_RISK_MODEL = Path(__file__).resolve().parents[2] / "models" / "voxintel_r_intent_rf.joblib"


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

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok", "service": "VoxIntel",
                "reliability_model": "loaded" if risk_model else "unavailable"}

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
