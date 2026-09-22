"""FastAPI app scaffolding for the VoxIntel speech-to-intent inference layer."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

try:
    from fastapi import FastAPI
except Exception:  # pragma: no cover
    FastAPI = None


def create_app(
    model: Any = None,
    tokenizer: Any = None,
    label_map: Optional[Dict[str, int]] = None,
    prediction_fn: Optional[Any] = None,
) -> Any:
    """Create a minimal FastAPI application for intent inference.

    The app intentionally keeps the inference logic pluggable so future
    notebooks can attach either a Hugging Face pipeline or a custom wrapper.
    """
    if FastAPI is None:
        raise ImportError("fastapi is required to create the serving app.")

    app = FastAPI(title="VoxIntel Speech-to-Intent API")

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok", "service": "VoxIntel"}

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

    return app


__all__ = ["create_app"]
