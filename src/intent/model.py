"""Intent-classification helpers for VoxIntel.

These helpers are meant to be used from notebooks that move beyond the
research-only analysis notebooks into reusable training and inference code.
The functions intentionally stay lightweight: they are compatible with the
SLURP-style transcripts and the DistilBERT baseline already established in the
project notebooks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd

from src.utils.text import clean_transcript

try:
    import torch
except Exception:  # pragma: no cover - optional for static importers.
    torch = None

try:
    from transformers import AutoTokenizer, pipeline
except Exception:  # pragma: no cover - optional for static importers.
    AutoTokenizer = None
    pipeline = None


def prepare_intent_dataframe(
    df: pd.DataFrame,
    text_col: str = "sentence",
    label_col: str = "intent",
    deduplicate: bool = True,
    min_support: int | None = None,
) -> pd.DataFrame:
    """Normalize the SLURP rows into a clean intent-classification table."""
    if df.empty:
        return df.copy()

    work = df[[text_col, label_col]].copy()
    work.columns = ["text", "intent"]
    work["text"] = work["text"].fillna("").astype(str).map(clean_transcript)
    work["intent"] = work["intent"].fillna("unknown").astype(str)

    if deduplicate:
        work = work.drop_duplicates(subset=["text", "intent"], keep="first").reset_index(drop=True)

    if min_support is not None:
        counts = work["intent"].value_counts()
        keep = counts[counts >= min_support].index
        work = work[work["intent"].isin(keep)].reset_index(drop=True)

    return work


def build_label_mapping(labels: Iterable[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create deterministic label IDs for the intent classes."""
    unique = sorted(dict.fromkeys(str(label) for label in labels if label is not None and str(label).strip()))
    label2id = {label: idx for idx, label in enumerate(unique)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def save_label_mapping(label2id: Mapping[str, int], path: str | Path) -> None:
    """Persist a label mapping to disk as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(label2id, handle, indent=2, sort_keys=True)


def load_label_mapping(path: str | Path) -> Dict[str, int]:
    """Load a saved label mapping from JSON."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_tokenizer(model_name: str = "distilbert-base-uncased"):
    """Create the tokenizer used for DistilBERT intent training."""
    if AutoTokenizer is None:
        raise ImportError("transformers is required to build the DistilBERT tokenizer.")
    return AutoTokenizer.from_pretrained(model_name)


def build_intent_pipeline(
    model_name: str = "distilbert-base-uncased",
    labels: Sequence[str] | None = None,
    device: int | str | None = None,
):
    """Build a Hugging Face text-classification pipeline for inference.

    The returned pipeline expects a list of transcript strings and uses the
    class mapping supplied in `labels`, if provided.
    """
    if pipeline is None:
        raise ImportError("transformers is required to build an intent pipeline.")

    # If no labels are supplied, pipeline will infer from the model config.
    if labels is None:
        return pipeline("text-classification", model=model_name, device=device)

    model = model_name
    return pipeline("text-classification", model=model, tokenizer=model_name, device=device, top_k=None)


def evaluate_prediction_quality(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, float]:
    """Lightweight accuracy and macro-F1 summary used in notebook prototyping."""
    from sklearn.metrics import accuracy_score, f1_score

    y_true = [str(v) for v in y_true]
    y_pred = [str(v) for v in y_pred]
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro")),
    }


__all__ = [
    "prepare_intent_dataframe",
    "build_label_mapping",
    "save_label_mapping",
    "load_label_mapping",
    "build_tokenizer",
    "build_intent_pipeline",
    "evaluate_prediction_quality",
]
