"""
WER/CER scoring helpers.

Promoted from notebooks/04_asr_evaluation.ipynb. Slightly earlier promotion
than the project's usual "3 notebooks unchanged" bar — this is thin,
non-experimental wrapper logic around jiwer (not a modeling choice under
active iteration), and notebook 06 (fine-tuned ASR evaluation) needs to
run the *exact* same grouping logic against the *exact* same validation
set for the baseline-vs-fine-tuned comparison to be meaningful. Duplicating
it by copy-paste into 06 risks a silent drift (e.g. a different min_count
default) that would quietly invalidate that comparison.
"""

import jiwer
import numpy as np
import pandas as pd

from src.utils.text import normalize_text


def add_wer_cer_columns(df: pd.DataFrame, ground_truth_col: str = "ground_truth",
                         prediction_col: str = "prediction") -> pd.DataFrame:
    """Add normalized text + per-sample WER/CER columns to a predictions dataframe.

    Expects a dataframe with a ground-truth transcript column and a
    prediction column (rows where prediction is null/NaN are dropped first).
    Returns a new dataframe; does not mutate the input.
    """
    scored = df.dropna(subset=[prediction_col]).copy()

    scored["ground_truth_norm"] = scored[ground_truth_col].apply(normalize_text)
    scored["prediction_norm"] = scored[prediction_col].apply(normalize_text)

    scored["wer"] = [
        _safe_wer(r, h) for r, h in zip(scored["ground_truth_norm"], scored["prediction_norm"])
    ]
    scored["cer"] = [
        _safe_cer(r, h) for r, h in zip(scored["ground_truth_norm"], scored["prediction_norm"])
    ]
    return scored


def corpus_wer_cer(df: pd.DataFrame) -> dict:
    """Corpus-level WER/CER over an already-scored dataframe (see add_wer_cer_columns).

    Corpus-level = total edit distance over total reference words/chars,
    NOT the mean of the per-sample wer/cer columns (those are a different,
    also-useful statistic — see notebook 04 Cell 7 for why both are reported).
    """
    refs = list(df["ground_truth_norm"])
    hyps = list(df["prediction_norm"])
    return {
        "wer": jiwer.wer(refs, hyps),
        "cer": jiwer.cer(refs, hyps),
    }


def corpus_wer_per_group(df: pd.DataFrame, group_col: str, min_count: int = 5) -> pd.DataFrame:
    """Corpus-level WER/CER computed within each group (e.g. per-intent, per-scenario).

    Groups with fewer than `min_count` samples are dropped — WER on 1-2
    examples isn't a meaningful per-class estimate. Returns a dataframe
    sorted worst (highest WER) first.

    Requires df to already have ground_truth_norm/prediction_norm columns
    (i.e. run add_wer_cer_columns first).
    """
    rows = []
    for key, group in df.groupby(group_col, observed=True):
        if len(group) < min_count:
            continue
        wer = jiwer.wer(list(group["ground_truth_norm"]), list(group["prediction_norm"]))
        cer = jiwer.cer(list(group["ground_truth_norm"]), list(group["prediction_norm"]))
        rows.append({group_col: key, "wer": wer, "cer": cer, "count": len(group)})

    return pd.DataFrame(rows).sort_values("wer", ascending=False).reset_index(drop=True)


def _safe_wer(ref: str, hyp: str) -> float:
    try:
        return jiwer.wer(ref, hyp)
    except Exception:
        return float("nan")


def _safe_cer(ref: str, hyp: str) -> float:
    try:
        return jiwer.cer(ref, hyp)
    except Exception:
        return float("nan")
