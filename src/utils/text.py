"""
Reusable text/transcript utilities for VoxIntel.

Used by the dataset-audit notebook, ASR evaluation (WER/CER need consistently
normalized text on both sides), and intent-model training/inference.
"""

import re
import string
import unicodedata

# Characters SLURP/ASR transcripts sometimes contain that we want to strip
# during normalization for WER/CER comparison and for the intent model's
# tokenizer input.
_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)
_MULTI_SPACE_RE = re.compile(r"\s+")

__all__ = [
    "normalize_text",
    "clean_transcript",
    "word_count",
    "remove_special_tokens",
    "is_empty_or_whitespace",
]


def normalize_text(text: str) -> str:
    """Normalize text for fair comparison (e.g. WER/CER computation between
    ground-truth and ASR-generated transcripts).

    Steps:
        1. Unicode-normalize (NFKC) so visually-identical characters compare equal.
        2. Lowercase.
        3. Strip punctuation.
        4. Collapse repeated whitespace and strip leading/trailing whitespace.

    This is intentionally aggressive — it's meant for *comparison*, not for
    producing user-facing text. Use `clean_transcript` if you want lighter,
    more preserving cleanup.
    """
    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = text.translate(_PUNCTUATION_TABLE)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def clean_transcript(text: str) -> str:
    """Light cleanup for a raw transcript before display or model input.

    Unlike `normalize_text`, this preserves punctuation and casing — it just
    fixes whitespace/encoding artifacts that occasionally show up in scraped
    or ASR-hypothesis transcripts.

    Steps:
        1. Unicode-normalize (NFKC).
        2. Strip leading/trailing whitespace.
        3. Collapse repeated internal whitespace.
        4. Remove stray whitespace before punctuation (e.g. "hello ." -> "hello.").
    """
    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text


def word_count(text: str) -> int:
    """Number of whitespace-separated words in a transcript."""
    if not text:
        return 0
    return len(text.strip().split())


def remove_special_tokens(text: str, tokens: list[str] | None = None) -> str:
    """Strip dataset-specific placeholder/special tokens from a transcript,
    e.g. entity markup like '[song_name : shape of you]' if present in some
    annotation variants. Pass a custom `tokens` list to strip literal strings
    instead of the default entity-bracket pattern.
    """
    if text is None:
        return ""

    if tokens:
        for tok in tokens:
            text = text.replace(tok, "")
        return _MULTI_SPACE_RE.sub(" ", text).strip()

    # Default: strip SLURP-style entity annotation brackets [type : value] -> value
    text = re.sub(r"\[[^:\]]+:\s*([^\]]+)\]", r"\1", text)
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def is_empty_or_whitespace(text: str) -> bool:
    """True if the transcript is None, empty, or only whitespace."""
    return text is None or text.strip() == ""
