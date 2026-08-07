"""
Reusable audio I/O and inspection utilities for VoxIntel.

Kept dependency-light (soundfile + numpy) so these functions can be reused
across the dataset-audit notebook, ASR training/inference pipeline, and the
FastAPI serving layer without pulling in torch/torchaudio where it's not
needed.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import soundfile as sf

PathLike = Union[str, Path]

__all__ = [
    "get_duration",
    "load_audio",
    "resample_audio",
    "is_valid_audio",
    "get_audio_stats",
]


def get_duration(path: PathLike) -> float:
    """Return the duration of an audio file in seconds.

    Reads only the file header (via soundfile.info), so this is fast even
    for large datasets — it does not decode the full waveform.

    Returns np.nan if the file can't be read (missing, corrupted, unsupported
    format), so callers can vectorize this with `.apply()` over a DataFrame
    without try/except at every call site.
    """
    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception:
        return float("nan")


def load_audio(
    path: PathLike,
    target_sr: Optional[int] = 16_000,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """Load an audio file as a float32 numpy array.

    Args:
        path: path to the audio file.
        target_sr: if set, resample to this sample rate (requires librosa).
            Pass None to keep the file's native sample rate.
        mono: if True and the file has multiple channels, average them down
            to a single channel.

    Returns:
        (waveform, sample_rate) where waveform is a 1D float32 array
        (mono) or 2D (channels, samples) if mono=False.

    Raises:
        FileNotFoundError: if the path doesn't exist.
        RuntimeError: if the file exists but can't be decoded.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        waveform, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as e:
        raise RuntimeError(f"Could not decode audio file {path}: {e}") from e

    if waveform.ndim > 1 and mono:
        waveform = waveform.mean(axis=1)

    if target_sr is not None and sr != target_sr:
        waveform, sr = resample_audio(waveform, sr, target_sr)

    return waveform.astype(np.float32), sr


def resample_audio(waveform: np.ndarray, orig_sr: int, target_sr: int) -> tuple[np.ndarray, int]:
    """Resample a waveform to a new sample rate.

    Uses librosa if available (better quality); falls back to a basic
    linear-interpolation resampler so this module doesn't hard-require
    librosa for the common case (SLURP audio is already 16kHz).
    """
    if orig_sr == target_sr:
        return waveform, orig_sr

    try:
        import librosa
        resampled = librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
        return resampled.astype(np.float32), target_sr
    except ImportError:
        # Fallback: naive linear interpolation resampler.
        # Fine for quick checks; prefer librosa/torchaudio for training data.
        duration = len(waveform) / orig_sr
        n_target_samples = int(round(duration * target_sr))
        orig_idx = np.linspace(0, len(waveform) - 1, num=len(waveform))
        target_idx = np.linspace(0, len(waveform) - 1, num=n_target_samples)
        resampled = np.interp(target_idx, orig_idx, waveform)
        return resampled.astype(np.float32), target_sr


def is_valid_audio(path: PathLike) -> bool:
    """Quick check: can this file be read as audio at all?"""
    try:
        sf.info(str(path))
        return True
    except Exception:
        return False


def get_audio_stats(path: PathLike) -> dict:
    """Return a small dict of metadata for one audio file: duration,
    sample rate, channels, format. Returns Nones on failure rather than
    raising, so it's safe to use in bulk profiling loops.
    """
    try:
        info = sf.info(str(path))
        return {
            "duration_sec": info.frames / info.samplerate,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "format": info.format,
            "subtype": info.subtype,
        }
    except Exception:
        return {
            "duration_sec": None,
            "sample_rate": None,
            "channels": None,
            "format": None,
            "subtype": None,
        }
