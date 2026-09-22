"""Model optimization utilities for the production-facing VoxIntel pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def quantize_dynamic_int8(model: Any, **kwargs: Any) -> Any:
    """Return a dynamically quantized PyTorch model when torch supports it."""
    if torch is None:
        raise ImportError("PyTorch is required for dynamic quantization.")

    try:
        return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, **kwargs)
    except Exception as exc:  # pragma: no cover - best-effort utility.
        raise RuntimeError(f"Dynamic quantization failed: {exc}") from exc


def benchmark_latency(func, *args, n_runs: int = 20, warmup: int = 5, **kwargs) -> Dict[str, float]:
    """Benchmark a callable and return average latency statistics in milliseconds."""
    if warmup > 0:
        for _ in range(warmup):
            func(*args, **kwargs)

    timings = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        timings.append((time.perf_counter() - start) * 1000.0)

    return {
        "runs": float(n_runs),
        "mean_ms": float(sum(timings) / len(timings)),
        "min_ms": float(min(timings)),
        "max_ms": float(max(timings)),
        "p95_ms": float(sorted(timings)[max(0, int(0.95 * len(timings)) - 1)]),
    }


def save_model_stats(model: Any, output_path: str | Path) -> None:
    """Persist a simple JSON summary of model metadata for notebook reporting."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "model_type": type(model).__name__,
        "parameters": getattr(model, "num_parameters", lambda: None)(),
    }

    with path.open("w", encoding="utf-8") as handle:
        import json
        json.dump(summary, handle, indent=2, sort_keys=True)


__all__ = ["quantize_dynamic_int8", "benchmark_latency", "save_model_stats"]
