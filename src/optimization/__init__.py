"""Optimization utilities for VoxIntel."""

from .quantization import benchmark_latency, quantize_dynamic_int8, save_model_stats

__all__ = ["benchmark_latency", "quantize_dynamic_int8", "save_model_stats"]
