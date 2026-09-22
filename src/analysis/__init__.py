"""Analysis utilities for VoxIntel-R (frozen-split validation, risk scoring)."""

from .frozen_split_eval import main as run_frozen_split_eval

__all__ = ["run_frozen_split_eval"]
