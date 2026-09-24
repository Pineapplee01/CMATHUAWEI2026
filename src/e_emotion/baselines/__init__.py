"""Project-owned comparison baselines."""

from e_emotion.baselines.adapters import split_to_torch_inputs
from e_emotion.baselines.direct_fusion import ConcatMLP, EarlyFusionGRU, FusionOutput

__all__ = ["ConcatMLP", "EarlyFusionGRU", "FusionOutput", "split_to_torch_inputs"]
