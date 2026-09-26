"""Project-owned comparison baselines.

The package keeps Torch-dependent implementations behind a lazy interface so
that registry and artifact commands remain usable without model dependencies.
"""

__all__ = ["ConcatMLP", "EarlyFusionGRU", "FusionOutput", "split_to_torch_inputs"]


def __getattr__(name: str):
    if name == "split_to_torch_inputs":
        from e_emotion.baselines.adapters import split_to_torch_inputs

        return split_to_torch_inputs
    if name in {"ConcatMLP", "EarlyFusionGRU", "FusionOutput"}:
        from e_emotion.baselines.direct_fusion import ConcatMLP, EarlyFusionGRU, FusionOutput

        return {"ConcatMLP": ConcatMLP, "EarlyFusionGRU": EarlyFusionGRU, "FusionOutput": FusionOutput}[name]
    raise AttributeError(name)
