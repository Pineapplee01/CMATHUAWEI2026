"""Adapters from the shared processed-data contract to torch model inputs."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor

from e_emotion.data.processed import MODALITIES, ProcessedSplit


def split_to_torch_inputs(
    split: ProcessedSplit,
    *,
    indices: Tensor | None = None,
    device: torch.device | str | None = None,
) -> tuple[Mapping[str, Tensor], Mapping[str, Tensor]]:
    """Convert a processed split without changing its masks or feature values."""
    if indices is not None:
        if indices.dtype != torch.long or indices.ndim != 1:
            raise ValueError("indices must be a one-dimensional torch.long tensor")
        selected = indices.detach().cpu().numpy()
    else:
        selected = slice(None)
    features = {
        modality: torch.as_tensor(split.features[modality][selected], device=device)
        for modality in MODALITIES
    }
    masks = {
        modality: torch.as_tensor(split.observed_mask[modality][selected], device=device)
        for modality in MODALITIES
    }
    return features, masks


__all__ = ["split_to_torch_inputs"]
