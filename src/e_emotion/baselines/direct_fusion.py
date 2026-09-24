"""Project-owned direct-fusion control models.

Both models expose the same two heads: three-class polarity logits and a
scalar raw intensity.  Data loading, masks, normalization and output scoring
remain outside the model so the controls can be compared with reference
methods under the shared competition protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn


FEATURE_DIMS: dict[str, int] = {"text": 768, "audio": 74, "vision": 35}
MODALITIES: tuple[str, ...] = ("text", "audio", "vision")


@dataclass(frozen=True)
class FusionOutput:
    """Raw multitask model output before competition output adaptation."""

    polarity_logits: Tensor
    intensity: Tensor

    def __getitem__(self, key: str) -> Tensor:
        if key == "polarity_logits":
            return self.polarity_logits
        if key == "intensity":
            return self.intensity
        raise KeyError(key)


def _validate_inputs(
    features: Mapping[str, Tensor],
    masks: Mapping[str, Tensor] | None,
    *,
    require_same_steps: bool,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    missing = [modality for modality in MODALITIES if modality not in features]
    if missing:
        raise ValueError(f"missing modality features: {missing}")
    if masks is not None:
        missing_masks = [modality for modality in MODALITIES if modality not in masks]
        if missing_masks:
            raise ValueError(f"missing modality masks: {missing_masks}")

    checked_features: dict[str, Tensor] = {}
    checked_masks: dict[str, Tensor] = {}
    batch_size: int | None = None
    steps: int | None = None
    for modality in MODALITIES:
        value = features[modality]
        if not isinstance(value, Tensor) or value.ndim != 3:
            raise ValueError(f"{modality} features must be a rank-3 torch tensor")
        expected_dim = FEATURE_DIMS[modality]
        if value.shape[-1] != expected_dim:
            raise ValueError(
                f"{modality} feature dimension must be {expected_dim}, got {value.shape[-1]}"
            )
        if batch_size is None:
            batch_size = int(value.shape[0])
        elif value.shape[0] != batch_size:
            raise ValueError("all modalities must have the same batch size")
        if steps is None:
            steps = int(value.shape[1])
        elif require_same_steps and value.shape[1] != steps:
            raise ValueError("all modalities must have the same sequence length")
        checked_features[modality] = value

        if masks is None:
            mask = torch.ones(value.shape[:2], dtype=torch.bool, device=value.device)
        else:
            mask = masks[modality]
            if not isinstance(mask, Tensor) or mask.dtype != torch.bool:
                raise ValueError(f"{modality} mask must be a boolean torch tensor")
            if mask.shape != value.shape[:2]:
                raise ValueError(f"{modality} mask must match feature batch and sequence dimensions")
            if mask.device != value.device:
                raise ValueError(f"{modality} mask and features must be on the same device")
        checked_masks[modality] = mask
    return checked_features, checked_masks


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    weights = mask.unsqueeze(-1).to(dtype=value.dtype)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (value * weights).sum(dim=1) / denominator


class _MultiTaskHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.polarity = nn.Linear(hidden_dim, 3)
        self.intensity = nn.Linear(hidden_dim, 1)

    def forward(self, representation: Tensor) -> FusionOutput:
        hidden = self.backbone(representation)
        return FusionOutput(
            polarity_logits=self.polarity(hidden),
            intensity=self.intensity(hidden).squeeze(-1),
        )


class ConcatMLP(nn.Module):
    """Masked mean-pool each modality, concatenate, and predict jointly.

    Each modality can have a different sequence length, which keeps this
    control usable as a pooled reference for both aligned and unaligned data.
    """

    def __init__(self, *, hidden_dim: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.head = _MultiTaskHead(sum(FEATURE_DIMS.values()), hidden_dim, dropout)

    def forward(
        self,
        features: Mapping[str, Tensor],
        masks: Mapping[str, Tensor] | None = None,
    ) -> FusionOutput:
        checked_features, checked_masks = _validate_inputs(features, masks, require_same_steps=False)
        pooled = [_masked_mean(checked_features[modality], checked_masks[modality]) for modality in MODALITIES]
        return self.head(torch.cat(pooled, dim=-1))


class EarlyFusionGRU(nn.Module):
    """Concatenate modalities at each aligned position, then encode with GRU."""

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if bidirectional and hidden_dim % 2:
            raise ValueError("hidden_dim must be even for bidirectional GRU")
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_size=sum(FEATURE_DIMS.values()),
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        representation_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = _MultiTaskHead(representation_dim, hidden_dim, dropout)

    def forward(
        self,
        features: Mapping[str, Tensor],
        masks: Mapping[str, Tensor] | None = None,
    ) -> FusionOutput:
        checked_features, checked_masks = _validate_inputs(features, masks, require_same_steps=True)
        step_mask = torch.stack([checked_masks[modality] for modality in MODALITIES], dim=0).any(dim=0)
        sequence = torch.cat(
            [checked_features[modality] * checked_masks[modality].unsqueeze(-1).to(checked_features[modality].dtype)
             for modality in MODALITIES],
            dim=-1,
        )
        encoded, _ = self.gru(sequence)
        pooled = _masked_mean(encoded, step_mask)
        return self.head(pooled)


__all__ = ["ConcatMLP", "EarlyFusionGRU", "FEATURE_DIMS", "FusionOutput"]
