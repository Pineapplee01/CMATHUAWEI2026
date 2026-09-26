"""Mask-aware early-fusion LSTM for canonical aligned multimodal features."""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as functional


class MaskAwareEFLSTM(nn.Module):
    """Fuse three aligned modalities without updating through empty timesteps."""

    def __init__(
        self,
        *,
        text_dim: int = 768,
        audio_dim: int = 74,
        vision_dim: int = 35,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        input_dim = text_dim + audio_dim + vision_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.cells = nn.ModuleList(
            [
                nn.LSTMCell(input_dim if layer == 0 else hidden_dim, hidden_dim)
                for layer in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _checked_mask(mask: Tensor, values: Tensor, name: str) -> Tensor:
        if mask.shape != values.shape[:2]:
            raise ValueError(f"{name} mask must have shape {tuple(values.shape[:2])}")
        return mask.to(device=values.device, dtype=torch.bool)

    def forward(
        self,
        text: Tensor,
        audio: Tensor,
        vision: Tensor,
        masks: Mapping[str, Tensor],
    ) -> Tensor:
        if text.ndim != audio.ndim or text.ndim != vision.ndim or text.ndim != 3:
            raise ValueError("modalities must be [batch, time, feature]")
        if text.shape[:2] != audio.shape[:2] or text.shape[:2] != vision.shape[:2]:
            raise ValueError("modalities must share batch and time dimensions")
        required = {"text", "audio", "vision"}
        if set(masks) != required:
            raise ValueError(f"masks must contain exactly {sorted(required)}")

        text_mask = self._checked_mask(masks["text"], text, "text")
        audio_mask = self._checked_mask(masks["audio"], audio, "audio")
        vision_mask = self._checked_mask(masks["vision"], vision, "vision")
        fused = torch.cat(
            [
                text * text_mask.unsqueeze(-1),
                audio * audio_mask.unsqueeze(-1),
                vision * vision_mask.unsqueeze(-1),
            ],
            dim=-1,
        )
        fused = self.input_norm(fused)
        active = text_mask | audio_mask | vision_mask
        batch_size = fused.shape[0]
        states = [
            (
                fused.new_zeros((batch_size, cell.hidden_size)),
                fused.new_zeros((batch_size, cell.hidden_size)),
            )
            for cell in self.cells
        ]
        for timestep in range(fused.shape[1]):
            update = active[:, timestep].unsqueeze(-1)
            value = fused[:, timestep]
            next_states = []
            for layer, cell in enumerate(self.cells):
                previous_hidden, previous_cell = states[layer]
                candidate_hidden, candidate_cell = cell(value, (previous_hidden, previous_cell))
                hidden = torch.where(update, candidate_hidden, previous_hidden)
                cell_state = torch.where(update, candidate_cell, previous_cell)
                next_states.append((hidden, cell_state))
                value = hidden
            states = next_states
        final_hidden = self.dropout(states[-1][0])
        final_hidden = self.dropout(functional.relu(self.hidden(final_hidden)))
        return self.output(final_hidden)
