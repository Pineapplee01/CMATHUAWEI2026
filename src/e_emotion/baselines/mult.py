"""Mask-aware MulT adapter with cross-modal attention over valid positions."""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as functional

from e_emotion.baselines.mult_attention import MaskedMultiheadAttention


class MaskedTransformerLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = MaskedMultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout)
        self.norm_attention = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )
        self.norm_feed_forward = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        query_mask: Tensor,
        key_mask: Tensor,
    ) -> Tensor:
        attended, _ = self.attention(
            self.norm_attention(query),
            key,
            key,
            key_padding_mask=~key_mask,
            query_mask=query_mask,
        )
        mask = query_mask.transpose(0, 1).unsqueeze(-1)
        output = (query + self.dropout(attended)).masked_fill(~mask, 0.0)
        feed_forward = self.feed_forward(self.norm_feed_forward(output))
        return (output + self.dropout(feed_forward)).masked_fill(~mask, 0.0)


class MaskedTransformerStack(nn.Module):
    def __init__(self, dim: int, heads: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([MaskedTransformerLayer(dim, heads, dropout) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, query: Tensor, key: Tensor, query_mask: Tensor, key_mask: Tensor) -> Tensor:
        output = query
        for layer in self.layers:
            output = layer(output, key, query_mask, key_mask)
        mask = query_mask.transpose(0, 1).unsqueeze(-1)
        return self.final_norm(output).masked_fill(~mask, 0.0)


class MaskedMULT(nn.Module):
    """MulT-style cross-modal encoder that accepts one mask per modality."""

    def __init__(
        self,
        *,
        text_dim: int = 768,
        audio_dim: int = 74,
        vision_dim: int = 35,
        model_dim: int = 30,
        heads: int = 6,
        layers: int = 4,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
        self.text_projection = nn.Conv1d(text_dim, model_dim, kernel_size=1, bias=False)
        self.audio_projection = nn.Conv1d(audio_dim, model_dim, kernel_size=1, bias=False)
        self.vision_projection = nn.Conv1d(vision_dim, model_dim, kernel_size=1, bias=False)
        self.text_from_audio = MaskedTransformerStack(model_dim, heads, layers, dropout)
        self.text_from_vision = MaskedTransformerStack(model_dim, heads, layers, dropout)
        self.audio_from_text = MaskedTransformerStack(model_dim, heads, layers, dropout)
        self.audio_from_vision = MaskedTransformerStack(model_dim, heads, layers, dropout)
        self.vision_from_text = MaskedTransformerStack(model_dim, heads, layers, dropout)
        self.vision_from_audio = MaskedTransformerStack(model_dim, heads, layers, dropout)
        self.text_memory = MaskedTransformerStack(2 * model_dim, heads, layers, dropout)
        self.audio_memory = MaskedTransformerStack(2 * model_dim, heads, layers, dropout)
        self.vision_memory = MaskedTransformerStack(2 * model_dim, heads, layers, dropout)
        combined = 6 * model_dim
        self.proj1 = nn.Linear(combined, combined)
        self.proj2 = nn.Linear(combined, combined)
        self.output = nn.Linear(combined, 1)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _mask(mask: Tensor, values: Tensor, name: str) -> Tensor:
        if mask.shape != values.shape[:2]:
            raise ValueError(f"{name} mask must be [batch, time]")
        return mask.to(device=values.device, dtype=torch.bool)

    @staticmethod
    def _last_observed(sequence: Tensor, mask: Tensor) -> Tensor:
        batch_size, length = mask.shape
        indices = torch.arange(length, device=sequence.device).unsqueeze(0).expand(batch_size, -1)
        positions = torch.where(mask, indices, torch.full_like(indices, -1)).amax(dim=1).clamp_min(0)
        return sequence.transpose(0, 1)[torch.arange(batch_size, device=sequence.device), positions]

    @staticmethod
    def _project(values: Tensor, mask: Tensor, projection: nn.Conv1d) -> Tensor:
        values = values.masked_fill(~mask.unsqueeze(-1), 0.0)
        return projection(values.transpose(1, 2)).transpose(1, 2).transpose(0, 1)

    def forward(
        self,
        text: Tensor,
        audio: Tensor,
        vision: Tensor,
        masks: Mapping[str, Tensor],
    ) -> Tensor:
        required = {"text", "audio", "vision"}
        if set(masks) != required:
            raise ValueError(f"masks must contain exactly {sorted(required)}")
        text_mask = self._mask(masks["text"], text, "text")
        audio_mask = self._mask(masks["audio"], audio, "audio")
        vision_mask = self._mask(masks["vision"], vision, "vision")
        text_values = self._project(text, text_mask, self.text_projection)
        audio_values = self._project(audio, audio_mask, self.audio_projection)
        vision_values = self._project(vision, vision_mask, self.vision_projection)

        text_pair = torch.cat(
            [
                self.text_from_audio(text_values, audio_values, text_mask, audio_mask),
                self.text_from_vision(text_values, vision_values, text_mask, vision_mask),
            ],
            dim=-1,
        )
        audio_pair = torch.cat(
            [
                self.audio_from_text(audio_values, text_values, audio_mask, text_mask),
                self.audio_from_vision(audio_values, vision_values, audio_mask, vision_mask),
            ],
            dim=-1,
        )
        vision_pair = torch.cat(
            [
                self.vision_from_text(vision_values, text_values, vision_mask, text_mask),
                self.vision_from_audio(vision_values, audio_values, vision_mask, audio_mask),
            ],
            dim=-1,
        )
        text_context = self.text_memory(text_pair, text_pair, text_mask, text_mask)
        audio_context = self.audio_memory(audio_pair, audio_pair, audio_mask, audio_mask)
        vision_context = self.vision_memory(vision_pair, vision_pair, vision_mask, vision_mask)
        representation = torch.cat(
            [
                self._last_observed(text_context, text_mask),
                self._last_observed(audio_context, audio_mask),
                self._last_observed(vision_context, vision_mask),
            ],
            dim=-1,
        )
        projected = self.proj2(self.dropout(functional.relu(self.proj1(representation))))
        return self.output(projected + representation)
