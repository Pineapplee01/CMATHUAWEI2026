"""Mask-capable multi-head attention used by the local MulT adapter."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as functional


class MaskedMultiheadAttention(nn.Module):
    def __init__(self, *, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        key_padding_mask: Tensor | None = None,
        query_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("attention tensors must be [time, batch, feature]")
        target_length, batch_size, width = query.shape
        source_length = key.shape[0]
        if width != self.embed_dim or key.shape != value.shape or key.shape[1:] != (batch_size, width):
            raise ValueError("incompatible attention shapes")
        if key_padding_mask is None:
            key_padding_mask = torch.zeros((batch_size, source_length), dtype=torch.bool, device=query.device)
        if key_padding_mask.shape != (batch_size, source_length):
            raise ValueError("key_padding_mask must be [batch, source_time]")
        if query_mask is None:
            query_mask = torch.ones((batch_size, target_length), dtype=torch.bool, device=query.device)
        if query_mask.shape != (batch_size, target_length):
            raise ValueError("query_mask must be [batch, target_time]")

        def split(values: Tensor) -> Tensor:
            return values.transpose(0, 1).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q = split(self.query(query)) * self.scale
        k = split(self.key(key))
        v = split(self.value(value))
        scores = torch.matmul(q, k.transpose(-2, -1))
        if attn_mask is not None:
            if attn_mask.shape != (target_length, source_length):
                raise ValueError("attn_mask must be [target_time, source_time]")
            scores = scores + attn_mask.to(device=scores.device, dtype=scores.dtype)[None, None]
        unavailable = key_padding_mask.to(device=scores.device, dtype=torch.bool)[:, None, None]
        scores = scores.masked_fill(unavailable, torch.finfo(scores.dtype).min)
        has_key = (~key_padding_mask).any(dim=1)
        weights = functional.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        weights = self.dropout(weights)
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).reshape(batch_size, target_length, self.embed_dim).transpose(0, 1)
        output = self.output(context)
        valid_query = query_mask.to(device=output.device, dtype=torch.bool).transpose(0, 1).unsqueeze(-1)
        output = output.masked_fill(~valid_query, 0.0)
        output = output.masked_fill(~has_key[None, :, None], 0.0)
        return output, weights.mean(dim=1)
