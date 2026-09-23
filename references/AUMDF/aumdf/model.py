"""Paper sections 3.4-3.8. Ambiguous choices are documented in REPRODUCTION.md."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

MODALITIES = ("text", "audio", "vision")


@dataclass(frozen=True)
class ModelConfig:
    input_dims: tuple[int, int, int] = (768, 74, 35)
    hidden_dim: int = 300
    heads: int = 6
    layers: int = 1
    dropout: float = 0.8
    window: int = 5
    conv_kernel: int = 3
    noise_threshold: float = 0.1
    readout: str = "regression"

    def __post_init__(self):
        if len(self.input_dims) != 3 or min(self.input_dims) < 1:
            raise ValueError("three positive input dimensions required")
        if self.hidden_dim <= 0 or self.heads <= 0 or self.hidden_dim % 2 or self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be positive, even and divisible by heads")
        if not 0 <= self.dropout < 1 or self.window < 0 or self.layers < 1:
            raise ValueError("invalid dropout/window/layers")
        if self.conv_kernel < 1 or self.conv_kernel % 2 == 0:
            raise ValueError("conv_kernel must be a positive odd integer")
        if self.readout not in {"regression", "classification"}:
            raise ValueError("readout must be regression or classification")


@dataclass
class ModelOutput:
    score: torch.Tensor
    fused: torch.Tensor
    modalities: dict[str, torch.Tensor]
    logits: torch.Tensor | None = None


def masked_mean(x, valid):
    weight = valid.to(x.dtype).unsqueeze(-1)
    return (x * weight).sum(1) / weight.sum(1).clamp_min(1)


def masked_moments(x, valid):
    mean = masked_mean(x, valid)
    variance = masked_mean((x - mean[:, None]).square(), valid)
    return mean, variance


class RepresentationEnhancement(nn.Module):
    """REM: independent global and filtering BiLSTMs, Eqs. (1)-(12)."""
    def __init__(self, input_dim, hidden_dim, threshold):
        super().__init__()
        self.global_lstm = nn.LSTM(input_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.filter_lstm = nn.LSTM(input_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.activation = nn.Linear(hidden_dim, hidden_dim)
        self.compensation = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh())
        self.mix_logit = nn.Parameter(torch.zeros(()))
        self.threshold = threshold

    @staticmethod
    def encode(lstm, x, valid):
        lengths = valid.sum(1).clamp_min(1).cpu()
        clean = x.masked_fill(~valid[..., None], 0)
        packed = pack_padded_sequence(clean, lengths, batch_first=True, enforce_sorted=False)
        hidden, _ = lstm(packed)
        hidden, _ = pad_packed_sequence(hidden, batch_first=True, total_length=x.shape[1])
        return hidden.masked_fill(~valid[..., None], 0)

    def forward(self, x, valid):
        global_hidden = self.encode(self.global_lstm, x, valid)
        mean, variance = masked_moments(global_hidden, valid)
        normalized = (global_hidden - mean[:, None]) / (variance[:, None] + 1e-5).sqrt()
        local_hidden = self.encode(self.filter_lstm, x, valid)
        activated = torch.relu(self.activation(local_hidden))
        mean, variance = masked_moments(local_hidden, valid)
        compensation = self.compensation(torch.cat((mean, (variance + 1e-5).sqrt()), -1))
        filtered = torch.where(activated >= self.threshold, activated, compensation[:, None])
        alpha = torch.sigmoid(self.mix_logit)
        enhanced = torch.cat((alpha * normalized, (1 - alpha) * filtered), -1)
        return enhanced.masked_fill(~valid[..., None], 0)


class RepresentationPreprocessing(nn.Module):
    """RPM: Conv1D + sinusoidal position encoding, Eqs. (13)-(16)."""
    def __init__(self, hidden_dim, kernel, dropout):
        super().__init__()
        self.conv = nn.Conv1d(2 * hidden_dim, hidden_dim, kernel, padding=kernel // 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        length, width = x.shape[1:]
        positions = torch.arange(length, device=x.device, dtype=x.dtype)[:, None]
        frequency = torch.exp(torch.arange(0, width, 2, device=x.device, dtype=x.dtype)
                              * (-math.log(10000) / width))
        pe = torch.zeros(length, width, device=x.device, dtype=x.dtype)
        pe[:, 0::2] = torch.sin(positions * frequency)
        pe[:, 1::2] = torch.cos(positions * frequency)
        return self.dropout(x + pe).masked_fill(~valid[..., None], 0)


class DynamicWeightAdjustment(nn.Module):
    """DWAM: pairwise ReLU gates, concatenate, multiply target projection."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.gates = nn.ModuleDict({
            f"{m}_{n}": nn.Linear(2 * hidden_dim, hidden_dim // 2)
            for m in MODALITIES for n in MODALITIES if m != n
        })
        self.projections = nn.ModuleDict({m: nn.Linear(hidden_dim, hidden_dim) for m in MODALITIES})

    def forward(self, features, valid):
        result = {}
        for m in MODALITIES:
            gates = [torch.relu(self.gates[f"{m}_{n}"](torch.cat((features[m], features[n]), -1)))
                     for n in MODALITIES if n != m]
            result[m] = (torch.cat(gates, -1) * self.projections[m](features[m])).masked_fill(
                ~valid[m][..., None], 0)
        return result


class TemporalCrossAttention(nn.Module):
    """Q, K, V from three different modalities; safe pre-softmax masking."""
    def __init__(self, hidden_dim, heads, window, dropout):
        super().__init__()
        self.heads, self.width, self.window = heads, hidden_dim // heads, window
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_valid):
        batch, time, width = query.shape
        def split(x):
            return x.reshape(batch, time, self.heads, self.width).transpose(1, 2)
        q, k, v = split(self.q(query)), split(self.k(key)), split(self.v(value))
        positions = torch.arange(time, device=query.device)
        distance = (positions[:, None] - positions[None, :]).abs()
        allowed = (distance <= self.window)[None, None] & key_valid[:, None, None, :]
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.width)
        scores = scores - distance.to(scores.dtype)[None, None] / max(1, self.window)
        weights = torch.softmax(scores.masked_fill(~allowed, torch.finfo(scores.dtype).min), -1)
        weights = weights * allowed
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        context = (self.dropout(weights) @ v).transpose(1, 2).reshape(batch, time, width)
        has_key = allowed.any(-1).squeeze(1)
        return self.output(context).masked_fill(~has_key[..., None], 0), weights


class MultimodalMaskedBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config.hidden_dim
        self.norms = nn.ModuleDict({m: nn.LayerNorm(h) for m in MODALITIES})
        self.ff_norms = nn.ModuleDict({m: nn.LayerNorm(h) for m in MODALITIES})
        self.attention = nn.ModuleDict({
            m: TemporalCrossAttention(h, config.heads, config.window, config.dropout)
            for m in MODALITIES
        })
        self.ff = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(h, 2*h), nn.ReLU(), nn.Dropout(config.dropout),
                             nn.Linear(2*h, h), nn.Dropout(config.dropout))
            for m in MODALITIES
        })

    def forward(self, features, valid, observed):
        z = {m: self.norms[m](features[m]) for m in MODALITIES}
        result = {}
        # Figure 3: text queries vision keys and audio values; cycle for others.
        triplets = (("text", "vision", "audio"), ("audio", "text", "vision"), ("vision", "audio", "text"))
        for m, key, value in triplets:
            available = valid[key] & valid[value] & observed[key] & observed[value]
            update, _ = self.attention[m](z[m], z[key], z[value], available)
            hidden = update + z[m]
            normalized = self.ff_norms[m](hidden)
            result[m] = (self.ff[m](normalized) + normalized).masked_fill(~valid[m][..., None], 0)
        return result


class AUMDF(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.rem = nn.ModuleDict({
            m: RepresentationEnhancement(d, config.hidden_dim, config.noise_threshold)
            for m, d in zip(MODALITIES, config.input_dims)
        })
        self.rpm = nn.ModuleDict({
            m: RepresentationPreprocessing(config.hidden_dim, config.conv_kernel, config.dropout)
            for m in MODALITIES
        })
        self.dwam = DynamicWeightAdjustment(config.hidden_dim)
        self.blocks = nn.ModuleList([MultimodalMaskedBlock(config) for _ in range(config.layers)])
        self.head = nn.Linear(3 * config.hidden_dim, 7 if config.readout == "classification" else 1)
        self.register_buffer("levels", torch.arange(-3, 4, dtype=torch.float32))

    def forward(self, features, valid, observed=None):
        shape = features["text"].shape[:2]
        for m, width in zip(MODALITIES, self.config.input_dims):
            if features[m].ndim != 3 or features[m].shape[:2] != shape:
                raise ValueError("AUMDF adapter requires aligned sequences with equal lengths")
            if features[m].shape[-1] != width or valid[m].shape != shape:
                raise ValueError(f"invalid feature dimension/mask for {m}")
        observed = observed or {m: valid[m] & features[m].ne(0).any(-1) for m in MODALITIES}
        z = {m: self.rpm[m](self.rem[m](features[m], valid[m]), valid[m]) for m in MODALITIES}
        z = self.dwam(z, valid)
        modalities = {m: masked_mean(z[m], valid[m]) for m in MODALITIES}
        for block in self.blocks:
            z = block(z, valid, observed)
        fused = torch.cat([masked_mean(z[m], valid[m]) for m in MODALITIES], -1)
        output = self.head(fused)
        if self.config.readout == "classification":
            score = (torch.softmax(output, -1) * self.levels).sum(-1)
            return ModelOutput(score, fused, modalities, output)
        return ModelOutput(output.squeeze(-1), fused, modalities)

    def qkv_penalty(self):
        matrices = [layer.weight for block in self.blocks for attn in block.attention.values()
                    for layer in (attn.q, attn.k, attn.v)]
        return sum(weight.square().sum() for weight in matrices)
