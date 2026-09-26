import pytest
import torch

from e_emotion.baselines.aumdf.model import AUMDF, ModelConfig, TemporalCrossAttention


def inputs(batch=3, time=6):
    generator = torch.Generator().manual_seed(31)
    xs = {m: torch.randn(batch, time, d, generator=generator)
          for m, d in zip(("text", "audio", "vision"), (8, 4, 2))}
    masks = {m: torch.arange(time)[None, :].expand(batch, -1) < 4 for m in xs}
    return xs, masks


def small_config(**kwargs):
    return ModelConfig(input_dims=(8, 4, 2), hidden_dim=12, heads=3, dropout=0, **kwargs)


def test_shapes_and_gradient_flow():
    model = AUMDF(small_config())
    xs, masks = inputs()
    out = model(xs, masks)
    assert out.score.shape == (3,)
    assert out.fused.shape == (3, 36)
    assert set(out.modalities) == {"text", "audio", "vision"}
    out.score.sum().backward()
    assert model.rem["text"].global_lstm.weight_ih_l0.grad is not None
    assert model.dwam.gates["text_audio"].weight.grad is not None


def test_padding_neither_changes_output_nor_receives_attention():
    model = AUMDF(small_config()).eval()
    xs, masks = inputs()
    before = model(xs, masks).score
    altered = {m: x.masked_fill(~masks[m][..., None], 1e5) for m, x in xs.items()}
    torch.testing.assert_close(before, model(altered, masks).score, atol=1e-5, rtol=1e-5)


def test_all_missing_attention_is_finite_and_zero():
    attention = TemporalCrossAttention(12, 3, window=1, dropout=0)
    x = torch.randn(2, 5, 12)
    result, weights = attention(x, x, x, torch.zeros(2, 5, dtype=torch.bool))
    assert torch.isfinite(result).all()
    assert torch.count_nonzero(result) == 0
    assert torch.count_nonzero(weights) == 0


def test_temporal_window_and_mask():
    attention = TemporalCrossAttention(12, 3, window=1, dropout=0)
    x = torch.randn(2, 5, 12)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[:, -1] = False
    _, weights = attention(x, x, x, mask)
    positions = torch.arange(5)
    far = (positions[:, None] - positions[None, :]).abs() > 1
    assert torch.count_nonzero(weights[..., far]) == 0
    assert torch.count_nonzero(weights[..., -1]) == 0


def test_entirely_missing_input_and_classification_mode():
    model = AUMDF(small_config(readout="classification"))
    xs, masks = inputs()
    out = model({m: torch.zeros_like(x) for m, x in xs.items()}, masks)
    assert torch.isfinite(out.score).all()
    assert out.logits.shape == (3, 7)
    assert out.score.abs().max() <= 3


def test_unaligned_shapes_are_not_silently_resized():
    model = AUMDF(small_config())
    xs, masks = inputs()
    xs["audio"] = xs["audio"][:, :2]
    with pytest.raises(ValueError, match="aligned"):
        model(xs, masks)
