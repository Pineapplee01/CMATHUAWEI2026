import pytest
import numpy as np

torch = pytest.importorskip("torch")

from e_emotion.baselines import ConcatMLP, EarlyFusionGRU
from e_emotion.baselines import split_to_torch_inputs
from e_emotion.data.processed import ProcessedSplit


def _inputs(batch=3, steps=5):
    generator = torch.Generator().manual_seed(7)
    features = {
        "text": torch.randn(batch, steps, 768, generator=generator),
        "audio": torch.randn(batch, steps, 74, generator=generator),
        "vision": torch.randn(batch, steps, 35, generator=generator),
    }
    masks = {
        "text": torch.ones(batch, steps, dtype=torch.bool),
        "audio": torch.ones(batch, steps, dtype=torch.bool),
        "vision": torch.ones(batch, steps, dtype=torch.bool),
    }
    masks["audio"][:, -1] = False
    masks["vision"][:, 0] = False
    return features, masks


@pytest.mark.parametrize("model_type", [ConcatMLP, EarlyFusionGRU])
def test_direct_fusion_models_return_competition_heads(model_type):
    features, masks = _inputs()
    model = model_type(hidden_dim=16)

    output = model(features, masks)

    assert output.polarity_logits.shape == (3, 3)
    assert output.intensity.shape == (3,)
    assert output.intensity.dtype == features["text"].dtype


def test_concat_mlp_ignores_masked_values():
    features, masks = _inputs()
    model = ConcatMLP(hidden_dim=16).eval()
    first = model(features, masks)

    changed = {name: value.clone() for name, value in features.items()}
    changed["audio"][~masks["audio"]] = 10_000
    changed["vision"][~masks["vision"]] = -10_000
    second = model(changed, masks)

    torch.testing.assert_close(first.polarity_logits, second.polarity_logits)
    torch.testing.assert_close(first.intensity, second.intensity)


def test_early_fusion_gru_backpropagates_through_all_modalities():
    features, masks = _inputs()
    model = EarlyFusionGRU(hidden_dim=16)
    output = model(features, masks)
    loss = output.polarity_logits.square().mean() + output.intensity.square().mean()
    loss.backward()

    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_early_fusion_gru_rejects_mismatched_steps():
    features, masks = _inputs()
    features["audio"] = features["audio"][:, :-1]
    masks["audio"] = masks["audio"][:, :-1]

    with pytest.raises(ValueError, match="same sequence length"):
        EarlyFusionGRU(hidden_dim=16)(features, masks)


def test_processed_split_adapter_preserves_observed_masks():
    n, steps = 2, 5
    native = {modality: torch.ones(n, steps, dtype=torch.bool).numpy() for modality in ("text", "audio", "vision")}
    observed = {modality: value.copy() for modality, value in native.items()}
    observed["audio"][0, 2] = False
    split = ProcessedSplit(
        features={
            "text": torch.zeros(n, steps, 768).numpy(),
            "audio": torch.zeros(n, steps, 74).numpy(),
            "vision": torch.zeros(n, steps, 35).numpy(),
        },
        native_valid_mask=native,
        observed_mask=observed,
        ids=np.asarray(["a", "b"]),
        regression=torch.zeros(n).numpy(),
        classification=torch.ones(n, dtype=torch.int64).numpy(),
        source_path="fixture.npz",
    )

    features, masks = split_to_torch_inputs(split)

    assert features["audio"].dtype == torch.float32
    assert masks["audio"].dtype == torch.bool
    assert not bool(masks["audio"][0, 2])
