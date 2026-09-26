import torch

from model import MaskAwareEFLSTM


def build_model():
    torch.manual_seed(7)
    return MaskAwareEFLSTM(
        text_dim=3,
        audio_dim=2,
        vision_dim=2,
        hidden_dim=5,
        num_layers=1,
        dropout=0.0,
    ).eval()


def test_ef_lstm_ignores_values_at_unobserved_positions():
    model = build_model()
    text = torch.randn(2, 4, 3)
    audio = torch.randn(2, 4, 2)
    vision = torch.randn(2, 4, 2)
    masks = {
        "text": torch.tensor([[True, True, False, False], [True, False, True, False]]),
        "audio": torch.tensor([[True, False, False, False], [True, False, True, False]]),
        "vision": torch.tensor([[True, True, False, False], [False, False, True, False]]),
    }
    baseline = model(text, audio, vision, masks)

    altered = {"text": text.clone(), "audio": audio.clone(), "vision": vision.clone()}
    for name, values in altered.items():
        values[~masks[name]] = 1000.0

    torch.testing.assert_close(baseline, model(**altered, masks=masks), rtol=0, atol=1e-6)


def test_ef_lstm_does_not_update_on_fully_unobserved_timestep():
    model = build_model()
    text = torch.randn(1, 3, 3)
    audio = torch.randn(1, 3, 2)
    vision = torch.randn(1, 3, 2)
    masks = {
        "text": torch.tensor([[True, False, True]]),
        "audio": torch.tensor([[True, False, True]]),
        "vision": torch.tensor([[True, False, True]]),
    }
    baseline = model(text, audio, vision, masks)
    text[:, 1] = -1000.0
    audio[:, 1] = 1000.0
    vision[:, 1] = -1000.0
    torch.testing.assert_close(baseline, model(text, audio, vision, masks), rtol=0, atol=1e-6)
