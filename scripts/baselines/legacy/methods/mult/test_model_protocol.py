import torch

from masked_mult import MaskedMULT


def test_mult_ignores_each_modality_at_unobserved_positions():
    torch.manual_seed(7)
    model = MaskedMULT(
        text_dim=3,
        audio_dim=2,
        vision_dim=2,
        model_dim=4,
        heads=2,
        layers=1,
        dropout=0.0,
    ).eval()
    text = torch.randn(2, 5, 3)
    audio = torch.randn(2, 5, 2)
    vision = torch.randn(2, 5, 2)
    masks = {
        "text": torch.tensor([[True, True, True, False, False], [True, True, False, True, False]]),
        "audio": torch.tensor([[True, True, False, False, False], [True, False, False, True, False]]),
        "vision": torch.tensor([[True, False, True, False, False], [True, True, False, False, False]]),
    }
    baseline = model(text, audio, vision, masks)
    changed = {"text": text.clone(), "audio": audio.clone(), "vision": vision.clone()}
    for name, values in changed.items():
        values[~masks[name]] = torch.randn_like(values[~masks[name]]) * 1000

    torch.testing.assert_close(baseline, model(**changed, masks=masks), rtol=0, atol=1e-6)
