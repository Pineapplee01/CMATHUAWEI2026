"""Regression test for P-RMF proxy weighting with a low-variance modality."""
from __future__ import annotations

import torch
from torch import nn

from models.generate_proxy_modality import Generate_Proxy_Modality


class FixedLowVarianceVAE(nn.Module):
    def forward(self, value: torch.Tensor):
        mu = torch.zeros_like(value)
        # exp(1 / exp(-15)) overflows in float32. The mixture should still be
        # a finite normalized distribution, because it is a softmax in effect.
        log_var = torch.full_like(value, -30.0)
        return value, mu, log_var


def test_proxy_weighting_remains_finite_for_low_variance() -> None:
    module = Generate_Proxy_Modality.__new__(Generate_Proxy_Modality)
    nn.Module.__init__(module)
    module.text_VAE = FixedLowVarianceVAE()
    module.image_VAE = FixedLowVarianceVAE()
    module.audio_VAE = FixedLowVarianceVAE()
    sequence = torch.zeros((2, 8, 4), dtype=torch.float32)

    loss, proxy, weights = module(sequence, sequence, sequence, None, None, None)

    assert torch.isfinite(loss)
    assert torch.isfinite(proxy).all()
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights.sum(dim=0), torch.ones_like(weights[0]))
