import torch

from masked_attention import MaskedMultiheadAttention


def test_cross_attention_ignores_unobserved_keys():
    torch.manual_seed(7)
    layer = MaskedMultiheadAttention(embed_dim=4, num_heads=2, dropout=0.0).eval()
    query = torch.randn(3, 1, 4)
    key = torch.randn(4, 1, 4)
    value = key.clone()
    key_mask = torch.tensor([[True, True, False, False]])

    baseline, _ = layer(query, key, value, key_padding_mask=~key_mask)
    key[2:] = 1000.0
    value[2:] = -1000.0
    changed, _ = layer(query, key, value, key_padding_mask=~key_mask)

    torch.testing.assert_close(baseline, changed, rtol=0, atol=1e-6)


def test_cross_attention_zeros_unobserved_queries():
    torch.manual_seed(7)
    layer = MaskedMultiheadAttention(embed_dim=4, num_heads=2, dropout=0.0).eval()
    query = torch.randn(3, 1, 4)
    key = torch.randn(4, 1, 4)
    output, _ = layer(
        query,
        key,
        key,
        key_padding_mask=torch.tensor([[False, False, False, False]]),
        query_mask=torch.tensor([[True, False, True]]),
    )

    torch.testing.assert_close(output[1], torch.zeros_like(output[1]), rtol=0, atol=0)
