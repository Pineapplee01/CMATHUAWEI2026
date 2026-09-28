"""全深度适配的梯度与训练/推理模式回归检查。"""
import unittest
from unittest.mock import patch
import torch
from transformers import BertConfig, BertModel
from problem2.text_encoder import TextEncoder


class TextAdaptationContracts(unittest.TestCase):
    def test_ffn_adapters_gradient_and_dropout_mode(self):
        torch.manual_seed(4)
        backbone = BertModel(BertConfig(vocab_size=512, hidden_size=768,
            intermediate_size=64, num_hidden_layers=2, num_attention_heads=12))
        with patch('problem2.text_encoder.AutoModel.from_pretrained', return_value=backbone):
            encoder = TextEncoder('unused')
        encoder.enable_adapters(2, 4, 2, ffn=True)
        encoder.backbone_dropout = True
        encoder.train()
        self.assertTrue(backbone.training)
        tokens = torch.zeros(2,3,5,dtype=torch.long)
        tokens[:,0,:3] = torch.tensor([101,105,102])
        observed = torch.zeros(2,5,dtype=torch.bool); observed[:,1] = True
        encoder(tokens, observed)[...,0].sum().backward()
        self.assertGreater(float(backbone.encoder.layer[-1].output.dense.lora_B.grad.abs().sum()),0)
        self.assertIsNone(backbone.embeddings.word_embeddings.weight.grad)
        self.assertEqual(len(encoder.cache), 0)
        encoder.eval()
        self.assertFalse(backbone.training)
        with torch.no_grad():
            a = encoder(tokens, observed); b = encoder(tokens, observed)
        torch.testing.assert_close(a, b)


if __name__=='__main__': unittest.main()
