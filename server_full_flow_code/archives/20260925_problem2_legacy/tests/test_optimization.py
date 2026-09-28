"""优化路径的梯度、加性账本与重建评价回归检查。"""
import unittest
import torch
from torch import nn
from problem2.adapters import LowRankLinear
from problem2.heads import AdditiveHead
from tests import test_model_contracts as contracts


class OptimizationContracts(unittest.TestCase):
    def test_low_rank_initial_identity_and_gradient(self):
        base = nn.Linear(8, 8)
        adapted = LowRankLinear(base, 2, 4)
        x = torch.randn(3, 8)
        torch.testing.assert_close(base(x), adapted(x))
        adapted(x).square().sum().backward()
        self.assertIsNone(base.weight.grad)
        self.assertGreater(float(adapted.lora_B.grad.abs().sum()), 0)

    def test_normalized_ledger_and_bias(self):
        head = AdditiveHead(8, 5, normalize=True)
        pooled = torch.randn(2, 3, 10, 11)
        pooled[..., -3:] = 0
        pooled[:, :, :4, -3] = 1
        head.decision_bias[1] = .2
        score, ledger = head(pooled)
        expected = ledger['bias'] + ledger['single'].sum((1, 2)) + ledger['pair'].sum((1, 2))
        torch.testing.assert_close(score, expected)
        self.assertEqual(float(ledger['single'][:, :, 4:].abs().sum()), 0)

    def test_missing_evaluation_collects_reconstruction(self):
        from unittest.mock import patch
        from problem2.validation import evaluate
        case = contracts.ModelContracts(); case.setUp()
        batch = case.batch
        batch['id'] = ['a$_$0', 'b$_$0']
        cfg = {'seed': 7, 'device': 'cpu', 'batch_size': 2}
        with patch('problem2.validation.loader', return_value=[batch]):
            pred, rec = evaluate(case.model, None, cfg, rate=.3)
        self.assertEqual(len(pred), 2)
        self.assertGreater(len(rec), 0)
        self.assertTrue(set(rec.modality) == {'text', 'audio', 'vision'})


if __name__ == '__main__':
    unittest.main()
