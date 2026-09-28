"""参数平均的恢复行为和对比损失数值检查。"""
import unittest
import torch
from problem2.averaging import ExponentialAverage
from problem2.losses import supervised_contrastive
from tests import test_model_contracts as contracts


class RegularizerContracts(unittest.TestCase):
    def test_average_restores_training_weights_on_failure(self):
        case=contracts.ModelContracts();case.setUp();model=case.model
        model.text.adapters_active=False
        ema=ExponentialAverage(model,.99)
        with torch.no_grad():model.head.bias.add_(1)
        ema.update(model);raw=model.head.bias.detach().clone()
        with self.assertRaises(RuntimeError):
            with ema.average_parameters(model):
                self.assertFalse(torch.equal(model.head.bias,raw))
                raise RuntimeError('simulated evaluation failure')
        torch.testing.assert_close(model.head.bias,raw)

    def test_supervised_contrastive_has_finite_gradients(self):
        a=torch.randn(4,3,10,11,requires_grad=True)
        b=torch.randn(4,3,10,11,requires_grad=True)
        loss=supervised_contrastive(a,b,torch.tensor([0,1,2,1]))
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertGreater(float(b.grad.abs().sum()),0.)


class TemporalContracts(contracts.ModelContracts):
    def setUp(self):
        from problem2.temporal import TemporalReadout
        super().setUp()
        self.model.temporal=TemporalReadout(self.cfg['hidden']).eval()


if __name__=='__main__':unittest.main()
