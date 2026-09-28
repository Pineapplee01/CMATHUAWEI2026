"""验证新正则项的数学性质，防止单侧detach或类别次序错误。"""
import unittest
import torch
from problem2.losses import dropout_consistency, task_loss

class BoundaryObjectiveTests(unittest.TestCase):
    def test_symmetric_kl_and_both_gradients(self):
        a=torch.tensor([[2.,0.,-1.]],requires_grad=True)
        b=torch.tensor([[0.,1.,-1.]],requires_grad=True)
        x=dropout_consistency(a,b)
        torch.testing.assert_close(x,dropout_consistency(b,a))
        self.assertGreater(float(x),0)
        x.backward()
        self.assertGreater(float(a.grad.abs().sum()),0)
        self.assertGreater(float(b.grad.abs().sum()),0)
        torch.testing.assert_close(dropout_consistency(a,a),torch.tensor(0.),atol=1e-6,rtol=0)

    def test_ordinal_penalizes_opposite_more_than_adjacent(self):
        batch={'c':torch.tensor([0]),'y':torch.tensor([0.])}
        cfg={'regression_weight':0.,'ordinal_weight':1.}
        def extra(logits):
            out={'logits':torch.tensor([logits]),'intensity':torch.zeros(1)}
            return task_loss(out,batch,cfg)-task_loss(out,batch,dict(cfg,ordinal_weight=0))
        self.assertGreater(float(extra([-8.,-8.,8.])),float(extra([-8.,8.,-8.])))

if __name__=='__main__': unittest.main()
