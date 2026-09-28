"""中性排序约束：平移不变、梯度方向、目标范围和缺少配对的处理。"""
import unittest
import torch
from problem2.model import neutral_boundary_loss, boundary_settings, objective


class NeutralBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.logits=torch.tensor([[.2,-.4,.1],[-.1,.6,.4],[.3,.1,-.2]],requires_grad=True)
        self.batch={'c':torch.tensor([1,2,0]),'y':torch.tensor([0.,1/3,-2.])}

    def test_gradient_improves_neutral_weak_separation(self):
        loss=neutral_boundary_loss(self.logits,self.batch)
        gradient=torch.autograd.grad(loss,self.logits)[0]
        self.assertLess(gradient[0,1].item(),0)
        self.assertGreater(gradient[1,1].item(),0)
        torch.testing.assert_close(gradient[2],torch.zeros(3))
        updated=self.logits.detach()-.1*gradient
        self.assertLess(neutral_boundary_loss(updated,self.batch).item(),loss.item())

    def test_invariant_to_per_sample_common_logit_shift(self):
        shift=torch.tensor([[10.],[-7.],[2.]])
        torch.testing.assert_close(neutral_boundary_loss(self.logits,self.batch),
                                   neutral_boundary_loss(self.logits+shift,self.batch))

    def test_no_valid_pair_has_zero_finite_gradient(self):
        for c,y in [([0,2,0],[-.3,.3,-2.]),([1,2,0],[0.,1.,-2.])]:
            loss=neutral_boundary_loss(self.logits,{'c':torch.tensor(c),'y':torch.tensor(y)})
            self.assertEqual(loss.item(),0.)
            torch.testing.assert_close(torch.autograd.grad(loss,self.logits)[0],torch.zeros_like(self.logits))

    def test_zero_boundary_weight_preserves_objective(self):
        out={'logits':self.logits,'intensity':torch.zeros(3)}
        torch.testing.assert_close(objective(out,self.batch),
                                   objective(out,self.batch,boundary_weight=0.),rtol=0,atol=0)

    def test_invalid_config_rejected(self):
        for cfg in ({'boundary_weight':-1},{'boundary_margin':-1},{'boundary_weight':float('nan')},
                    {'boundary_warmup_epochs':0},{'boundary_warmup_epochs':1.5},
                    {'boundary_max_intensity':0},{'boundary_max_intensity':4}):
            with self.subTest(cfg=cfg),self.assertRaises(ValueError):boundary_settings(cfg)


if __name__=='__main__':unittest.main()
