"""SAM扰动半径、冻结参数和异常恢复契约。"""
import unittest
import torch
from problem2.sharpness import sharpness_neighborhood

class SharpnessTests(unittest.TestCase):
    def test_radius_and_exact_restore(self):
        model=torch.nn.Linear(2,1)
        model.bias.requires_grad_(False)
        original=model.weight.detach().clone()
        bias=model.bias.detach().clone()
        model(torch.ones(1,2)).square().sum().backward()
        with sharpness_neighborhood(model,.05):
            self.assertAlmostEqual(float((model.weight-original).norm()),.05,places=6)
            torch.testing.assert_close(model.bias,bias,rtol=0,atol=0)
        torch.testing.assert_close(model.weight,original,rtol=0,atol=0)

    def test_exception_restores_parameters(self):
        model=torch.nn.Linear(2,1)
        model(torch.ones(1,2)).sum().backward()
        before={k:v.clone() for k,v in model.state_dict().items()}
        with self.assertRaises(RuntimeError):
            with sharpness_neighborhood(model,.1):raise RuntimeError('test')
        for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)

class WeightAveragingTests(unittest.TestCase):
    def test_average_keeps_discrete_state_and_weighting(self):
        from problem2.weight_soup import mean_state
        a={'w':torch.tensor([1.,3.]),'n':torch.tensor(4)}
        b={'w':torch.tensor([4.,0.]),'n':torch.tensor(4)}
        r=mean_state(a,b,2)
        torch.testing.assert_close(r['w'],torch.tensor([2.,2.]))
        self.assertEqual(int(r['n']),4)
        with self.assertRaises(ValueError):mean_state(a,dict(b,n=torch.tensor(5)),2)

if __name__=='__main__':unittest.main()
