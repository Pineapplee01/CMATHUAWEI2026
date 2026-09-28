"""关键回归：不可见值隔离、空观测有限、逐槽有效、梯度与检查点一致。"""
import unittest
import tempfile
import torch
import numpy as np
from torch.nn import functional as F
from problem2.model import AlignedEmotionModel, objective
from problem2.output_contract import project_intensity


class AlignedContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(8)
        self.b={k:torch.randn(3,50,w) for k,w in zip(('XT','XA','XV'),(768,74,35))}
        self.b.update(P=torch.ones(3,3,50,dtype=torch.bool),O=torch.ones(3,3,50,dtype=torch.bool),
                      c=torch.tensor([0,1,2]),y=torch.tensor([-1.,0.,1.]))
        self.b['O'][:,:,10:20]=False

    def test_hidden_values_do_not_change_predictions(self):
        for architecture in ('pooled','cross_attention'):
            model=AlignedEmotionModel({'architecture':architecture,'hidden':32}).eval()
            first=model(self.b)
            changed={**self.b}
            for m,key in enumerate(('XT','XA','XV')):
                changed[key]=self.b[key].clone();changed[key][~self.b['O'][:,m]]=100000.
            second=model(changed)
            torch.testing.assert_close(first['logits'],second['logits'])
            torch.testing.assert_close(first['intensity'],second['intensity'])

    def test_all_missing_finite_and_bounded(self):
        self.b['O'].zero_();self.b['P'][0].zero_()
        for architecture in ('pooled','cross_attention'):
            out=AlignedEmotionModel({'architecture':architecture,'hidden':32})(self.b)
            self.assertTrue(torch.isfinite(out['logits']).all())
            self.assertTrue(torch.isfinite(out['intensity']).all())
            self.assertTrue((out['intensity'].abs()<=3).all())

    def test_backprop_both_heads_and_restore(self):
        for architecture in ('pooled','cross_attention'):
            cfg={'architecture':architecture,'hidden':32}
            model=AlignedEmotionModel(cfg)
            objective(model(self.b),self.b).backward()
            for name in ('text_class','class_head','regression_head'):
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in getattr(model,name).parameters()))
            model.eval()
            with tempfile.TemporaryDirectory() as d:
                torch.save(model.state_dict(),d+'/w.pt')
                restored=AlignedEmotionModel(cfg).eval();restored.load_state_dict(torch.load(d+'/w.pt',weights_only=True))
                torch.testing.assert_close(model(self.b)['logits'],restored(self.b)['logits'])

    def test_projection(self):
        result=project_intensity(np.array([0,1,2]),np.array([.5,-.2,-.5]))
        np.testing.assert_array_equal(np.sign(result).astype(int)+1,[0,1,2])

    def test_prohibited_dataset_rejected(self):
        from problem2.data import load_split
        with self.assertRaises(ValueError):load_split('processed_sentiment','train')
        with self.assertRaises(ValueError):load_split('processed_deberta','train')

    def test_npz_padding_semantics_and_no_rescaling(self):
        from pathlib import Path
        from problem2.data import load_split
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('processed','processed_po'):
                directory=Path(tmp)/name;directory.mkdir()
                I=np.zeros((1,50),dtype=np.int64);I[0,:4]=[101,2000,2020,102]
                q=I!=0;content=q&~np.isin(I,[101,102])
                o=np.repeat(content[:,None],3,1)
                p=np.repeat(q[:,None],3,1);p[:,0]=content
                payload={'XT':np.ones((1,50,768),np.float32)*2.5,
                         'XA':np.ones((1,50,74),np.float32)*-3.5,
                         'XV':np.ones((1,50,35),np.float32)*.75,
                         'I':I,'Q':q,'id':np.array(['a']),
                         'regression_labels':np.array([1.]),'classification_labels':np.array([2])}
                if name=='processed':payload.update(P=~q,mT=content,mA=content,mV=content)
                else:payload.update(P=p,O=o)
                np.savez(directory/'train.npz',**payload)
                result=load_split(directory,'train')
                np.testing.assert_array_equal(result['P'].numpy(),p)
                np.testing.assert_array_equal(result['O'].numpy(),o)
                for key in ('XT','XA','XV'):
                    np.testing.assert_array_equal(result[key].numpy(),payload[key])


    def test_float64_forward_backward_and_state(self):
        from problem2.model import objective
        model=AlignedEmotionModel({'architecture':'cross_attention','hidden':32}).double()
        data={k:v.double() if torch.is_tensor(v) and v.is_floating_point() else v for k,v in self.b.items()}
        out=model(data);loss=objective(out,data);loss.backward()
        self.assertEqual(out['logits'].dtype,torch.float64)
        self.assertEqual(out['intensity'].dtype,torch.float64)
        self.assertEqual(loss.dtype,torch.float64)
        for parameter in model.parameters():
            self.assertEqual(parameter.dtype,torch.float64)
            if parameter.grad is not None:
                self.assertEqual(parameter.grad.dtype,torch.float64)
                self.assertTrue(torch.isfinite(parameter.grad).all())


class NeutralWeightContracts(unittest.TestCase):
    """检验训练权重改变学习信号，同时保留既有损失和回归约束。"""

    def test_default_preserves_original_loss_and_gradients(self):
        # 这是加权功能加入前的损失契约，不复刻新的样本加权算法。
        logits = torch.tensor([[.7, -.2, .1], [.3, .8, -.6], [-.4, .2, 1.2]],
                              dtype=torch.float32, requires_grad=True)
        intensity = torch.tensor([-.2, .1, .7], dtype=torch.float32, requires_grad=True)
        batch = {'c': torch.tensor([0, 1, 2]),
                 'y': torch.tensor([-1.2, 0., 1.8], dtype=torch.float32)}
        output = {'logits': logits, 'intensity': intensity}
        target_cdf = F.one_hot(batch['c'], 3).to(logits.dtype).cumsum(-1)[:, :2]
        original = (F.cross_entropy(logits, batch['c'], label_smoothing=.03)
                    + .2 * F.huber_loss(intensity, batch['y'])
                    + .1 * (logits.softmax(-1).cumsum(-1)[:, :2]
                            - target_cdf).square().mean())
        original_gradients = torch.autograd.grad(original, (logits, intensity))
        for options in ({}, {'neutral_weight': 1.0}):
            current = objective(output, batch, **options)
            gradients = torch.autograd.grad(current, (logits, intensity))
            torch.testing.assert_close(current, original, rtol=0, atol=0)
            for actual, expected in zip(gradients, original_gradients):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_weight_changes_relative_class_gradients_not_regression(self):
        # 同样的不确定预测和每类一个样本，排除难度、频次造成的梯度差异。
        batch = {'c': torch.tensor([0, 1, 2]),
                 'y': torch.tensor([-1., 0., 1.], dtype=torch.float32)}
        previous_norms = previous_regression = None
        for alpha in (1., 2., 5.):
            logits = torch.zeros(3, 3, dtype=torch.float32, requires_grad=True)
            intensity = torch.tensor([-.5, .2, .3], dtype=torch.float32, requires_grad=True)
            loss = objective({'logits': logits, 'intensity': intensity}, batch,
                             neutral_weight=alpha)
            class_gradients, regression_gradients = torch.autograd.grad(loss, (logits, intensity))
            norms = class_gradients.norm(dim=-1)
            self.assertLess(class_gradients[1, 1].item(), 0)
            if previous_norms is not None:
                self.assertGreater(norms[1].item(), previous_norms[1].item())
                self.assertTrue((norms[[0, 2]] < previous_norms[[0, 2]]).all())
                self.assertGreater((norms[1] / norms[[0, 2]].mean()).item(),
                                   (previous_norms[1] / previous_norms[[0, 2]].mean()).item())
                torch.testing.assert_close(regression_gradients, previous_regression,
                                           rtol=0, atol=0)
            previous_norms, previous_regression = norms, regression_gradients

    def test_single_class_batch_retains_loss_scale(self):
        for label in (0, 1, 2):
            logits = torch.tensor([[.4, -.2, .1], [-.3, .7, .5]],
                                  dtype=torch.float32, requires_grad=True)
            intensity = torch.tensor([-.3, .6], dtype=torch.float32, requires_grad=True)
            batch = {'c': torch.full((2,), label, dtype=torch.long),
                     'y': torch.full((2,), float(label - 1), dtype=torch.float32)}
            output = {'logits': logits, 'intensity': intensity}
            baseline = objective(output, batch)
            baseline_gradients = torch.autograd.grad(baseline, (logits, intensity))
            for alpha in (.25, 4.):
                with self.subTest(label=label, neutral_weight=alpha):
                    weighted = objective(output, batch, neutral_weight=alpha)
                    gradients = torch.autograd.grad(weighted, (logits, intensity))
                    torch.testing.assert_close(weighted, baseline, rtol=1e-6, atol=1e-7)
                    for actual, expected in zip(gradients, baseline_gradients):
                        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_invalid_neutral_weight_rejected(self):
        output = {'logits': torch.zeros(3, 3), 'intensity': torch.zeros(3)}
        batch = {'c': torch.tensor([0, 1, 2]), 'y': torch.zeros(3)}
        for alpha in (0., -1., float('nan'), float('inf'), -float('inf')):
            with self.subTest(neutral_weight=alpha):
                with self.assertRaises(ValueError):
                    objective(output, batch, neutral_weight=alpha)


if __name__=='__main__':unittest.main()
