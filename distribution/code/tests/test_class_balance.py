"""类别平衡的数学性质及真实训练/检查点连接。"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from problem2.model import class_balance_metadata, objective
from problem2 import training


class ClassBalanceTests(unittest.TestCase):
    def test_inverse_frequency_equalizes_class_weight_totals(self):
        labels=torch.repeat_interleave(torch.arange(3),torch.tensor([967,758,1670]))
        meta=class_balance_metadata(labels)
        self.assertEqual(meta['train_class_counts'],[967,758,1670])
        np.testing.assert_allclose(meta['class_weights'],[1.1702861082,1.4929639402,.6776447106])
        np.testing.assert_allclose(np.array(meta['class_weights'])*meta['train_class_counts'],3395/3)

    def test_tempering_is_between_unweighted_and_balanced(self):
        labels=[0,0,1,2,2,2,2]
        zero=class_balance_metadata(labels,0.)
        half=class_balance_metadata(labels,.5)
        full=class_balance_metadata(labels,1.)
        self.assertEqual(zero['class_weights'],[1.,1.,1.])
        for weight in (half,full):
            self.assertAlmostEqual(sum(n*w for n,w in zip(weight['train_class_counts'],weight['class_weights'])),len(labels))
        self.assertLess(half['class_weights'][1],full['class_weights'][1])
        self.assertGreater(half['class_weights'][2],full['class_weights'][2])

    def test_extra_neutral_multiplier_applied_once(self):
        plain=class_balance_metadata([0,1,2,2],1.,1.)['class_weights']
        extra=class_balance_metadata([0,1,2,2],1.,1.5)['class_weights']
        self.assertAlmostEqual(extra[1]/extra[0],1.5*plain[1]/plain[0])
        self.assertAlmostEqual(extra[2]/extra[0],plain[2]/plain[0])
        out={'logits':torch.zeros(3,3),'intensity':torch.zeros(3)}
        b={'c':torch.tensor([0,1,2]),'y':torch.zeros(3)}
        with self.assertRaisesRegex(ValueError,'重复'):
            objective(out,b,1.5,class_weights=extra)

    def test_invalid_labels_and_weights_rejected(self):
        for labels in ([],[0,0,2],[0,1,3],[0.,1.,2.],[[0,1,2]]):
            with self.subTest(labels=labels),self.assertRaises(ValueError):
                class_balance_metadata(labels)
        for beta in (-1.,1.01,float('nan'),float('inf')):
            with self.assertRaises(ValueError):class_balance_metadata([0,1,2],beta)
        out={'logits':torch.zeros(3,3),'intensity':torch.zeros(3)}
        b={'c':torch.tensor([0,1,2]),'y':torch.zeros(3)}
        for weights in ([1,1],[1,0,1],[1,float('nan'),1]):
            with self.assertRaises(ValueError):objective(out,b,class_weights=weights)

    def test_disabled_balance_preserves_loss_and_gradients(self):
        torch.manual_seed(22)
        logits=torch.randn(4,3,requires_grad=True)
        intensity=torch.randn(4,requires_grad=True)
        b={'c':torch.tensor([0,1,2,2]),'y':torch.tensor([-1.,0.,1.,2.])}
        output={'logits':logits,'intensity':intensity}
        original=objective(output,b)
        disabled=objective(output,b,class_weights=class_balance_metadata(b['c'],0.)['class_weights'])
        torch.testing.assert_close(original,disabled,rtol=0,atol=0)
        a=torch.autograd.grad(original,(logits,intensity),retain_graph=True)
        z=torch.autograd.grad(disabled,(logits,intensity))
        for expected,actual in zip(a,z):torch.testing.assert_close(expected,actual,rtol=0,atol=0)

    def test_training_uses_only_train_counts_for_full_and_missing(self):
        torch.set_num_threads(2)
        torch.manual_seed(12)
        def data(labels):
            n=len(labels)
            b={key:torch.randn(n,50,width) for key,width in zip(('XT','XA','XV'),(768,74,35))}
            b.update(c=torch.tensor(labels),y=torch.tensor(labels,dtype=torch.float32)-1,
                     P=torch.ones(n,3,50,dtype=torch.bool),O=torch.ones(n,3,50,dtype=torch.bool),
                     id=[str(i) for i in range(n)])
            return b
        train=data([0,0,1,2,2,2]);valid=data([1,1,1])
        train['y']=torch.tensor([-1/3,-1.,0.,1/3,1.,2.])
        partial={**train,'O':train['O'].clone()};partial['O'][:,1,10:20]=False
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'processed_po';directory.mkdir()
            for name in ('train','valid','scaler_params'):
                np.savez(directory/(name+'.npz'),fixture=np.array([1]))
            cfg={'architecture':'problem2','hidden':8,'seed':7,'device':'cpu','epochs':1,
                 'patience':1,'batch_size':6,'lr':.0005,'name':'balance_test',
                 'data_dir':str(directory),'output':'run','checkpoint':'weights.pt',
                 'balance_beta':1.,'neutral_weight':1.,'boundary_weight':.1,'boundary_warmup_epochs':3}
            class TaskStub(torch.nn.Module):
                def __init__(self):
                    super().__init__();self.head=torch.nn.Linear(74,4)
                def forward(self,b):
                    z=self.head(b['XA'].mean(1))
                    return {'logits':z[:,:3],'intensity':3*z[:,3].tanh()}
            with patch.object(training,'build_model',side_effect=lambda *a,**k:TaskStub()), \
                 patch.object(training,'path',side_effect=lambda value:root/Path(value)), \
                 patch.object(training,'objective',wraps=objective) as observed, \
                 contextlib.redirect_stdout(io.StringIO()):
                result=training.train_candidate(cfg,train,valid,[partial],valid)
                self.assertEqual(observed.call_count,2)
                for call in observed.call_args_list:
                    np.testing.assert_allclose(call.kwargs['class_weights'],[1.,2.,2/3])
                self.assertAlmostEqual(observed.call_args_list[0].kwargs['boundary_weight'],.1/3)
                self.assertEqual(observed.call_args_list[1].kwargs.get('boundary_weight',0.),0.)
                model,restored=training.restore('weights.pt','cpu')
            self.assertEqual(result['train_class_counts'],[2,1,3])
            self.assertEqual(restored['class_weights'],result['class_weights'])
            saved=json.loads((root/'run/history.json').read_text())[0]
            self.assertEqual(saved['train_class_counts'],[2,1,3])
            self.assertEqual(saved['class_weights'],result['class_weights'])
            self.assertAlmostEqual(saved['effective_boundary_weight'],.1/3)
            with torch.no_grad():self.assertTrue(torch.isfinite(model(valid)['logits']).all())


if __name__=='__main__':unittest.main()
