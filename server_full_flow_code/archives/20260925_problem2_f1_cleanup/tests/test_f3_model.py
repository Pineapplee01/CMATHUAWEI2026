"""F3结构契约：补全观测保留、严格跨模态读取、缺失隔离与梯度。"""
import unittest
from unittest.mock import patch
import torch
from transformers import BertModel
import tests.test_bert_finetune as fixtures
from problem2.model import build_model,objective
from problem2.model_f3 import CrossModalFusion


class F3Tests(unittest.TestCase):
    def setUp(self):
        fixtures.BertFinetuneTests.setUp(self)
        self.cfg.update(architecture='f3',f3_cross=True,f3_head='concat')

    def make(self):
        with patch('transformers.AutoModel.from_pretrained',return_value=BertModel(self.bc)):
            return build_model(self.cfg)

    def test_cross_reads_only_other_modalities(self):
        fusion=CrossModalFusion(8,2,0).eval();h=torch.randn(2,3,50,8);P=torch.ones(2,3,50,dtype=torch.bool)
        recorded=[]
        for module in fusion.attn:module.register_forward_pre_hook(lambda m,args:recorded.append(args))
        fusion(h,P)
        for m,(q,k,v) in enumerate(recorded):
            torch.testing.assert_close(q,h[:,m])
            expected=torch.cat([h[:,j] for j in range(3) if j!=m],1)
            torch.testing.assert_close(k[:,:100],expected);torch.testing.assert_close(k,v)
        torch.testing.assert_close(fusion.alpha,torch.full((3,),.1))

    def test_observed_slots_preserved_and_missing_tokens_hidden(self):
        model=self.make().eval();self.b['O'][:,0,2]=False
        changed={k:v.clone() for k,v in self.b.items()}
        changed['I'][:,2]=99;changed['XT'].fill_(999)
        with torch.no_grad():a=model(self.b);b=model(changed)
        M=self.b['O']&self.b['P']
        torch.testing.assert_close(a['completed'][M],a['observed_embedding'][M],rtol=0,atol=0)
        torch.testing.assert_close(a['q'][M],torch.ones_like(a['q'][M]))
        torch.testing.assert_close(a['logits'],b['logits'],rtol=0,atol=0)
        self.assertEqual(a['pooled'].shape,(2,3,10,11))

    def test_tasks_train_bert_completion_and_fusion(self):
        model=self.make().eval();self.b['O'][:,0,2]=False
        out=model(self.b);objective(out,self.b,boundary_weight=.1).backward()
        for parameter in [model.bert.embeddings.word_embeddings.weight,model.mean[0].weight,
                          model.logvar[0].weight,model.fusion.alpha,model.head.mlp[-1].weight]:
            self.assertIsNotNone(parameter.grad);self.assertGreater(float(parameter.grad.abs().sum()),0.)

    def test_cross_toggle_preserves_shared_initialization(self):
        self.cfg['f3_cross']=False;torch.manual_seed(31);first=self.make()
        self.cfg['f3_cross']=True;torch.manual_seed(31);second=self.make()
        other=second.state_dict()
        for key,value in first.state_dict().items():
            torch.testing.assert_close(value,other[key],rtol=0,atol=0)

    def test_empty_inputs_finite_and_reload(self):
        model=self.make().eval();self.b['O'].zero_();self.b['P'].zero_();self.b['I'].zero_()
        with patch('transformers.AutoConfig.from_pretrained',return_value=self.bc):
            restored=build_model(self.cfg,pretrained=False).eval()
        restored.load_state_dict(model.state_dict(),strict=True)
        with torch.no_grad():a=model(self.b);b=restored(self.b)
        self.assertTrue(torch.isfinite(a['logits']).all());self.assertTrue(torch.isfinite(a['intensity']).all())
        torch.testing.assert_close(a['logits'],b['logits'],rtol=0,atol=0)

    def test_special_outputs_receive_gradients_without_missing_text_leak(self):
        self.cfg.update(f3_cross=False,special_tokens='cls_sep')
        model=self.make().eval();self.b['O'][:,0,2]=False
        changed={k:v.clone() for k,v in self.b.items()};changed['I'][:,2]=99
        first=model(self.b);second=model(changed)
        torch.testing.assert_close(first['logits'],second['logits'],rtol=0,atol=0)
        objective(first,self.b).backward()
        grad=model.head.special_linear.weight.grad
        for part in grad.chunk(2,dim=1):self.assertGreater(float(part.abs().sum()),0.)
        self.assertGreater(float(model.special_project[0].weight.grad.abs().sum()),0.)
        with patch('transformers.AutoConfig.from_pretrained',return_value=self.bc):
            restored=build_model(self.cfg,pretrained=False).eval()
        restored.load_state_dict(model.state_dict(),strict=True)
        torch.testing.assert_close(first['logits'],restored(self.b)['logits'],rtol=0,atol=0)

    def test_special_branch_preserves_f1_initialization_and_handles_absent_markers(self):
        self.cfg['f3_cross']=False;torch.manual_seed(31);base=self.make()
        self.cfg['special_tokens']='cls_sep';torch.manual_seed(31);model=self.make().eval()
        for key,value in base.state_dict().items():
            torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
        self.b['I'].zero_();self.b['P'].zero_();self.b['O'].zero_()
        seen=[]
        model.head.special_linear.register_forward_pre_hook(lambda m,args:seen.append(args[0]))
        out=model(self.b)
        self.assertTrue(torch.isfinite(out['logits']).all())
        self.assertEqual(float(seen[0].abs().sum()),0.)


if __name__=='__main__':unittest.main()
