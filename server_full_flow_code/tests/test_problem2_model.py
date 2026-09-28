"""问题二正式模型与六组研究消融：输入隔离、梯度、组合开关及连续缺失契约。"""
import unittest
from unittest.mock import patch
import torch
from transformers import BertModel
import tests.test_bert_finetune as fixtures
from problem2.model import build_model,objective,ablation_regularizers
from problem2.local_missingness import synchronized_view
from problem2.ablation import VARIANTS


class Problem2ModelTests(unittest.TestCase):
    def setUp(self):
        fixtures.BertFinetuneTests.setUp(self)
        self.b['id']=['a','b']
    def make(self,cfg=None):
        with patch('transformers.AutoModel.from_pretrained',return_value=BertModel(self.bc)):
            return build_model(cfg or self.cfg)
    def test_official_preserves_observed_and_hides_missing_tokens(self):
        model=self.make().eval();self.b['O'][:,0,2]=False
        a=model(self.b);changed={k:v.clone() if torch.is_tensor(v) else v for k,v in self.b.items()}
        changed['I'][:,2]=99;changed['XA'][~self.b['O'][:,1]]=1e5;changed['XT'].fill_(1e5)
        b=model(changed);M=self.b['O']&self.b['P']
        torch.testing.assert_close(a['logits'],b['logits'],rtol=0,atol=0)
        torch.testing.assert_close(a['completed'][M],a['observed_embedding'][M],rtol=0,atol=0)
        objective(a,self.b).backward()
        for p in (model.mean[0].weight,model.logvar[0].weight,model.head.mlp[-1].weight):
            self.assertGreater(float(p.grad.abs().sum()),0)
    def test_ablation_shared_initialization_and_switches(self):
        models={}
        for v,(g,c,r) in VARIANTS.items():
            torch.manual_seed(9)
            models[v]=self.make(self.cfg|{'architecture':'ablation','ablation_gate':g,'ablation_reconstruction':r})
        for v,m in models.items():
            for key,value in models['Bm'].state_dict().items():torch.testing.assert_close(value,m.state_dict()[key],rtol=0,atol=0)
        missing,_=synchronized_view(self.b,.5,(0,1,2),'middle',1)
        for v,m in models.items():
            m.eval();out=m(missing)
            if v in ('Bm','C','G'):self.assertEqual(float(out['reconstruction'].abs().sum()),0.)
            if v in ('Bm','C','R'):torch.testing.assert_close(out['gates'],self.b['P'].float())
            self.assertTrue(torch.isfinite(out['logits']).all())
    def test_regularizers_only_removed_observed_slots_and_detached_teacher(self):
        cfg=self.cfg|{'architecture':'ablation','ablation_gate':True,'ablation_reconstruction':True}
        model=self.make(cfg).eval();partial,_=synchronized_view(self.b,.5,(0,1,2),'middle',1)
        teacher=model(self.b);teacher['logits'].retain_grad();teacher['observed_embedding'].retain_grad()
        student=model(partial);lc,lr=ablation_regularizers(student,teacher,self.b,partial)
        (lc+lr).backward()
        self.assertIsNone(teacher['logits'].grad);self.assertIsNone(teacher['observed_embedding'].grad)
        self.assertGreater(float(model.mean[0].weight.grad.abs().sum()),0.)
        self.assertGreater(float(model.gate[-1].weight.grad.abs().sum()),0.)
        clean=student.copy();clean['reconstruction']=student['reconstruction'].detach().clone()
        removed=self.b['O']&~partial['O'];clean['reconstruction'][~removed]=1e6
        _,lr2=ablation_regularizers(clean,teacher,self.b,partial)
        torch.testing.assert_close(lr,lr2)
    def test_synchronized_masks_keep_nonempty_modalities_and_same_intervals(self):
        for position in ('start','middle','end','random'):
            masked,records=synchronized_view(self.b,.9,(0,1,2),position,1)
            self.assertTrue((masked['O'].sum(-1)>=1).all())
            for sid in self.b['id']:
                intervals={(r['start_slot'],r['end_slot_exclusive']) for r in records if r['id']==sid}
                self.assertEqual(len(intervals),1)
            again,_=synchronized_view(self.b,.9,(0,1,2),position,1)
            torch.testing.assert_close(masked['O'],again['O'])
    def test_ablation_reload_and_empty_views(self):
        cfg=self.cfg|{'architecture':'ablation','ablation_gate':True,'ablation_reconstruction':True}
        m=self.make(cfg).eval()
        with patch('transformers.AutoConfig.from_pretrained',return_value=self.bc):other=build_model(cfg,pretrained=False).eval()
        other.load_state_dict(m.state_dict(),strict=True)
        self.b['O'].zero_();self.b['P'].zero_();self.b['I'].zero_()
        a=m(self.b);b=other(self.b)
        self.assertTrue(torch.isfinite(a['logits']).all())
        torch.testing.assert_close(a['logits'],b['logits'],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
