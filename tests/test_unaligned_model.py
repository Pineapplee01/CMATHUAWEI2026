"""原生50/500/500软对齐：远端帧连通、缺失隔离、空输入和检查点重载。"""
import unittest
from unittest.mock import patch
import torch
from transformers import BertModel
import tests.test_bert_finetune as fixtures
from problem2.model import build_model,objective
from problem2.data import masked_view


class UnalignedModelTests(unittest.TestCase):
    def setUp(self):
        fixtures.BertFinetuneTests.setUp(self)
        self.cfg.update(architecture='soft_alignment')
        self.b.pop('P');self.b.pop('O');self.b['id']=['a','b']
        for m,key,length,width in zip('TAV',('XT','XA','XV'),(50,500,500),(768,74,35)):
            self.b[key]=torch.randn(2,length,width)
            self.b['P_'+m]=torch.ones(2,length,dtype=torch.bool)
            self.b['O_'+m]=torch.ones(2,length,dtype=torch.bool)
        self.b['P_T'][:,0]=False;self.b['P_T'][:,4:]=False;self.b['O_T']&=self.b['P_T']
    def make(self):
        with patch('transformers.AutoModel.from_pretrained',return_value=BertModel(self.bc)):
            return build_model(self.cfg).eval()
    def test_all_500_slots_connect_and_attention_shapes(self):
        model=self.make();self.b['XA'].requires_grad_();self.b['XV'].requires_grad_()
        out=model(self.b,return_attention=True)
        for name in ('attention_audio','attention_vision'):
            self.assertEqual(out[name].shape,(2,4,50,500))
            torch.testing.assert_close(out[name][:,:,1:4].sum(-1),torch.ones(2,4,3))
        objective(out,self.b).backward()
        for key in ('XA','XV'):
            self.assertGreater(float(self.b[key].grad[:,499].abs().sum()),0.)
        self.assertGreater(float(model.bert.embeddings.word_embeddings.weight.grad[15].abs().sum()),0.)
    def test_masked_content_cannot_leak(self):
        model=self.make();self.b['O_T'][:,2]=False;self.b['O_A'][:,100:200]=False;self.b['O_V'][:,200:300]=False
        changed={k:v.clone() if torch.is_tensor(v) else v for k,v in self.b.items()}
        changed['I'][:,2]=99;changed['XA'][:,100:200]=10000;changed['XV'][:,200:300]=-10000
        with torch.no_grad():a=model(self.b,return_attention=True);b=model(changed,return_attention=True)
        torch.testing.assert_close(a['logits'],b['logits'],rtol=0,atol=0)
        self.assertEqual(float(a['attention_audio'][...,100:200].abs().sum()),0.)
        self.assertEqual(float(a['attention_vision'][...,200:300].abs().sum()),0.)
    def test_native_contiguous_masks_retain_axes_and_observations(self):
        for position in ('start','middle','end','random'):
            view,records=masked_view(self.b,None,.7,(0,1,2),position,2026,reencode=False)
            for m,length in zip('TAV',(50,500,500)):
                self.assertEqual(view['O_'+m].shape,(2,length));self.assertTrue(view['O_'+m].any(-1).all())
            self.assertTrue(all(r['observed_after']>0 for r in records))
            self.assertTrue(any(r['end_slot_exclusive']>50 for r in records if r['modality']!='T'))
    def test_empty_modalities_finite_and_strict_reload(self):
        model=self.make()
        with patch('transformers.AutoConfig.from_pretrained',return_value=self.bc):restored=build_model(self.cfg,pretrained=False).eval()
        restored.load_state_dict(model.state_dict(),strict=True)
        for m in 'TAV':self.b['P_'+m].zero_();self.b['O_'+m].zero_()
        self.b['I'].zero_()
        with torch.no_grad():a=model(self.b);b=restored(self.b)
        self.assertTrue(torch.isfinite(a['logits']).all());torch.testing.assert_close(a['logits'],b['logits'])


if __name__=='__main__':unittest.main()
