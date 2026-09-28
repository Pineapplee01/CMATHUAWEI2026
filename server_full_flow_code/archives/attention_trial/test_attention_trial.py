"""注意力试验：对齐、填充隔离、完整观测梯度与账本。"""
import unittest
from unittest.mock import patch
import torch
from problem2.token_alignment import retained_text,overlap_map
from problem2.attention_readout import AttentionPool
from problem2.model import EmotionModel
from tests import test_model_contracts as contracts

class AttentionTrialTests(unittest.TestCase):
    def test_wordpieces_offsets(self):
        text,spans=retained_text(['i','don',"'",'t','like','play','##ing','.'])
        self.assertEqual(text,"i don't like playing.")
        weights=overlap_map(spans,[(0,0),(0,1),(2,7),(8,12),(13,20),(20,21),(0,0)],'cpu')
        torch.testing.assert_close(weights.sum(-1),torch.ones(8))
        self.assertEqual(float(weights[:,0].sum()),0)

    def test_padding_and_empty_pool(self):
        pool=AttentionPool(8,5)
        h=torch.randn(2,3,50,8);P=torch.zeros(2,3,50,dtype=torch.bool);P[0,:,:3]=True
        a=pool(h,P,P,P.float());h[~P]=10000
        b=pool(h,P,P,P.float());torch.testing.assert_close(a,b)
        self.assertTrue(torch.isfinite(a).all());self.assertEqual(float(a[1].abs().sum()),0.)

    def test_full_observations_train_fusion_and_reconstruct(self):
        case=contracts.ModelContracts();case.setUp()
        cfg=dict(case.cfg,attention_readout=True,gated_fusion=True)
        stats={n:{'mean':[0.]*d,'std':[1.]*d} for n,d in [('audio',74),('vision',35)]}
        with patch('problem2.model.TextEncoder',contracts.FakeText):model=EmotionModel(cfg,stats)
        out=model(case.batch)
        ledger=out['ledger'];score=ledger['bias']+ledger['single'].sum((1,2))+ledger['pair'].sum((1,2))
        torch.testing.assert_close(out['logits'],score[:,:3])
        out['logits'].square().sum().backward()
        self.assertGreater(float(model.layers[0].cross.in_proj_weight.grad.abs().sum()),0.)
        self.assertGreater(float(model.fusion_gate.grad.abs().sum()),0.)

if __name__=='__main__':unittest.main()
