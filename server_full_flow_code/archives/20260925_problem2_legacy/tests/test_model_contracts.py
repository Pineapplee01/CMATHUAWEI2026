"""待用户允许运行后执行：python -m unittest discover -s tests -v。
不依赖真实数据/权重；检验缺失泄漏、全空证据和账本结构。
"""
import unittest
from unittest.mock import patch
import torch
from torch import nn
from problem2.model import EmotionModel
from problem2.losses import objective


class FakeText(nn.Module):
    def __init__(self, _): super().__init__()
    def forward(self, tokens, observed):
        # 模拟上下文编码；编码前是否遮蔽由真实 TextEncoder 的独立测试验证。
        visible = tokens[:,0].float() * observed
        x = visible + visible.sum(1,keepdim=True)
        return x[...,None].expand(-1,-1,768) * observed[...,None]


class ModelContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.cfg = {'hidden':8,'heads':2,'layers':1,'dropout':0.,'window':5,
                    'tau':[1.,1.,1.],'ablation':'main','bert_path':'unused',
                    'lambda_rec':.1,'lambda_miss':1.,'lambda_con':0.}
        stats = {n:{'mean':[0.]*d,'std':[1.]*d} for n,d in [('audio',74),('vision',35)]}
        with patch('problem2.model.TextEncoder', FakeText):
            self.model = EmotionModel(self.cfg,stats).eval()
        self.batch = {'tokens':torch.ones(2,3,50,dtype=torch.long),
                      'audio':torch.randn(2,50,74),'vision':torch.randn(2,50,35),
                      'P':torch.ones(2,3,50,dtype=torch.bool),
                      'O':torch.ones(2,3,50,dtype=torch.bool),
                      'y':torch.tensor([-.5,.5]),'c':torch.tensor([0,2])}

    def test_missing_values_do_not_leak(self):
        keep = torch.ones_like(self.batch['O']); keep[:,:,10:20]=False
        a = self.model(self.batch,keep)['logits']
        changed = {k:v.clone() for k,v in self.batch.items()}
        changed['tokens'][:,0,10:20]=999
        changed['audio'][:,10:20]=9999; changed['vision'][:,10:20]=-9999
        b = self.model(changed,keep)['logits']
        torch.testing.assert_close(a,b)

    def test_empty_and_ledger(self):
        out = self.model(self.batch,torch.zeros_like(self.batch['O']))
        self.assertTrue(torch.isfinite(out['logits']).all())
        self.assertTrue(out['low_evidence'].all())
        ledger = out['ledger']
        score = ledger['bias']+ledger['single'].sum((1,2))+ledger['pair'].sum((1,2))
        torch.testing.assert_close(score[:,:3],out['logits'])
        torch.testing.assert_close(3*score[:,3].tanh(),out['intensity'])

    def test_task_gradient_reaches_completion(self):
        keep = torch.ones_like(self.batch['O']); keep[:,1,10:20]=False
        out = self.model(self.batch,keep)
        out['logits'].square().sum().backward()
        self.assertGreater(float(self.model.mean[1].weight.grad.abs().sum()),0)

    def test_padding_not_reconstructed(self):
        self.batch['P'][:,:,40:]=False
        self.batch['O'][:,:,40:]=False
        full = self.model(self.batch)
        keep = torch.ones_like(self.batch['O']); keep[:,:,40:]=False
        missing = self.model(self.batch,keep)
        _, log = objective(full,missing,self.batch,self.cfg,None)
        self.assertEqual(log['reconstruction_nll'],0.)


class EncoderMasking(unittest.TestCase):
    def test_real_encoder_removes_ids_before_backbone(self):
        from types import SimpleNamespace
        from problem2.model import TextEncoder
        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=768)
                self.seen = None
            def forward(self, **inputs):
                self.seen = inputs
                n,t = inputs['input_ids'].shape
                return SimpleNamespace(last_hidden_state=torch.ones(n,t,768))
        backbone = Backbone()
        with patch('problem2.text_encoder.AutoModel.from_pretrained', return_value=backbone):
            encoder = TextEncoder('unused')
        tokens = torch.zeros(1,3,50,dtype=torch.long)
        tokens[0,0,:5] = torch.tensor([101,2023,2003,2204,102])
        tokens[0,1,:5] = 1
        observed = torch.zeros(1,50,dtype=torch.bool); observed[0,1:3]=True
        encoder(tokens,observed)
        self.assertEqual(int(backbone.seen['input_ids'][0,3]),0)
        self.assertEqual(int(backbone.seen['attention_mask'][0,3]),0)
        self.assertEqual(int(backbone.seen['input_ids'][0,0]),101)


if __name__ == '__main__':
    unittest.main()
