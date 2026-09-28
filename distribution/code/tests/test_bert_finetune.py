"""在线文本微调的梯度、缺失隔离和权重重载契约。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from transformers import BertConfig, BertModel
from problem2.model import build_model, objective


class BertFinetuneTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(7)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        np.savez(Path(self.temp.name)/'scaler_params.npz',mu_T=np.zeros(768),sigma_T=np.ones(768))
        self.cfg={'finetune_bert':True,'architecture':'problem2','hidden':8,'data_dir':self.temp.name}
        self.bc=BertConfig(vocab_size=200,hidden_size=768,num_hidden_layers=1,
                           num_attention_heads=4,intermediate_size=32,max_position_embeddings=64,
                           hidden_dropout_prob=0.,attention_probs_dropout_prob=0.)
        self.b={k:torch.randn(2,50,w) for k,w in [('XT',768),('XA',74),('XV',35)]}
        self.b.update(I=torch.tensor([[101,15,16,17,102]+[0]*45]*2),
                      O=torch.zeros(2,3,50,dtype=torch.bool),P=torch.zeros(2,3,50,dtype=torch.bool),
                      c=torch.tensor([1,2]),y=torch.tensor([0.,1/3]))
        self.b['O'][:,:,1:4]=True;self.b['P'][:,:,1:4]=True

    def model(self):
        with patch('transformers.AutoModel.from_pretrained',return_value=BertModel(self.bc)):
            return build_model(self.cfg)

    def test_task_gradient_updates_bert(self):
        model=self.model().eval()
        weight=model.bert.embeddings.word_embeddings.weight
        before=weight.detach().clone()
        loss=objective(model(self.b),self.b,boundary_weight=.1)
        loss.backward()
        self.assertGreater(float(weight.grad[15].abs().sum()),0.)
        torch.optim.SGD(model.parameters(),lr=.01).step()
        self.assertFalse(torch.equal(before,weight))

    def test_hidden_tokens_and_cached_xt_do_not_leak(self):
        model=self.model().eval();self.b['O'][:,0,2]=False
        changed={k:v.clone() for k,v in self.b.items()}
        changed['I'][:,2]=99;changed['XT'].normal_(100,50)
        with torch.no_grad():
            first=model(self.b);second=model(changed)
        torch.testing.assert_close(first['logits'],second['logits'],rtol=0,atol=0)
        torch.testing.assert_close(first['intensity'],second['intensity'],rtol=0,atol=0)

    def test_partial_finetuning_explicitly_rejected(self):
        self.bc.num_hidden_layers=2;self.cfg['bert_trainable_layers']=1
        with self.assertRaises(ValueError):self.model()

    def test_reload_and_empty_text(self):
        model=self.model().eval();self.b['O'][:,0]=False;self.b['I'].zero_()
        with patch('transformers.AutoConfig.from_pretrained',return_value=self.bc):
            restored=build_model(self.cfg,pretrained=False).eval()
        restored.load_state_dict(model.state_dict(),strict=True)
        with torch.no_grad():
            first=model(self.b);second=restored(self.b)
        self.assertTrue(torch.isfinite(first['logits']).all())
        torch.testing.assert_close(first['logits'],second['logits'],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
