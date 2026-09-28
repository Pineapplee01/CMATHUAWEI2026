"""新增三分类先验路径的无泄漏与数值账本检查。"""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from torch import nn
from problem2.sentiment_encoder import SentimentEncoder
from tests import test_model_contracts as contracts


class PriorContracts(unittest.TestCase):
    def test_prior_is_explicit_and_empty_text_gives_zero(self):
        case=contracts.ModelContracts();case.setUp();model=case.model
        model.cfg['sentiment_prior']=True
        out=model(case.batch);ledger=out['ledger']
        reconstructed=ledger['bias']+ledger['single'].sum((1,2))+ledger['pair'].sum((1,2))+ledger['text_prior']
        torch.testing.assert_close(reconstructed[:,:3],out['logits'])
        keep=torch.ones_like(case.batch['O']);keep[:,0]=False
        missing=model(case.batch,keep)
        self.assertEqual(float(missing['ledger']['text_prior'].abs().sum()),0.)

    def test_remove_before_retokenization(self):
        class Source:
            def batch_decode(self,rows,**kwargs):
                self.rows=rows
                return [' '.join(map(str,row)) for row in rows]
        class Target:
            def __call__(self,rows,**kwargs):
                return {'input_ids':torch.tensor([[len(s)] for s in rows])}
        class Backbone(nn.Module):
            def __init__(self):
                super().__init__();self.config=SimpleNamespace(id2label={0:'negative',1:'neutral',2:'positive'},max_position_embeddings=514)
            def forward(self,input_ids,**kwargs):
                x=input_ids.float()
                return SimpleNamespace(logits=x.expand(-1,3),hidden_states=[x[:,:,None].expand(-1,-1,768)])
        source=Source()
        with patch('problem2.sentiment_encoder.AutoModelForSequenceClassification.from_pretrained',return_value=Backbone()), \
             patch('problem2.sentiment_encoder.AutoTokenizer.from_pretrained',side_effect=[source,Target()]):
            encoder=SentimentEncoder('unused','unused')
        tokens=torch.zeros(1,3,50,dtype=torch.long);tokens[0,0,:4]=torch.tensor([101,111,999,102])
        observed=torch.zeros(1,50,dtype=torch.bool);observed[0,1]=True
        with torch.no_grad():first=encoder(tokens,observed)
        self.assertEqual(source.rows,[[111]])
        tokens[0,0,2]=888
        with torch.no_grad():second=encoder(tokens,observed)
        torch.testing.assert_close(first,second)
        self.assertEqual(source.rows,[[111]])


if __name__=='__main__':unittest.main()
