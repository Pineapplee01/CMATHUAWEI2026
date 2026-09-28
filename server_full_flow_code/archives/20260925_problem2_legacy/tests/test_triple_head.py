"""低秩三模态残差的热启动等价性及账本重建。"""
import unittest
import torch
from problem2.heads import AdditiveHead


class TripleHeadContracts(unittest.TestCase):
    def test_zero_initialization_preserves_previous_scores(self):
        old=AdditiveHead(8,5,True);new=AdditiveHead(8,5,True,2)
        new.load_state_dict(old.state_dict(),strict=False)
        values=torch.randn(2,3,10,11);values[...,-3]=1;values[...,-2]=0
        torch.testing.assert_close(old(values)[0],new(values)[0])

    def test_triple_is_in_reconstruction(self):
        head=AdditiveHead(8,5,True,2)
        with torch.no_grad():head.triple_out.weight.fill_(.2)
        values=torch.randn(2,3,10,11);values[...,-3]=1;values[...,-2]=0
        score,ledger=head(values)
        rebuilt=ledger['bias']+ledger['single'].sum((1,2))+ledger['pair'].sum((1,2))+ledger['triple'].sum(1)
        torch.testing.assert_close(score,rebuilt)
        self.assertGreater(float(ledger['triple'].abs().sum()),0.)


if __name__=='__main__':unittest.main()
