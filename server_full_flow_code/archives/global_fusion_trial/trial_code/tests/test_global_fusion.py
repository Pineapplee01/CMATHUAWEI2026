"""全局融合的隐藏值隔离、空观测和初始模型等价性。"""
import unittest
import torch
from problem2.global_fusion import GlobalFusion, auxiliary_loss


class GlobalFusionContracts(unittest.TestCase):
    def test_combined_prediction_ledger(self):
        from tests.test_model_contracts import ModelContracts
        fixture = ModelContracts()
        fixture.setUp()
        fixture.model.global_fusion = GlobalFusion(dict(fixture.cfg, global_fusion='concat')).eval()
        torch.nn.init.normal_(fixture.model.global_fusion.out.weight, std=.01)
        result = fixture.model(fixture.batch)
        ledger = result['ledger']
        score = ledger['bias'] + ledger['single'].sum((1,2)) + ledger['pair'].sum((1,2)) + ledger['global_fusion']
        torch.testing.assert_close(score[:,:3], result['logits'])
        torch.testing.assert_close(3*score[:,3].tanh(), result['intensity'])

    def test_observation_contracts(self):
        torch.set_num_threads(2)
        for mode in ('concat', 'attention'):
            with self.subTest(mode=mode):
                model = GlobalFusion({'hidden':8, 'heads':2,
                                      'global_fusion':mode, 'unimodal_aux':.1}).eval()
                values = [torch.randn(4,50,d) for d in (768,74,35)]
                mask = torch.rand(4,3,50) > .3
                mask[0] = False
                mask[1,1] = False
                initial, _, _ = model(values, mask)
                torch.testing.assert_close(initial, torch.zeros_like(initial))
                torch.nn.init.normal_(model.out.weight, std=.01)
                first, aux, available = model(values, mask)
                changed = [v.masked_fill(~mask[:,m,:,None], 1e6)
                           for m,v in enumerate(values)]
                second, _, _ = model(changed, mask)
                torch.testing.assert_close(first, second)
                self.assertTrue(torch.isfinite(first).all())
                torch.testing.assert_close(first[0], torch.zeros(4))
                output = {'logits':first[:,:3], 'aux_logits':aux,
                          'aux_available':available}
                loss = first.square().sum() + auxiliary_loss(output, torch.tensor([0,1,2,0]))
                loss.backward()
                self.assertTrue(torch.isfinite(model.project[1][0].weight.grad).all())


if __name__ == '__main__':
    unittest.main()
