import unittest
import numpy as np
import torch
from problem2.local_missingness import local_mask, scenario_grid


class LocalMissingnessTests(unittest.TestCase):
    def test_sparse_observations_never_become_whole_missing(self):
        batch = {'id':['a','b'], 'P':torch.ones(2,3,50,dtype=torch.bool),
                 'O':torch.zeros(2,3,50,dtype=torch.bool)}
        batch['O'][0,:,20:22] = True
        batch['O'][1,:,25] = True
        for position in ('start','middle','end','random'):
            mask, records = local_mask(batch,.7,(0,1,2),position,2026)
            self.assertTrue((batch['O'] & mask).any(-1).all())
            for sample in mask:
                for modality in sample:
                    hidden = (~modality).nonzero().flatten().numpy()
                    self.assertTrue(len(hidden)<2 or np.all(np.diff(hidden)==1))
            self.assertTrue(all(r['observed_after']>=1 for r in records))

    def test_batching_does_not_change_masks(self):
        batch = {'id':['a','b','c'], 'P':torch.ones(3,3,50,dtype=torch.bool),
                 'O':torch.ones(3,3,50,dtype=torch.bool)}
        full, _ = local_mask(batch,.3,(0,2),'random',2027)
        for i in range(3):
            one = {k:v[i:i+1] for k,v in batch.items()}
            part, _ = local_mask(one,.3,(0,2),'random',2027,offset=i)
            torch.testing.assert_close(full[i:i+1],part)
        self.assertTrue(full[:,1].all())

    def test_grid_covers_required_factors_without_whole_dropout(self):
        grid = list(scenario_grid())
        self.assertEqual(len(grid),168)
        self.assertEqual({row[0] for row in grid},{'T','A','V','TA','TV','AV','TAV'})
        self.assertTrue(all(0 < row[2] < 1 for row in grid))

    def test_test_grid_adds_intermediate_rates_and_keeps_validation_protocol(self):
        old=list(scenario_grid())
        expanded=list(scenario_grid(rates=(.1,.2,.3,.4,.5,.7)))
        self.assertEqual(len(expanded),252)
        self.assertEqual(len(set(expanded)),252)
        self.assertTrue(set(old).issubset(set(expanded)))
        for name in ('T','A','V','TA','TV','AV','TAV'):
            for rate in (.1,.2,.3,.4,.5,.7):
                rows=[r for r in expanded if r[0]==name and r[2]==rate]
                self.assertEqual(sum(r[3]=='random' for r in rows),3)
                self.assertEqual(sum(r[3]!='random' for r in rows),3)


if __name__ == '__main__': unittest.main()
