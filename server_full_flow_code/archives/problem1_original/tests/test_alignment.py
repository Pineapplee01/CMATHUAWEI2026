"""区间聚合、长序列覆盖和填充语义的数值测试。"""
import unittest
import numpy as np
from problem1.alignment import aggregate, view50


class AlignmentContracts(unittest.TestCase):
    def test_overlap_weighted_mean(self):
        target = np.array([[0.,2.],[2.,3.]])
        source = np.array([[0.,1.],[1.,2.]])
        result, observed, mapping = aggregate(target,source,np.array([[2.],[4.]]),np.array([1.,.5]))
        np.testing.assert_allclose(result[:,0],[8/3,0])
        np.testing.assert_array_equal(observed,[True,False])
        self.assertEqual(mapping[1]['indices'],[])

    def test_view50_truncates_prefix_synchronously(self):
        intervals = np.stack([np.arange(73),np.arange(1,74)],-1).astype(float)
        features = {n:np.broadcast_to(np.arange(73,dtype=np.float32)[:,None], (73,d)).copy() for n,d in [('text',768),('audio',25),('vision',35)]}
        view,observed,times,groups = view50(intervals,features,np.ones((3,73),dtype=bool))
        self.assertEqual([i for g in groups for i in g],list(range(50)))
        self.assertEqual(times[-1,1],50)
        for name in features:
            np.testing.assert_array_equal(view[name],features[name][:50])
        self.assertEqual(view['text'].shape,(50,768))
        self.assertTrue(observed.all())

    def test_padding_is_not_observed(self):
        intervals=np.array([[0.,1.],[1.,2.]])
        features={n:np.ones((2,d)) for n,d in [('text',768),('audio',25),('vision',35)]}
        _,observed,_,_=view50(intervals,features,np.ones((3,2),dtype=bool))
        self.assertFalse(observed[:,2:].any())


if __name__=='__main__': unittest.main()
