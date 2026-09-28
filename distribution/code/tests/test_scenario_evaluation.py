"""场景汇总的重复权重与valid/test产物隔离。"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
from problem2.inference import evaluate_scenarios


class ScenarioEvaluationTests(unittest.TestCase):
    def test_split_outputs_preserve_all_metrics_and_equal_position_weight(self):
        grid=[('T',(0,),.3,p,s) for p in ('start','middle','end','random')
              for s in ((2026,2027,2028) if p=='random' else (2026,))]
        def view(data,encoder,rate,modalities,position,seed,**kwargs):
            masks=[{'id':x,'observed_before':2,'observed_after':1,'actual_removed_fraction':.5} for x in data['id']]
            return data|{'score':float(position=='random')},masks
        def evaluation(model,data,device):
            keys=('accuracy','macro_f1','neutral_precision','neutral_recall','neutral_f1','mae','pearson')
            return {k:data['score'] for k in keys},pd.DataFrame({'id':data['id'],'class':[1,1]})
        with tempfile.TemporaryDirectory() as tmp, \
             patch('problem2.inference.scenario_grid',return_value=grid), \
             patch('problem2.inference.masked_view',side_effect=view), \
             patch('problem2.inference.evaluate',side_effect=evaluation):
            root=Path(tmp);model=SimpleNamespace(cfg={'finetune_bert':True,'seed':7})
            for split in ('test','valid'):
                evaluate_scenarios(model,{'id':['a','b'],'split':split},None,'cpu',root,figures=root/'figures')
                summary=pd.read_csv(root/f'{split}_scenarios/duration_effects.csv')
                self.assertEqual(float(summary.accuracy.iloc[0]),.25)
                self.assertEqual(float(summary.neutral_f1.iloc[0]),.25)
            self.assertTrue((root/'test_scenarios/predictions/T_0.3_random_2028.csv').exists())
            self.assertTrue((root/'valid_scenarios/predictions/T_0.3_random_2028.csv').exists())
            self.assertTrue((root/'figures/test/local_missingness.pdf').exists())


if __name__=='__main__':unittest.main()
