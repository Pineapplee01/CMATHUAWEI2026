"""自由seed入口：参数、单seed统计及恢复行为。"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from problem2 import experiments as runner


class RunProblem2Tests(unittest.TestCase):
    def test_custom_seeds(self):
        self.assertEqual(runner.arguments(['--seed','7']).seeds,[7])
        self.assertEqual(runner.arguments(['--seeds','42','123','888']).seeds,[42,123,888])
        self.assertEqual(len(runner.arguments([]).seeds),5)

    def test_training_budget_inherits_config_and_cli_overrides(self):
        cfg={'epochs':12,'patience':4,'batch_size':16,'lr':1e-4,
             'data_dir':'AAAdata/Appendix_2/标准化/对齐版本/processed_po'}
        inherited=runner.training_settings(runner.arguments([]),cfg)
        self.assertEqual(inherited,{'epochs':12,'patience':4,'batch_size':16,'lr':1e-4,'datasets':['processed_po']})
        overridden=runner.training_settings(runner.arguments(['--epochs','2','--lr','0.00002','--datasets','processed']),cfg)
        self.assertEqual(overridden['epochs'],2);self.assertEqual(overridden['lr'],2e-5)
        self.assertEqual(overridden['datasets'],['processed']);self.assertEqual(overridden['batch_size'],16)
        with self.assertRaises(ValueError):runner.training_settings(runner.arguments([]),cfg|{'batch_size':0})

    def test_invalid_seed_rejected(self):
        for args in [['--seeds','7','7'],['--seed','-1'],['--seed',str(2**32)]]:
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):runner.arguments(args)

    def test_neutral_weight_arguments(self):
        self.assertIsNone(runner.arguments([]).neutral_weight)
        self.assertEqual(runner.arguments(['--neutral-weight','1.5']).neutral_weight,1.5)
        for value in ('0','-1','nan','inf','-inf'):
            with self.subTest(value=value),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                runner.arguments(['--neutral-weight='+value])

    def row(self):
        return {'run_id':'unit','dataset':'processed_po','precision':'float32','seed':7,
                'best_epoch':1,'valid_accuracy':.6,'test_accuracy':.7,'test_local30_accuracy':.5,
                'checkpoint':'not_loaded.pt','checkpoint_sha256':'abc'}

    def test_single_seed_std_not_faked(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);runner.summarize([self.row()],root)
            frame=pd.read_csv(root/'results.csv');summary=pd.read_csv(root/'summary.csv')
            self.assertEqual(frame.seed.tolist(),[7])
            self.assertTrue(pd.isna(summary.test_accuracy_std.iloc[0]))
            self.assertEqual(frame.neutral_weight.tolist(),[1.0])

    def test_summary_keeps_different_weights_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            runner.summarize([self.row(),self.row()|{'seed':8,'neutral_weight':1.5}],root)
            summary=pd.read_csv(root/'summary.csv')
            self.assertEqual(summary.neutral_weight.tolist(),[1.,1.5])
            self.assertEqual(summary.seed_count.tolist(),[1,1])

    def test_weight_config_precedence_and_result_propagation(self):
        from problem2.evaluation import metrics
        from problem2.model import class_balance_metadata, boundary_settings
        frame=pd.DataFrame({'id':['a','b','c'],'true_class':[0,1,2],
                            'true_intensity':[-1.,0.,1.],'class':[0,1,2],
                            'intensity':[-1.,0.,1.],'raw_intensity':[-1.,0.,1.]})
        measured=metrics(frame)
        for config_weight,cli_weight,expected,config_beta,cli_beta,beta in (
                (2.,1.5,1.5,.25,.5,.5),(2.,None,2.,0.,None,0.),(None,None,1.,None,None,1.)):
            with self.subTest(expected=expected),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);config=root/'config.json'
                base={} if config_weight is None else {'neutral_weight':config_weight}
                if config_beta is not None:base['balance_beta']=config_beta
                config.write_text(json.dumps(base))
                captured=[]
                def fake_fit(cfg,*unused):
                    cfg.update(class_balance_metadata([0,1,2,2],cfg['balance_beta'],cfg['neutral_weight']))
                    captured.append(cfg)
                    output=root/cfg['output'];output.mkdir(parents=True)
                    frame.to_csv(output/'valid_full.csv',index=False)
                    frame.to_csv(output/'valid_local30.csv',index=False)
                    return {'seed':cfg['seed'],**runner.balance_fields(cfg),**runner.boundary_settings(cfg),'best_epoch':1,
                            'epochs_run':1,'complete':measured,'checkpoint':cfg['checkpoint'],
                            'checkpoint_sha256':'abc'}
                args=['--config',str(config),'--seed','7','--device','cpu','--run-name','weight_test',
                      '--datasets','processed_po']
                if cli_weight is not None:args+=['--neutral-weight',str(cli_weight)]
                if cli_beta is not None:args+=['--balance-beta',str(cli_beta)]
                args+=['--boundary-weight','0.1']
                with patch.object(runner,'path',side_effect=lambda value:root/Path(value)), \
                     patch.object(runner,'environment',return_value={}), \
                     patch.object(runner,'MissingTextEncoder',return_value=object()), \
                     patch.object(runner,'prepare',return_value=(object(),[object()])), \
                     patch.object(runner,'train_candidate',side_effect=fake_fit), \
                     patch.object(runner,'restore',side_effect=lambda *a:(object(),captured[-1])), \
                     patch.object(runner,'evaluate',side_effect=lambda *a:(measured,frame.copy())), \
                     contextlib.redirect_stdout(io.StringIO()):
                    output=runner.run(runner.arguments(args))
                self.assertEqual(captured[0]['neutral_weight'],expected)
                protocol=json.loads((output/'protocol.json').read_text())
                self.assertEqual(protocol['settings']['neutral_weight'],expected)
                self.assertEqual(protocol['settings']['balance_beta'],beta)
                self.assertEqual(protocol['settings']['boundary_weight'],.1)
                self.assertEqual(protocol['class_balance']['processed_po']['train_class_counts'],[1,1,2])
                result=json.loads((output/'results.json').read_text())[0]
                self.assertEqual(result['seed'],7)
                self.assertEqual(result['neutral_weight'],expected)
                self.assertEqual(result['test_neutral_recall'],1.)
                self.assertEqual(result['balance_beta'],beta)
                self.assertEqual(result['boundary_weight'],.1)
                self.assertEqual(result['class_weights'],captured[0]['class_weights'])
                for name in ('valid_full.csv','test_full_seed7.csv','test_local30_seed7.csv'):
                    predictions=pd.read_csv(output/'processed_po_seed7'/name)
                    self.assertEqual(predictions.neutral_weight.unique().tolist(),[expected])
                    self.assertEqual(predictions.balance_beta.unique().tolist(),[beta])
                    self.assertAlmostEqual(predictions.weight_neutral.iloc[0],result['class_weights'][1])

    def test_resume_completed_without_training_or_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'processed_po_seed7';out.mkdir()
            (out/'experiment_result.json').write_text(json.dumps(self.row()))
            protocol={'runner':'run_problem2_v1','run_id':'unit','base_config':{},
                      'settings':{'seeds':[7],'datasets':['processed_po'],'device':'cpu'}}
            (root/'protocol.json').write_text(json.dumps(protocol))
            with patch.object(runner,'sha256',return_value='abc'), \
                 patch.object(runner,'MissingTextEncoder',side_effect=AssertionError('不要重新加载编码器')), \
                 patch.object(runner,'train_candidate',side_effect=AssertionError('不要重训')):
                runner.run(runner.arguments(['--resume',str(root)]))
            self.assertEqual(pd.read_csv(root/'results.csv').seed.tolist(),[7])
            self.assertEqual(pd.read_csv(root/'results.csv').neutral_weight.tolist(),[1.0])
            self.assertEqual(pd.read_csv(root/'results.csv').balance_beta.tolist(),[0.0])

    def test_balance_beta_validation_and_resume_mismatch(self):
        for value in ('-0.1','1.1','nan','inf'):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                runner.arguments(['--balance-beta='+value])
        self.assertEqual(runner.arguments(['--balance-beta','0']).balance_beta,0.)
        with self.assertRaisesRegex(ValueError,'平衡强度不匹配'):
            runner.check_balance({'neutral_weight':1.,'balance_beta':.5},
                                 {'neutral_weight':1.,'balance_beta':1.})
        with self.assertRaisesRegex(ValueError,'边界训练参数不匹配'):
            runner.check_balance({'neutral_weight':1.,'balance_beta':0.,'boundary_weight':.1},
                                 {'neutral_weight':1.,'balance_beta':0.,'boundary_weight':0.})

    def test_summary_keeps_balance_strengths_separate(self):
        from problem2.model import class_balance_metadata
        with tempfile.TemporaryDirectory() as directory:
            rows=[self.row()|class_balance_metadata([0,1,2,2],beta) for beta in (0.,1.)]
            runner.summarize(rows,Path(directory))
            summary=pd.read_csv(Path(directory)/'summary.csv')
            self.assertEqual(summary.balance_beta.tolist(),[0.,1.])
            self.assertEqual(summary.seed_count.tolist(),[1,1])

    def test_resume_rejects_different_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'processed_po_seed7';out.mkdir()
            (out/'experiment_result.json').write_text(json.dumps(self.row()|{'neutral_weight':1.5}))
            protocol={'runner':'run_problem2_v2','run_id':'unit','base_config':{},
                      'settings':{'seeds':[7],'datasets':['processed_po'],'device':'cpu','neutral_weight':2.}}
            (root/'protocol.json').write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError,'中性权重不匹配'), \
                 patch.object(runner,'train_candidate',side_effect=AssertionError('不要重训')):
                runner.run(runner.arguments(['--resume',str(root)]))


if __name__=='__main__':unittest.main()
