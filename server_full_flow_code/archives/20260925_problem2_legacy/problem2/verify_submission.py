"""独立核验附件3 CSV与168组局部缺失契约；不修改预测或真实标签。"""
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from shared.common import path, config, json_write
from shared.data import load_fields
from problem2.runtime import dataset
from problem2.local_missingness import scenario_grid
from problem2.output_contract import project_intensity


def verify(cfg):
    output = path(cfg['output'])
    csv = pd.read_csv(output/'附件3_预测结果.csv')
    source_files = sorted(path(cfg['appendix3']).glob('*.pkl'))
    assert len(csv) == len(source_files) == 30
    assert not csv.sample_id.duplicated().any()
    assert set(csv.source_file) == {f.name for f in source_files}
    assert set(csv.sample_id) == {f.stem for f in source_files}
    assert csv.polarity.isin(['Negative','Neutral','Positive']).all()
    assert np.isfinite(csv.intensity).all() and csv.intensity.abs().le(3).all()
    classes = csv.polarity.map({'Negative':0,'Neutral':1,'Positive':2}).to_numpy()
    np.testing.assert_array_equal(classes,np.sign(csv.intensity).astype(int)+1)
    audit = pd.read_csv(output/'附件3_推理审计.csv').set_index('sample_id').loc[csv.sample_id]
    np.testing.assert_allclose(csv.intensity,project_intensity(audit['class'],audit.raw_intensity),rtol=0,atol=1e-12)
    for file in source_files:
        fields=load_fields(file)
        assert len(fields['text_bert']) == 1
        assert not any(k in fields for k in ('regression_labels','classification_labels'))
    manifest=json.loads((output/'附件3_提交核验.json').read_text())
    digest=hashlib.sha256(path(cfg['checkpoint']).read_bytes()).hexdigest()
    assert manifest['checkpoint_sha256'] == digest
    root=path('problem2/results/local_robustness')
    frozen=json.loads((root/'frozen_selection.json').read_text())
    assert frozen['sha256'] == digest
    ds=dataset(cfg,'valid')
    observed=np.stack([ds[i]['O'].numpy() for i in range(len(ds))])
    expected={f'{name}_{rate}_{position}_{seed}' for name,_,rate,position,seed in scenario_grid()}
    folders={p.name for p in (root/'selected_full/scenarios').iterdir() if p.is_dir()}
    assert folders == expected
    for name in sorted(expected):
        folder=root/'selected_full/scenarios'/name
        with np.load(folder/'mask.npz') as data:
            keep=data['B']
            assert keep.shape == observed.shape and keep.dtype == np.bool_
            np.testing.assert_array_equal(data['ids'],ds.ids)
        before=observed.sum(-1);after=(observed&keep).sum(-1)
        assert (after[before>0]>0).all(), f'{name}: 存在整模态人工删除'
        # 每个样本/模态至多有一个连续删除区间。
        deleted=~keep
        starts=deleted & ~np.concatenate([np.zeros_like(deleted[...,:1]),deleted[...,:-1]],axis=-1)
        assert (starts.sum(-1)<=1).all()
        pred=pd.read_csv(folder/'predictions.csv')
        np.testing.assert_array_equal(pred.id,ds.ids)
        np.testing.assert_array_equal(pred.true_class,np.asarray(ds.fields['classification_labels']).reshape(-1))
        np.testing.assert_allclose(pred.true_intensity,np.asarray(ds.fields['regression_labels']).reshape(-1))
        assert np.isfinite(pred.intensity).all() and pred.intensity.abs().le(3).all()
    report={'passed':True,'appendix3_samples':30,'validation_samples':len(ds),
            'local_scenarios':len(expected),'all_masks_contiguous':True,
            'no_observed_modality_fully_removed':True,'labels_unchanged':True,
            'submission_polarity_intensity_consistent':True,'checkpoint_sha256':digest,
            'special_test_accuracy_available':False}
    json_write(output/'独立提交校验.json',report)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/problem2_robust.json')
    args=parser.parse_args();print(json.dumps(verify(config(args.config)),ensure_ascii=False,indent=2))
