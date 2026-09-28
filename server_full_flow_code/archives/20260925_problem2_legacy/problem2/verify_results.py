"""独立复核预测CSV：样本覆盖、标签对应、指标与70%目标口径。"""
import argparse
import json
import numpy as np
import pandas as pd
from shared.common import config, path, json_write
from problem2.runtime import dataset
from problem2.evaluation import metrics


def verify(cfg, split):
    ds = dataset(cfg, split)
    frame = pd.read_csv(path(cfg['output']) / split / 'predictions.csv')
    if len(frame) != len(ds) or frame.id.duplicated().any() or set(frame.id) != set(ds.ids):
        raise AssertionError('预测样本覆盖或主键不一致')
    ordered = frame.set_index('id').loc[ds.ids]
    np.testing.assert_allclose(ordered.true_intensity, ds.fields['regression_labels'])
    np.testing.assert_array_equal(ordered.true_class, ds.fields['classification_labels'])
    if not np.isfinite(ordered.intensity).all() or (ordered.intensity.abs()>3).any():
        raise AssertionError('强度非有限或超界')
    if not ordered['class'].isin([0,1,2]).all(): raise AssertionError('三分类输出编码错误')
    actual = metrics(ordered)
    saved = json.loads((path(cfg["output"]) / split / "metrics.json").read_text())["metrics"]
    for key, value in actual.items():
        if value is not None and not np.isclose(value, saved[key], rtol=1e-7, atol=1e-9):
            raise AssertionError(f"保存指标与预测重算不一致: {key}")
    report = {'target_split':cfg.get('target_split', 'test'),
              'is_acceptance_split':split == cfg.get('target_split', 'test'), 'split':split, 'samples':len(ds), 'definition':'three_class_complete_observations',
              'metrics':actual, 'target_accuracy':cfg.get('target_accuracy',.7),
              'accuracy_target_met':actual['accuracy']>=cfg.get('target_accuracy',.7),
              'correct_predictions':int((ordered['class']==ordered.true_class).sum()),
              'labels_rechecked_against_original_split':True, 'saved_metrics_match':True}
    json_write(path(cfg['output']) / split / 'verified_metrics.json',report)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/default.json')
    parser.add_argument('--split',choices=['valid','test'],default='valid')
    args=parser.parse_args()
    print(verify(config(args.config),args.split))


if __name__=='__main__': main()
