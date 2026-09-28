"""核验已保存的全部特征，不把结构验证冒充人工对齐精度。"""
import argparse
import numpy as np
import pandas as pd
from shared.common import config, path, json_write
from problem1.alignment import validate_intervals


def verify(cfg):
    root = path(cfg['q1']['output'])
    summary = {}
    for variant in ('word', 'uniform'):
        table = pd.read_csv(root / f'manifest_{variant}.csv')
        if len(table) != 100 or table.id.nunique() != 100:
            raise AssertionError('原始100个样本必须完整且不重复')
        if not table.status.eq('ok').all():
            raise AssertionError('存在提取失败，不能通过全量验收')
        for row in table.itertuples():
            with np.load(root / row.filename) as data:
                k = len(data['intervals'])
                validate_intervals(data['intervals'], row.duration)
                for name, width in [('text', 768), ('audio', 25), ('vision', 35)]:
                    if data[name].shape != (k, width): raise AssertionError('全长形状错误')
                    if data[f'view50_{name}'].shape != (50, width): raise AssertionError('50视图形状错误')
                    np.testing.assert_array_equal(data[f'view50_{name}'][:min(k,50)], data[name][:50])
                    if np.any(data[f'view50_{name}'][min(k,50):]): raise AssertionError('填充不是零')
                np.testing.assert_array_equal(data['view50_intervals'][:min(k,50)], data['intervals'][:50])
                np.testing.assert_array_equal(data['view50_observed_mask'][:,:min(k,50)], data['observed_mask'][:,:50])
                for name in data.files:
                    if not np.isfinite(data[name]).all(): raise AssertionError(f'{row.id}/{name}: 非有限值')
                if np.any(data['observed_mask'] & ~data['valid_mask']): raise AssertionError('掩码越界')
        summary[variant] = {'samples': len(table), 'alignment': table.alignment.value_counts().to_dict(),
                            'audio_status': table.audio_status.value_counts().to_dict()}
    summary.update(passed=True, manual_boundary_accuracy='not_verified_no_human_reference')
    json_write(root / 'verification.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', default='configs/default.json')
    print(verify(config(parser.parse_args().config)))
