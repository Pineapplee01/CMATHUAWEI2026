"""从已提取的全长特征重建截断视图，无需重复运行媒体提取器。"""
import argparse
import json
import numpy as np
import pandas as pd
from shared.common import config, path, json_write
from problem1.alignment import view50, view50_metadata


def rebuild(cfg):
    root = path(cfg['q1']['output'])
    rows = []
    for variant in ('word', 'uniform'):
        manifest = pd.read_csv(root / f'manifest_{variant}.csv')
        for row in manifest.itertuples():
            file = root / row.filename
            with np.load(file) as saved:
                arrays = {key: saved[key].copy() for key in saved.files}
            features = {name: arrays[name] for name in ('text', 'audio', 'vision')}
            count = len(arrays['intervals'])
            view, observed, times, _ = view50(arrays['intervals'], features, arrays['observed_mask'])
            arrays.update({f'view50_{name}': value for name, value in view.items()})
            arrays.update(view50_intervals=times, view50_observed_mask=observed,
                          view50_valid_mask=np.broadcast_to(np.arange(50)<min(count,50),(3,50)))
            temp = file.with_suffix('.tmp.npz')
            np.savez_compressed(temp, **arrays); temp.replace(file)
            metadata = json.loads(file.with_suffix('.json').read_text())
            metadata.pop('view50_merge_map', None)
            metadata.pop('granularity_loss', None)
            metadata.update(view50_metadata(count))
            json_write(file.with_suffix('.json'), metadata)
            rows.append({'id':row.id, 'variant':variant, **view50_metadata(count)})
    table = pd.DataFrame(rows).drop(columns=['view50_index_map'])
    table.to_csv(root / 'view50_truncation.csv', index=False)
    json_write(root / 'view50_policy.json', {'policy':'truncate_prefix_50',
               'archive':'full length retained; view only truncated',
               'text_context':'existing full-text BERT features retained; not re-encoded',
               'variants':{'word':'first 50 words', 'uniform':'first 50 uniform bins'}})
    print(table[table.view50_truncated].to_string(index=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/default.json')
    rebuild(config(parser.parse_args().config))
