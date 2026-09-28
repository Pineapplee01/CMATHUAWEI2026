"""冻结检查点的连续局部缺失验证；不读取附件3标签或test调参。"""
import argparse
import hashlib
import time
import numpy as np
import pandas as pd
import torch
from shared.common import path, config, to_device, json_write
from problem2.runtime import restore, dataset, loader
from problem2.evaluation import metrics
from problem2.local_missingness import local_mask, scenario_grid


@torch.no_grad()
def evaluate_case(model, ds, cfg, modalities=(), rate=0, position='start', seed=2026):
    model.eval()
    rows, audits, masks = [], [], []
    offset = 0
    for raw in loader(ds, cfg):
        keep, record = local_mask(raw, rate, modalities, position, seed, offset)
        out = model(to_device(raw, cfg['device']), keep.to(cfg['device']))
        predictions = out['logits'].argmax(-1).cpu().tolist()
        intensity = out['intensity'].cpu().tolist()
        for i, sid in enumerate(raw['id']):
            rows.append({'id':sid, 'class':predictions[i], 'intensity':intensity[i],
                         'true_class':int(raw['c'][i]), 'true_intensity':float(raw['y'][i])})
        audits.extend(record)
        masks.append(keep.numpy())
        offset += len(raw['id'])
    return pd.DataFrame(rows), pd.DataFrame(audits), np.concatenate(masks)


def run(cfg, output, compact=False, split='valid'):
    model, saved = restore(cfg['checkpoint'], cfg['device'])
    saved['data_dir'] = cfg['data_dir']
    ds = dataset(saved, split)
    output = path(output); output.mkdir(parents=True, exist_ok=True)
    json_write(output/'protocol.json', {
        'split':split, 'checkpoint':cfg['checkpoint'],
        'checkpoint_sha256':hashlib.sha256(path(cfg['checkpoint']).read_bytes()).hexdigest(),
        'samples':len(ds), 'strict_partial':True, 'duration_unit':'aligned_slot / inferred_effective_span',
        'padding_rule':saved['mask_rule'], 'random_seeds':[2026,2027,2028],
        'compact_grid':compact, 'time_seconds_available':False,
        'selection_use':'post-training diagnosis; checkpoints frozen before this grid'})
    baseline, _, _ = evaluate_case(model, ds, saved)
    baseline.to_csv(output/'complete_predictions.csv', index=False)
    json_write(output/'complete_metrics.json', metrics(baseline))
    rows = []
    for name, modalities, rate, position, seed in scenario_grid():
        if compact and (rate != .3 or position != 'random'): continue
        key = f'{name}_{rate}_{position}_{seed}'
        start = time.monotonic()
        pred, audit, keep = evaluate_case(model, ds, saved, modalities, rate, position, seed)
        folder = output/'scenarios'/key; folder.mkdir(parents=True, exist_ok=True)
        pred.to_csv(folder/'predictions.csv', index=False)
        audit.to_csv(folder/'mask_audit.csv', index=False)
        np.savez_compressed(folder/'mask.npz', B=keep, ids=np.asarray(ds.ids))
        assert (audit.loc[audit.observed_before > 0, 'observed_after'] > 0).all()
        result = {'modalities':name, 'rate':rate, 'position':position, 'seed':seed,
                  'mean_interval_slots':float(audit.interval_slots.mean()),
                  'actual_removed_fraction':float(audit.removed_observed_slots.sum()/max(audit.observed_before.sum(),1)),
                  'unchanged_sample_modality_fraction':float((audit.removed_observed_slots == 0).mean()),
                  **metrics(pred), 'seconds':time.monotonic()-start}
        rows.append(result)
        pd.DataFrame(rows).to_csv(output/'scenarios.csv', index=False)
        print(key, {k:round(result[k],4) for k in ('accuracy','macro_f1','mae')}, flush=True)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/final.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--compact', action='store_true', help='仅30%随机局部缺失，七类×三个种子')
    parser.add_argument('--split', default='valid', choices=['valid','test'])
    args = parser.parse_args()
    run(config(args.config), args.output, args.compact, args.split)
