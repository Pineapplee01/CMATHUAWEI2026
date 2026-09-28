"""级联中性检测方案第1步：主模型与文本先验的逐样本证据落盘。

主模型部分只在完整观测（B=全开）下前向，导出 train/valid/test 的
3 类 logits、回归强度；文本先验部分由 zero_shot_probs.py 独立完成。
输出 problem2/results/cascade/{split}_logits.npz，供 cascade.py 离线组合。
"""
import argparse
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from shared.common import config, path, to_device
from problem2.runtime import restore, dataset, loader


@torch.no_grad()
def dump(split, model, cfg, destination):
    ds = dataset(cfg, split)
    logits, intensities, labels, regression = [], [], [], []
    for raw in loader(ds, cfg):
        batch = to_device(raw, cfg['device'])
        out = model(batch)  # B=None：完整观测视图
        logits.append(out['logits'].cpu().numpy())
        intensities.append(out['intensity'].cpu().numpy())
        labels.append(batch['c'].cpu().numpy())
        regression.append(batch['y'].cpu().numpy())
    np.savez_compressed(destination, logits=np.concatenate(logits),
                        intensity=np.concatenate(intensities),
                        label=np.concatenate(labels).astype(np.int64),
                        regression=np.concatenate(regression),
                        ids=np.asarray(ds.ids))
    print(f'{split}: {len(ds)} samples -> {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/final.json')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--output', default='problem2/results/cascade')
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'])
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    cfg = config(args.config)
    checkpoint = args.checkpoint or cfg['checkpoint']
    model, restored = restore(checkpoint, args.device)
    output = path(args.output); output.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        if split == 'test':
            from shared.data import assert_disjoint
            assert_disjoint(dataset(restored, 'train'), dataset(restored, 'valid'), dataset(restored, 'test'))
        dump(split, model, restored, output / f'{split}_logits.npz')
