"""快速评估软标签变体最佳检查点在 valid 上的混淆结构。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from shared.common import config, to_device
from problem2.runtime import restore, dataset, loader

@torch.no_grad()
def run(cfg_path, checkpoint):
    cfg = config(cfg_path)
    model, _ = restore(checkpoint, 'cuda:3')
    ds = dataset(cfg, 'valid')
    preds, ys = [], []
    for raw in loader(ds, cfg):
        b = to_device(raw, 'cuda:3')
        out = model(b)
        preds.append(out['logits'].argmax(-1).cpu().numpy())
        ys.append(b['c'].cpu().numpy())
    p, y = np.concatenate(preds), np.concatenate(ys)
    cm = np.zeros((3, 3), int)
    for t, q in zip(y, p):
        cm[t, q] += 1
    acc = (p == y).mean()
    recalls = [cm[c, c] / cm[c].sum() for c in range(3)]
    print(f'{checkpoint}: valid acc={acc:.4f} recalls(neg/neu/pos)={[round(r,3) for r in recalls]}')
    print('  pred-dist:', (np.bincount(p, minlength=3) / len(p)).round(3).tolist(),
          'true-dist:', (np.bincount(y, minlength=3) / len(y)).round(3).tolist())
    print(cm)

if __name__ == '__main__':
    run('configs/soft_label.json', 'AAAmodel/checkpoints/soft_label.pt')
    run('configs/soft_label_tight.json', 'AAAmodel/checkpoints/soft_label_tight.pt')
