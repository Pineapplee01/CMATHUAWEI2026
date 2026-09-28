"""模块4-发现1验证：中性误判为正是否由视觉模态驱动（加性账本逐模态分解）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from shared.common import config, to_device
from problem2.runtime import restore, dataset, loader

cfg = config('configs/final.json')
model, saved = restore(cfg['checkpoint'], 'cuda:3')
ds = dataset(saved, 'valid')
single, q_rel, ys, ps = [], [], [], []
with torch.no_grad():
    for raw in loader(ds, cfg):
        b = to_device(raw, 'cuda:3')
        out = model(b)
        # single: [B,3,windows,4] 每模态逐窗口对4维输出的加性贡献（已背景中心化）
        single.append(out['ledger']['single'].sum(2).cpu().numpy())  # [B,3,4]
        q_rel.append(out['q'].mean(-1).cpu().numpy())                # [B,3] 平均可靠性
        ys.append(b['c'].cpu().numpy()); ps.append(out['logits'].argmax(-1).cpu().numpy())
S = np.concatenate(single); Q = np.concatenate(q_rel)
y = np.concatenate(ys); p = np.concatenate(ps)
np.savez('/tmp/ledger_valid.npz', S=S, Q=Q, y=y, p=p)

MOD = ('text', 'audio', 'vision')
err_np = (y == 1) & (p == 2)   # 中性误判为正
cor_n = (y == 1) & (p == 1)    # 正确中性
err_nn = (y == 1) & (p == 0)   # 中性误判为负
print(f'中性→正错误 n={err_np.sum()}, 正确中性 n={cor_n.sum()}, 中性→负 n={err_nn.sum()}')
print('\n各模态对【正类logit】的平均贡献（背景中心化后，>0 表示推向正）:')
for m, name in enumerate(MOD):
    e = S[err_np, m, 2].mean(); c = S[cor_n, m, 2].mean()
    print(f'  {name:6s}: 误判为正 {e:+.3f} vs 正确中性 {c:+.3f}  差 {e-c:+.3f}')
print('\n误判为正样本中各模态正类贡献为正的占比:')
for m, name in enumerate(MOD):
    print(f'  {name:6s}: {(S[err_np, m, 2] > 0).mean():.3f}')
print('\n对照：正确中性样本中各模态正类贡献为正的占比:')
for m, name in enumerate(MOD):
    print(f'  {name:6s}: {(S[cor_n, m, 2] > 0).mean():.3f}')
print('\n模态平均可靠性 q（误判为正 vs 正确中性）:')
for m, name in enumerate(MOD):
    print(f'  {name:6s}: {Q[err_np, m].mean():.3f} vs {Q[cor_n, m].mean():.3f}')
