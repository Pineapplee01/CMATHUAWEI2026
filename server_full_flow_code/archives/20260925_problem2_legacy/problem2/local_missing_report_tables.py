"""168组局部缺失规律分析：汇总表格与论文图。

数据源：problem2/results/local_robustness/selected_full/（冻结模型 robust_main
的完整 7类型×4时长×位置 网格，168 行）与 model_comparison.csv。
输出：tables/*.csv 与 figures/local_missing/*.png|pdf。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
from shared.common import path
from shared.plotting import pyplot

SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'valid'
SUB = 'selected_full' if SPLIT == 'valid' else 'selected_full_test'
BASE = path('problem2/results/local_robustness')
OUT_T = BASE / (f'tables' if SPLIT == 'valid' else f'tables_{SPLIT}')
OUT_F = path('problem2/figures') / (f'local_missing' if SPLIT == 'valid' else f'local_missing_{SPLIT}')
OUT_T.mkdir(parents=True, exist_ok=True)
OUT_F.mkdir(parents=True, exist_ok=True)

sc = pd.read_csv(BASE / SUB / 'scenarios.csv')
complete = json.load(open(BASE / SUB / 'complete_metrics.json'))
full_acc = complete['accuracy']

# random 位置3个种子取均值，固定位置单种子；得到 112 组唯一场景
fixed = sc[sc.position != 'random']
rand = sc[sc.position == 'random'].groupby(['modalities', 'rate', 'position'], as_index=False).mean(numeric_only=True)
uniq = pd.concat([fixed, rand], ignore_index=True)
uniq['acc_drop'] = full_acc - uniq['accuracy']

TYPE_NAME = {'T': '仅文本', 'A': '仅音频', 'V': '仅视觉', 'TA': '文本+音频',
             'TV': '文本+视觉', 'AV': '音频+视觉', 'TAV': '三者全部'}
ORDER = ['T', 'A', 'V', 'TA', 'TV', 'AV', 'TAV']

# ---- 表1：总览 ----
summary = pd.DataFrame([
    {'场景': '完整观测（无缺失）', 'Accuracy': full_acc, 'Macro-F1': complete['macro_f1'],
     'MAE': complete['mae'], 'Pearson': complete['pearson']},
    {'场景': '局部缺失 168 组均值', 'Accuracy': uniq['accuracy'].mean(),
     'Macro-F1': uniq['macro_f1'].mean(), 'MAE': uniq['mae'].mean(), 'Pearson': uniq['pearson'].mean()},
])

# ---- 表2：类型规律（行=类型，列=缺失时长） ----
type_rate = uniq.pivot_table(index='modalities', columns='rate', values='accuracy', aggfunc='mean')
type_rate = type_rate.reindex(ORDER)
type_rate['均值'] = type_rate.mean(1)
type_rate['相对完整观测平均下降'] = full_acc - type_rate['均值']
type_rate.index = [f'{TYPE_NAME[m]}（{m}）' for m in type_rate.index]
type_rate.round(4).to_csv(OUT_T / '表2_缺失类型规律.csv', encoding='utf-8-sig')

# ---- 表3：时长规律（按率聚合全部类型与位置） ----
rate_tab = uniq.groupby('rate').agg(
    Accuracy=('accuracy', 'mean'), MacroF1=('macro_f1', 'mean'), MAE=('mae', 'mean'),
    Accuracy下降=('acc_drop', 'mean')).round(4)
rate_tab.index.name = '缺失时长比例'
rate_tab.to_csv(OUT_T / '表3_缺失时长规律.csv', encoding='utf-8-sig')

# ---- 表4：位置规律（0.3 与全率两种口径） ----
pos_tab = uniq.groupby('position').agg(Accuracy=('accuracy', 'mean'), MAE=('mae', 'mean'),
                                       Accuracy下降=('acc_drop', 'mean')).round(4)
pos_tab.index.name = '缺失位置'
pos_tab.to_csv(OUT_T / '表4_缺失位置规律.csv', encoding='utf-8-sig')

# ---- 表5：模型对照（★=冻结提交模型） ----
comp = pd.read_csv(BASE / 'model_comparison.csv')  # 模型对照仅valid网格
NAME = {'original': '原完整观测主模型', 'robust_main': '鲁棒（补全+门控）',
        'robust_no_completion': '消融：无补全', 'robust_no_reliability': '消融：无门控 ★提交'}
comp_tab = comp[['model', 'full_accuracy', 'local_accuracy', 'local_accuracy_drop',
                 'local_macro_f1', 'full_mae', 'local_mae']].copy()
comp_tab['model'] = comp_tab['model'].map(NAME)
comp_tab.columns = ['模型', '完整观测Acc', '局部缺失Acc', '局部Acc下降', '缺失Macro-F1', '完整MAE', '缺失MAE']
comp_tab.round(4).to_csv(OUT_T / '表5_模型对照.csv', index=False, encoding='utf-8-sig')

# ---- 绘图 ----
plt = pyplot()
plt.rcParams['font.sans-serif'] = ['AR PL UMing CN', 'DejaVu Sans']

# 图1：时长效应——各类型 Accuracy vs 缺失比例
fig, ax = plt.subplots(figsize=(6.6, 4.2))
colors = plt.get_cmap('tab10')
for i, m in enumerate(ORDER):
    sub = uniq[uniq.modalities == m].groupby('rate')['accuracy'].mean()
    ax.plot(sub.index, sub.values, 'o-', ms=4, color=colors(i),
            label=TYPE_NAME[m], lw=1.5 if m in ('T', 'TA', 'TV', 'TAV') else 1.1)
ax.axhline(full_acc, color='gray', ls='--', lw=1)
ax.text(0.11, full_acc + 0.002, f'完整观测 {full_acc:.2%}', fontsize=8.5, color='gray')
ax.set_xlabel('缺失时长比例（相对有效时段）')
ax.set_ylabel('valid Accuracy')
ax.set_title(f'缺失时长×模态类型对预测精度的影响（{SPLIT}，168组，位置平均）')
ax.set_xticks([0.1, 0.3, 0.5, 0.7])
ax.legend(fontsize=7.5, ncol=2)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(OUT_F / f'图1_时长与类型效应.{ext}', dpi=200)

# 图2：类型效应——平均下降条形图
fig, ax = plt.subplots(figsize=(6.6, 3.8))
drop = uniq.groupby('modalities')['acc_drop'].mean().reindex(ORDER)
bars = ax.bar([TYPE_NAME[m] for m in ORDER], drop.values,
              color=['#c0392b' if v == drop.max() else '#5b8db8' for v in drop.values])
for b, v in zip(bars, drop.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.0006, f'{v:.3f}', ha='center', fontsize=8.5)
ax.set_ylabel('相对完整观测的 Accuracy 下降')
ax.set_title(f'缺失模态类型的影响（{SPLIT}，全率全位置平均）')
ax.set_ylim(0, drop.max() * 1.18)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(OUT_F / f'图2_类型效应.{ext}', dpi=200)

# 图3：位置效应——分组条形（各位置 × 0.3时长下的代表性类型 或 全类型平均）
fig, ax = plt.subplots(figsize=(6.6, 3.8))
pos_mean = uniq.groupby('position')['acc_drop'].mean().reindex(['start', 'middle', 'end', 'random'])
pos03 = uniq[uniq.rate == 0.3].groupby('position')['acc_drop'].mean().reindex(['start', 'middle', 'end', 'random'])
x = np.arange(4)
ax.bar(x - 0.18, pos03.values, 0.36, label='缺失时长 30%', color='#5b8db8')
ax.bar(x + 0.18, pos_mean.values, 0.36, label='全时长平均', color='#c0392b', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(['起始段', '中间段', '末尾段', '随机位置'])
ax.set_ylabel('相对完整观测的 Accuracy 下降')
ax.set_title(f'缺失位置的影响（{SPLIT}）')
ax.legend(fontsize=9)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(OUT_F / f'图3_位置效应.{ext}', dpi=200)

# 图4：模型对照
fig, ax = plt.subplots(figsize=(6.6, 3.8))
x = np.arange(len(comp_tab))
ax.bar(x - 0.19, comp_tab['完整观测Acc'], 0.38, label='完整观测', color='#8ea9c9')
ax.bar(x + 0.19, comp_tab['局部缺失Acc'], 0.38, label='局部缺失均值', color='#c0392b', alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(comp_tab['模型'], fontsize=8.5)
ax.set_ylabel('valid Accuracy')
ax.set_title(f'模型对照：完整观测 vs 局部缺失（{SPLIT}选择网格）')
for xi, (f1, l1) in enumerate(zip(comp_tab['完整观测Acc'], comp_tab['局部缺失Acc'])):
    ax.text(xi - 0.19, f1 + 0.003, f'{f1:.3f}', ha='center', fontsize=8)
    ax.text(xi + 0.19, l1 + 0.003, f'{l1:.3f}', ha='center', fontsize=8)
ax.legend(fontsize=9)
ax.set_ylim(0.55, 0.72)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(OUT_F / f'图4_模型对照.{ext}', dpi=200)

print('=== 表1 总览 ===')
print(summary.round(4).to_string(index=False))
print('\n=== 表2 类型规律（Accuracy） ===')
print(type_rate.round(4).to_string())
print('\n=== 表3 时长规律 ===')
print(rate_tab.to_string())
print('\n=== 表4 位置规律 ===')
print(pos_tab.to_string())
print('\n=== 表5 模型对照 ===')
print(comp_tab.round(4).to_string(index=False))
print('\ntables ->', OUT_T)
print('figures ->', OUT_F)
