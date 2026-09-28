"""冻结 B0 seed2026：对齐 / 未对齐各自 test 252 组连续缺失评估。
python -m problem2_retrain_v2.run_test252 --branch aligned --device cuda:3
python -m problem2_retrain_v2.run_test252 --branch unaligned --device cuda:2
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from shared.common import path, json_write
from problem2_retrain_v2.data import load_split, masked_view, sha256
from problem2_retrain_v2.training import restore, evaluate
from problem2_retrain_v2.inference import evaluate_scenarios

SPECS = {
    'aligned': {
        'checkpoint': 'AAAmodel/checkpoints/ablations_retrain_v2/f1_gate_component_ablation_five_seeds/B0_seed2026.pt',
        'output': 'problem2_retrain_v2/results/aligned/full_runs/gate_B0_test252',
        'figures': 'problem2_retrain_v2/results/aligned/full_runs/gate_B0_test252/figures',
        'title': 'problem2_retrain_v2 对齐 F1+门控（B0 seed2026）252组连续缺失',
    },
    'unaligned': {
        'checkpoint': 'AAAmodel/checkpoints/ablations_retrain_v2/softalign_gate_component_ablation_five_seeds/B0_seed2026.pt',
        'output': 'problem2_retrain_v2/results/unaligned/full_runs/gate_B0_test252',
        'figures': 'problem2_retrain_v2/results/unaligned/full_runs/gate_B0_test252/figures',
        'title': 'problem2_retrain_v2 未对齐 soft_alignment+门控（B0 seed2026）252组连续缺失',
    },
}


def pct(x):
    return f'{float(x):.2%}'


def write_report(root, title, ckpt, digest, full, local30):
    scen = root / 'test_scenarios'
    overall = pd.read_csv(scen / 'overall_by_rate.csv')
    duration = pd.read_csv(scen / 'duration_effects.csv')
    local = pd.read_csv(scen / 'local_scenarios.csv')
    lines = [
        f'# {title}',
        '',
        f'权重：`{ckpt}`；SHA256=`{digest}`；冻结评估，不重训。',
        '网格：7 模态组合 × 6 跨度（10/20/30/40/50/70%）×（3 固定位置 + 3 次随机掩码）= **252** 组；每组 test 727 条。',
        '',
        '## 1. 完整输入与 TAV 随机 30% 参考',
        '',
        '| 设定 | ACC | Macro-F1 | 中性F1 | MAE | Pearson |',
        '|---|---:|---:|---:|---:|---:|',
        f'| 完整test | {pct(full["accuracy"])} | {pct(full["macro_f1"])} | {pct(full["neutral_f1"])} | {full["mae"]:.4f} | {full["pearson"]:.4f} |',
        f'| TAV随机30% | {pct(local30["accuracy"])} | {pct(local30["macro_f1"])} | {pct(local30["neutral_f1"])} | {local30["mae"]:.4f} | {local30["pearson"]:.4f} |',
        '',
        '## 2. 按缺失跨度汇总（七模态×四位置等权）',
        '',
        '| 标称跨度 | ACC | Macro-F1 | 中性F1 | MAE | Pearson | 实际删观测比例 |',
        '|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in overall.itertuples():
        lines.append(
            f'| {r.rate:.0%} | {pct(r.accuracy)} | {pct(r.macro_f1)} | {pct(r.neutral_f1)} | '
            f'{r.mae:.4f} | {r.pearson:.4f} | {r.actual_removed_fraction:.2%} |')
    lines += ['', '## 3. 模态 × 跨度（四位置等权后的 ACC）', '',
              '| 模态 | 10% | 20% | 30% | 40% | 50% | 70% |', '|---|---:|---:|---:|---:|---:|---:|']
    rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
    for name in ('T', 'A', 'V', 'TA', 'TV', 'AV', 'TAV'):
        cells = []
        for rate in rates:
            row = duration[(duration.modalities == name) & np.isclose(duration.rate, rate)]
            cells.append(pct(row.accuracy.iloc[0]) if len(row) else '—')
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    lines += [
        '',
        f'## 4. 全部 252 组',
        '',
        f'逐场景见 `test_scenarios/local_scenarios.csv`（{len(local)} 行）。',
        '聚合：`position_effects.csv` → `duration_effects.csv` → `overall_by_rate.csv`。',
        '',
    ]
    (root / '全部结果表.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(branch, device):
    spec = SPECS[branch]
    out = path(spec['output'])
    out.mkdir(parents=True, exist_ok=True)
    ckpt = spec['checkpoint']
    digest = sha256(path(ckpt))
    model, cfg = restore(ckpt, device)
    data = load_split(cfg['data_dir'], 'test')
    bs = 16 if cfg.get('architecture') == 'soft_alignment' else 128
    full, full_frame = evaluate(model, data, device, batch_size=bs)
    full_frame.to_csv(out / 'test_full.csv', index=False)
    json_write(out / 'test_full.json', full)
    encoder = None
    partial, records = masked_view(
        data, encoder, .3, (0, 1, 2), 'random', 4026,
        reencode=not cfg.get('finetune_bert', False))
    local30, local_frame = evaluate(model, partial, device, batch_size=bs)
    local_frame.to_csv(out / 'test_local30.csv', index=False)
    json_write(out / 'test_local30.json', local30)
    json_write(out / 'test_local30_mask.json', records)
    print(json.dumps({'branch': branch, 'full': full['accuracy'], 'local30': local30['accuracy']},
                     ensure_ascii=False), flush=True)
    evaluate_scenarios(
        model, data, encoder, device, out,
        figures=spec['figures'], checkpoint=ckpt)
    write_report(out, spec['title'], ckpt, digest, full, local30)
    json_write(out / 'run_meta.json', {
        'branch': branch,
        'model': spec['title'],
        'checkpoint': ckpt,
        'checkpoint_sha256': digest,
        'grid': '7 modalities × 6 rates × (3 fixed positions + 3 random seeds) = 252',
        'device': device,
        'full_accuracy': full['accuracy'],
        'local30_accuracy': local30['accuracy'],
    })
    print(f'完成 {branch}: {out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch', choices=('aligned', 'unaligned', 'both'), default='both')
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--aligned-device')
    parser.add_argument('--unaligned-device')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.branch in ('aligned', 'both'):
        run('aligned', args.aligned_device or args.device)
    if args.branch in ('unaligned', 'both'):
        run('unaligned', args.unaligned_device or args.device)


if __name__ == '__main__':
    main()
