"""按报告三 seed 冻评 test 252，再跨训练 seed 取均值。
对齐：2027/2028/2035；未对齐：2026/2031/2033。

python -m problem2_retrain_v2.run_test252_mean3 --branch aligned --device cuda:3
python -m problem2_retrain_v2.run_test252_mean3 --branch unaligned --device cuda:2
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

BRANCH = {
    'aligned': {
        'seeds': (2027, 2028, 2035),
        'ckpt': {
            2027: 'AAAmodel/checkpoints/ablations_retrain_v2/f1_gate_component_ablation_five_seeds/B0_seed2027.pt',
            2028: 'AAAmodel/checkpoints/ablations_retrain_v2/f1_gate_component_ablation_five_seeds/B0_seed2028.pt',
            2035: 'AAAmodel/checkpoints/ablations_retrain_v2/f1_gate_component_ablation_s2031_2035/B0_seed2035.pt',
        },
        'root': 'problem2_retrain_v2/results/aligned/full_runs/gate_B0_test252_mean3',
        'title': '对齐 F1+门控 B0（2027/2028/2035）252组均值',
    },
    'unaligned': {
        'seeds': (2026, 2031, 2033),
        'ckpt': {
            2026: 'AAAmodel/checkpoints/ablations_retrain_v2/softalign_gate_component_ablation_five_seeds/B0_seed2026.pt',
            2031: 'AAAmodel/checkpoints/ablations_retrain_v2/softalign_gate_component_ablation_s2031_2035/B0_seed2031.pt',
            2033: 'AAAmodel/checkpoints/ablations_retrain_v2/softalign_gate_component_ablation_s2031_2035/B0_seed2033.pt',
        },
        'root': 'problem2_retrain_v2/results/unaligned/full_runs/gate_B0_test252_mean3',
        'title': '未对齐 soft_alignment+门控 B0（2026/2031/2033）252组均值',
    },
}

METRIC_COLS = [
    'accuracy', 'macro_f1', 'weighted_f1', 'neutral_precision', 'neutral_recall',
    'neutral_f1', 'mae', 'pearson', 'head_disagreement', 'actual_removed_fraction',
]


def pct(x):
    return f'{float(x):.2%}'


def run_one(branch, seed, device, root):
    spec = BRANCH[branch]
    ckpt = spec['ckpt'][seed]
    out = root / f'seed{seed}'
    done = out / 'test_scenarios' / 'local_scenarios.csv'
    if done.exists() and (out / 'run_meta.json').exists():
        print(f'REUSE {branch} seed={seed}', flush=True)
        return json.loads((out / 'run_meta.json').read_text())
    out.mkdir(parents=True, exist_ok=True)
    digest = sha256(path(ckpt))
    model, cfg = restore(ckpt, device)
    data = load_split(cfg['data_dir'], 'test')
    bs = 16 if cfg.get('architecture') == 'soft_alignment' else 128
    full, full_frame = evaluate(model, data, device, batch_size=bs)
    full_frame.to_csv(out / 'test_full.csv', index=False)
    json_write(out / 'test_full.json', full)
    partial, records = masked_view(
        data, None, .3, (0, 1, 2), 'random', 4026,
        reencode=not cfg.get('finetune_bert', False))
    local30, local_frame = evaluate(model, partial, device, batch_size=bs)
    local_frame.to_csv(out / 'test_local30.csv', index=False)
    json_write(out / 'test_local30.json', local30)
    json_write(out / 'test_local30_mask.json', records)
    print(json.dumps({
        'branch': branch, 'seed': seed,
        'full': full['accuracy'], 'local30': local30['accuracy'],
    }, ensure_ascii=False), flush=True)
    evaluate_scenarios(
        model, data, None, device, out,
        figures=str(out / 'figures'), checkpoint=ckpt)
    meta = {
        'branch': branch, 'seed': seed, 'checkpoint': ckpt,
        'checkpoint_sha256': digest,
        'full_accuracy': full['accuracy'], 'local30_accuracy': local30['accuracy'],
        'device': device,
    }
    json_write(out / 'run_meta.json', meta)
    print(f'DONE {branch} seed={seed}', flush=True)
    return meta


def aggregate(branch, root):
    spec = BRANCH[branch]
    seeds = list(spec['seeds'])
    full_rows, local30_rows, scen_frames = [], [], []
    for seed in seeds:
        out = root / f'seed{seed}'
        full_rows.append({'seed': seed, **json.loads((out / 'test_full.json').read_text())})
        local30_rows.append({'seed': seed, **json.loads((out / 'test_local30.json').read_text())})
        frame = pd.read_csv(out / 'test_scenarios' / 'local_scenarios.csv')
        frame['train_seed'] = seed
        scen_frames.append(frame)
    full_df = pd.DataFrame(full_rows)
    local30_df = pd.DataFrame(local30_rows)
    full_df.to_csv(root / 'full_by_seed.csv', index=False)
    local30_df.to_csv(root / 'local30_by_seed.csv', index=False)
    all_scen = pd.concat(scen_frames, ignore_index=True)
    all_scen.to_csv(root / 'local_scenarios_by_seed.csv', index=False)

    # 随机位置：先对掩码 seed 平均，再跨训练 seed
    available = [c for c in METRIC_COLS if c in all_scen.columns]
    pos = (all_scen.groupby(['train_seed', 'modalities', 'rate', 'position'], as_index=False)[available]
           .mean())
    pos.to_csv(root / 'position_by_train_seed.csv', index=False)
    # 跨训练 seed：场景级 mean/std
    keys = ['modalities', 'rate', 'position']
    mean = pos.groupby(keys, as_index=False)[available].mean()
    std = pos.groupby(keys, as_index=False)[available].std(ddof=1)
    mean.to_csv(root / 'position_effects_mean.csv', index=False)
    std.to_csv(root / 'position_effects_std.csv', index=False)
    duration = mean.groupby(['modalities', 'rate'], as_index=False)[available].mean()
    duration.to_csv(root / 'duration_effects_mean.csv', index=False)
    overall = duration.groupby('rate', as_index=False)[available].mean()
    overall.to_csv(root / 'overall_by_rate_mean.csv', index=False)

    def mean_std(frame, col):
        return float(frame[col].mean()), float(frame[col].std(ddof=1))

    fa, fs = mean_std(full_df, 'accuracy')
    la, ls = mean_std(local30_df, 'accuracy')
    lines = [
        f'# {spec["title"]}',
        '',
        f'训练种子：**{"、".join(str(s) for s in seeds)}**；每 seed 冻评完整 252 组后跨 seed 取均值。',
        '随机位置先对 3 个掩码 seed 平均，再对四种位置等权，再对七模态等权。',
        '',
        '## 1. 完整输入与 TAV 随机 30%（跨训练 seed）',
        '',
        '| 设定 | ACC均值 | ACC标准差 | Macro-F1均值 | MAE均值 |',
        '|---|---:|---:|---:|---:|',
        f'| 完整test | {pct(fa)} | {fs:.2%} | {pct(full_df.macro_f1.mean())} | {full_df.mae.mean():.4f} |',
        f'| TAV随机30% | {pct(la)} | {ls:.2%} | {pct(local30_df.macro_f1.mean())} | {local30_df.mae.mean():.4f} |',
        '',
        '各 seed 完整 ACC：'
        + '、'.join(f'{int(r.seed)}={r.accuracy:.2%}' for r in full_df.sort_values('seed').itertuples()),
        '',
        '## 2. 按缺失跨度汇总（均值）',
        '',
        '| 标称跨度 | ACC | Macro-F1 | 中性F1 | MAE | Pearson | 实际删观测比例 |',
        '|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in overall.itertuples():
        lines.append(
            f'| {r.rate:.0%} | {pct(r.accuracy)} | {pct(r.macro_f1)} | {pct(r.neutral_f1)} | '
            f'{r.mae:.4f} | {r.pearson:.4f} | {r.actual_removed_fraction:.2%} |')
    lines += ['', '## 3. 模态 × 跨度 ACC（均值）', '',
              '| 模态 | 10% | 20% | 30% | 40% | 50% | 70% |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name in ('T', 'A', 'V', 'TA', 'TV', 'AV', 'TAV'):
        cells = []
        for rate in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7):
            row = duration[(duration.modalities == name) & np.isclose(duration.rate, rate)]
            cells.append(pct(row.accuracy.iloc[0]) if len(row) else '—')
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    lines += [
        '',
        '逐 seed 场景：`seed<seed>/test_scenarios/local_scenarios.csv`；',
        '合并：`local_scenarios_by_seed.csv`；均值表：`overall_by_rate_mean.csv`。',
        '',
    ]
    (root / '全部结果表.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    json_write(root / 'summary.json', {
        'branch': branch, 'seeds': seeds, 'title': spec['title'],
        'full_accuracy_mean': fa, 'full_accuracy_std': fs,
        'local30_accuracy_mean': la, 'local30_accuracy_std': ls,
        'n_scenarios_per_seed': 252,
    })
    print(f'AGGREGATED {branch}: full={fa:.2%}±{fs:.2%} local30={la:.2%}±{ls:.2%}', flush=True)


def maybe_import_legacy_unaligned_2026(root):
    """复用先前单跑的 unaligned seed2026（若场景齐全）。"""
    legacy = path('problem2_retrain_v2/results/unaligned/full_runs/gate_B0_test252')
    dest = root / 'seed2026'
    scen = legacy / 'test_scenarios' / 'local_scenarios.csv'
    if dest.exists() or not scen.exists():
        return
    if len(pd.read_csv(scen)) != 252:
        return
    dest.mkdir(parents=True)
    import shutil
    for name in ('test_full.csv', 'test_full.json', 'test_local30.csv', 'test_local30.json',
                 'test_local30_mask.json', 'test_scenarios', 'figures', 'run_meta.json'):
        src = legacy / name
        if src.exists():
            target = dest / name
            if src.is_dir():
                shutil.copytree(src, target)
            else:
                shutil.copy2(src, target)
    meta = json.loads((dest / 'run_meta.json').read_text())
    meta['seed'] = 2026
    meta['imported_from'] = str(legacy)
    json_write(dest / 'run_meta.json', meta)
    print('IMPORTED unaligned seed2026 from legacy single-seed run', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch', choices=('aligned', 'unaligned', 'both'), default='both')
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--aggregate-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    branches = ('aligned', 'unaligned') if args.branch == 'both' else (args.branch,)
    for branch in branches:
        root = path(BRANCH[branch]['root'])
        root.mkdir(parents=True, exist_ok=True)
        if branch == 'unaligned':
            maybe_import_legacy_unaligned_2026(root)
        if not args.aggregate_only:
            for seed in BRANCH[branch]['seeds']:
                run_one(branch, seed, args.device, root)
        aggregate(branch, root)


if __name__ == '__main__':
    main()
