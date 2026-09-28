"""十 seed（2026–2035）B0 冻评 test 252，跨训练 seed 取均值。
复用 mean3 / 单 seed 已有结果；缺的补跑。

python -m problem2_retrain_v2.run_test252_mean10 --branch aligned --device cuda:3
python -m problem2_retrain_v2.run_test252_mean10 --branch unaligned --device cuda:2
"""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from shared.common import path, json_write
from problem2_retrain_v2.data import load_split, masked_view, sha256
from problem2_retrain_v2.training import restore, evaluate
from problem2_retrain_v2.inference import evaluate_scenarios

SEEDS = tuple(range(2026, 2036))
METRIC_COLS = [
    'accuracy', 'macro_f1', 'weighted_f1', 'neutral_precision', 'neutral_recall',
    'neutral_f1', 'mae', 'pearson', 'head_disagreement', 'actual_removed_fraction',
]

BRANCH = {
    'aligned': {
        'root': 'problem2_retrain_v2/results/aligned/full_runs/gate_B0_test252_mean10',
        'title': '对齐 F1+门控 B0（2026–2035 十 seed）252组均值',
        'ckpt_dir': {
            **{s: f'AAAmodel/checkpoints/ablations_retrain_v2/f1_gate_component_ablation_five_seeds/B0_seed{s}.pt'
               for s in range(2026, 2031)},
            **{s: f'AAAmodel/checkpoints/ablations_retrain_v2/f1_gate_component_ablation_s2031_2035/B0_seed{s}.pt'
               for s in range(2031, 2036)},
        },
        'reuse': {
            2026: 'problem2_retrain_v2/results/aligned/full_runs/gate_B0_test252',
            2027: 'problem2_retrain_v2/results/aligned/full_runs/gate_B0_test252_mean3/seed2027',
            2028: 'problem2_retrain_v2/results/aligned/full_runs/gate_B0_test252_mean3/seed2028',
            2035: 'problem2_retrain_v2/results/aligned/full_runs/gate_B0_test252_mean3/seed2035',
        },
    },
    'unaligned': {
        'root': 'problem2_retrain_v2/results/unaligned/full_runs/gate_B0_test252_mean10',
        'title': '未对齐 soft_alignment+门控 B0（2026–2035 十 seed）252组均值',
        'ckpt_dir': {
            **{s: f'AAAmodel/checkpoints/ablations_retrain_v2/softalign_gate_component_ablation_five_seeds/B0_seed{s}.pt'
               for s in range(2026, 2031)},
            **{s: f'AAAmodel/checkpoints/ablations_retrain_v2/softalign_gate_component_ablation_s2031_2035/B0_seed{s}.pt'
               for s in range(2031, 2036)},
        },
        'reuse': {
            2026: 'problem2_retrain_v2/results/unaligned/full_runs/gate_B0_test252_mean3/seed2026',
            2031: 'problem2_retrain_v2/results/unaligned/full_runs/gate_B0_test252_mean3/seed2031',
            2033: 'problem2_retrain_v2/results/unaligned/full_runs/gate_B0_test252_mean3/seed2033',
        },
    },
}


def pct(x):
    return f'{float(x):.2%}'


def link_or_copy(src, dest):
    src, dest = Path(src), Path(dest)
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        dest.symlink_to(src.resolve())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            shutil.copy2(src, dest)


def ensure_reuse(branch, root):
    for seed, src in BRANCH[branch]['reuse'].items():
        src = path(src)
        scen = src / 'test_scenarios' / 'local_scenarios.csv'
        if not scen.exists() or len(pd.read_csv(scen)) != 252:
            continue
        dest = root / f'seed{seed}'
        if dest.exists():
            continue
        link_or_copy(src, dest)
        meta_path = dest / 'run_meta.json'
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        else:
            meta = {}
        meta.update({'seed': seed, 'branch': branch, 'reused_from': str(src)})
        # ensure full/local30 json exist for aggregate
        if not (dest / 'test_full.json').exists() and (src / 'test_full.json').exists():
            pass  # symlink covers
        json_write(dest / 'run_meta.json', meta)
        print(f'REUSE {branch} seed={seed} <- {src}', flush=True)


def run_one(branch, seed, device, root):
    out = root / f'seed{seed}'
    done = out / 'test_scenarios' / 'local_scenarios.csv'
    if done.exists() and len(pd.read_csv(done)) == 252 and (out / 'test_full.json').exists():
        print(f'SKIP {branch} seed={seed}', flush=True)
        return
    ckpt = BRANCH[branch]['ckpt_dir'][seed]
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
    json_write(out / 'run_meta.json', {
        'branch': branch, 'seed': seed, 'checkpoint': ckpt,
        'checkpoint_sha256': digest,
        'full_accuracy': full['accuracy'], 'local30_accuracy': local30['accuracy'],
        'device': device,
    })
    print(f'DONE {branch} seed={seed}', flush=True)


def aggregate(branch, root):
    seeds = list(SEEDS)
    full_rows, local30_rows, scen_frames = [], [], []
    for seed in seeds:
        out = root / f'seed{seed}'
        full = json.loads((out / 'test_full.json').read_text())
        local30 = json.loads((out / 'test_local30.json').read_text())
        full_rows.append({'seed': seed, **full})
        local30_rows.append({'seed': seed, **local30})
        frame = pd.read_csv(out / 'test_scenarios' / 'local_scenarios.csv')
        frame['train_seed'] = seed
        scen_frames.append(frame)
    full_df = pd.DataFrame(full_rows)
    local30_df = pd.DataFrame(local30_rows)
    full_df.to_csv(root / 'full_by_seed.csv', index=False)
    local30_df.to_csv(root / 'local30_by_seed.csv', index=False)
    all_scen = pd.concat(scen_frames, ignore_index=True)
    all_scen.to_csv(root / 'local_scenarios_by_seed.csv', index=False)

    available = [c for c in METRIC_COLS if c in all_scen.columns]
    pos = (all_scen.groupby(['train_seed', 'modalities', 'rate', 'position'], as_index=False)[available]
           .mean())
    pos.to_csv(root / 'position_by_train_seed.csv', index=False)
    mean = pos.groupby(['modalities', 'rate', 'position'], as_index=False)[available].mean()
    std = pos.groupby(['modalities', 'rate', 'position'], as_index=False)[available].std(ddof=1)
    mean.to_csv(root / 'position_effects_mean.csv', index=False)
    std.to_csv(root / 'position_effects_std.csv', index=False)
    duration = mean.groupby(['modalities', 'rate'], as_index=False)[available].mean()
    duration.to_csv(root / 'duration_effects_mean.csv', index=False)
    overall = duration.groupby('rate', as_index=False)[available].mean()
    overall.to_csv(root / 'overall_by_rate_mean.csv', index=False)

    # 每 seed：先掩码平均→位置等权→模态等权→跨度等权，得到一颗 seed 的 252 协议综合 ACC
    per_seed_proto = []
    for seed, g in pos.groupby('train_seed'):
        dur = g.groupby(['modalities', 'rate'], as_index=False)[available].mean()
        ov = dur.groupby('rate', as_index=False)[available].mean()
        per_seed_proto.append({
            'seed': int(seed),
            'protocol_acc': float(ov.accuracy.mean()),
            'protocol_macro_f1': float(ov.macro_f1.mean()),
            'protocol_mae': float(ov.mae.mean()),
        })
    proto_df = pd.DataFrame(per_seed_proto).sort_values('seed')
    proto_df.to_csv(root / 'protocol_summary_by_seed.csv', index=False)

    fa, fs = float(full_df.accuracy.mean()), float(full_df.accuracy.std(ddof=1))
    la, ls = float(local30_df.accuracy.mean()), float(local30_df.accuracy.std(ddof=1))
    pa, ps = float(proto_df.protocol_acc.mean()), float(proto_df.protocol_acc.std(ddof=1))

    lines = [
        f'# {BRANCH[branch]["title"]}',
        '',
        f'训练种子：2026–2035 共 **{len(seeds)}** 颗；每 seed 冻评完整 252 组（同掩码协议）后跨 seed 取均值。',
        '',
        '## 1. 完整 / TAV30% / 252协议综合',
        '',
        '| 设定 | ACC均值 | ACC标准差 |',
        '|---|---:|---:|',
        f'| 完整test | {pct(fa)} | {fs:.2%} |',
        f'| TAV随机30% | {pct(la)} | {ls:.2%} |',
        f'| 252组等权综合 | {pct(pa)} | {ps:.2%} |',
        '',
        '各 seed 完整 ACC：'
        + '、'.join(f'{int(r.seed)}={r.accuracy:.2%}' for r in full_df.sort_values('seed').itertuples()),
        '',
        '各 seed 252协议综合 ACC：'
        + '、'.join(f'{int(r.seed)}={r.protocol_acc:.2%}' for r in proto_df.itertuples()),
        '',
        '## 2. 按缺失跨度汇总（十 seed 均值）',
        '',
        '| 标称跨度 | ACC | Macro-F1 | 中性F1 | MAE | Pearson |',
        '|---:|---:|---:|---:|---:|---:|',
    ]
    for r in overall.itertuples():
        lines.append(
            f'| {r.rate:.0%} | {pct(r.accuracy)} | {pct(r.macro_f1)} | {pct(r.neutral_f1)} | '
            f'{r.mae:.4f} | {r.pearson:.4f} |')
    lines += ['', '## 3. 模态 × 跨度 ACC', '',
              '| 模态 | 10% | 20% | 30% | 40% | 50% | 70% |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name in ('T', 'A', 'V', 'TA', 'TV', 'AV', 'TAV'):
        cells = []
        for rate in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7):
            row = duration[(duration.modalities == name) & np.isclose(duration.rate, rate)]
            cells.append(pct(row.accuracy.iloc[0]) if len(row) else '—')
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    (root / '全部结果表.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    json_write(root / 'summary.json', {
        'branch': branch, 'seeds': seeds, 'title': BRANCH[branch]['title'],
        'full_accuracy_mean': fa, 'full_accuracy_std': fs,
        'local30_accuracy_mean': la, 'local30_accuracy_std': ls,
        'protocol252_accuracy_mean': pa, 'protocol252_accuracy_std': ps,
        'n_scenarios_per_seed': 252,
    })
    print(f'AGGREGATED {branch}: full={fa:.2%}±{fs:.2%} local30={la:.2%}±{ls:.2%} '
          f'protocol252={pa:.2%}±{ps:.2%}', flush=True)


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
        ensure_reuse(branch, root)
        if not args.aggregate_only:
            for seed in SEEDS:
                run_one(branch, seed, args.device, root)
        aggregate(branch, root)


if __name__ == '__main__':
    main()
