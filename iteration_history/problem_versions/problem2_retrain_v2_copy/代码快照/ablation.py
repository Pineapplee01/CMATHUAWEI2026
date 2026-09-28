"""组件拆除消融（对齐F1 / 未对齐 soft_alignment）：
python -m problem2_retrain_v2.ablation --config configs/problem2_retrain_v2_ablation.json
python -m problem2_retrain_v2.ablation --config configs/problem2_retrain_v2_ablation_unaligned.json
"""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import torch
from shared.common import path, json_write, environment
from problem2_retrain_v2.data import load_split, sha256, masked_view
from problem2_retrain_v2.local_missingness import synchronized_view
from problem2_retrain_v2.training import train_candidate, restore

VARIANTS = ('B0', 'noC', 'noQ', 'noB', 'noBeta')
_CODE_FILES = [
    'problem2_retrain_v2/model.py', 'problem2_retrain_v2/training.py', 'problem2_retrain_v2/data.py',
    'problem2_retrain_v2/local_missingness.py', 'problem2_retrain_v2/ablation.py',
    'problem2_retrain_v2/ablation_analysis.py',
]
_CKPT_ROOT = 'AAAmodel/checkpoints/ablations_retrain_v2'


def is_unaligned(base):
    return base.get('architecture') == 'soft_alignment' or base.get('data_layout') == 'unaligned_50_500_500'


def results_root(base):
    return ('problem2_retrain_v2/results/unaligned/ablations'
            if is_unaligned(base) else 'problem2_retrain_v2/results/aligned/ablations')


def variant_config(base, variant, seed, spec, root):
    unaligned = is_unaligned(base)
    cfg = dict(base)
    cfg.update({
        'name': f'{variant}_seed{seed}',
        'architecture': 'soft_alignment' if unaligned else 'f1',
        'seed': seed,
        'device': spec['device'],
        'ablation_disable_completion': False,
        'ablation_disable_reliability': False,
        'ablation_disable_span': False,
        'ablation_disable_time_penalty': False,
        'special_tokens': 'none', 'f3_cross': False, 'f3_head': 'concat',
        'reset_training_rng': True,
        'epochs': spec['epochs'], 'patience': spec['patience'], 'smoke_test': spec['smoke'],
        'output': str(root / 'runs' / f'{variant}_seed{seed}'),
        'checkpoint': str(path(_CKPT_ROOT) / root.name / f'{variant}_seed{seed}.pt'),
        'ablation_variant': variant,
        'report_local30': False,
    })
    cfg.pop('init_checkpoint', None)  # 未对齐禁止；对齐消融各组也从共享初始化由base决定
    if not unaligned and base.get('init_checkpoint'):
        cfg['init_checkpoint'] = base['init_checkpoint']
    if variant == 'B0':
        pass
    elif variant == 'noC':
        if unaligned:
            cfg['ablation_disable_span'] = True
        else:
            cfg['ablation_disable_completion'] = True
    elif variant == 'noQ':
        if unaligned:
            cfg['ablation_disable_time_penalty'] = True
        else:
            cfg['ablation_disable_reliability'] = True
    elif variant == 'noB':
        cfg['boundary_weight'] = 0.0
    elif variant == 'noBeta':
        cfg['balance_beta'] = 0.0
        cfg.pop('class_weights', None)
        cfg.pop('train_class_counts', None)
    else:
        raise ValueError(f'未知消融组: {variant}')
    return cfg


def scenarios(spec):
    rows = []
    for name in ('T', 'A', 'V', 'TA', 'TV', 'AV', 'TAV'):
        rates = spec['rates'] if len(name) == 1 else [spec['main_rate']]
        for rate in sorted(set(rates + [spec['main_rate']])):
            positions = ('start', 'middle', 'end', 'random') if rate == spec['main_rate'] else ('random',)
            for position in positions:
                for seed in (spec['mask_seeds'] if position == 'random' else spec['mask_seeds'][:1]):
                    rows.append({
                        'modalities': name, 'rate': rate, 'position': position,
                        'mask_seed': seed, 'key': f'{name}_{rate:g}_{position}_{seed}',
                    })
    return rows


def subset(data, limit):
    if not limit:
        return data
    return {k: v[:limit] if torch.is_tensor(v) or k == 'id' else v for k, v in data.items()}


def validate(spec):
    if len(set(spec['seeds'])) != len(spec['seeds']) or not spec['seeds']:
        raise ValueError('seed必须非空且唯一')
    if any(not isinstance(s, int) or not 0 <= s < 2**32 for s in spec['seeds'] + spec['mask_seeds']):
        raise ValueError('非法seed')
    if not spec['mask_seeds']:
        raise ValueError('必须提供缺失掩码seed')
    if any(not 0 < r < 1 for r in spec['rates'] + [spec['main_rate']]):
        raise ValueError('缺失比例必须在(0,1)内')
    if spec['epochs'] < 1 or spec['patience'] < 1:
        raise ValueError('训练预算必须为正')


def make_views(data, unaligned, main_rate):
    train_views = []
    if unaligned:
        for name in ('T', 'A', 'V', 'TA', 'TV', 'AV', 'TAV'):
            for rate, seed in zip((.1, .3, .5), (3026, 3027, 3028)):
                train_views.append(masked_view(
                    data['train'], None, rate, tuple('TAV'.index(m) for m in name),
                    'random', seed, reencode=False)[0])
        valid_view, records = masked_view(
            data['valid'], None, main_rate, (0, 1, 2), 'random', 4026, reencode=False)
    else:
        for name in ('T', 'A', 'V', 'TA', 'TV', 'AV', 'TAV'):
            for rate, seed in zip((.1, .3, .5), (3026, 3027, 3028)):
                train_views.append(synchronized_view(
                    data['train'], rate, tuple('TAV'.index(m) for m in name), 'random', seed)[0])
        valid_view, records = synchronized_view(data['valid'], main_rate, (0, 1, 2), 'random', 4026)
    return train_views, valid_view, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/problem2_retrain_v2_ablation.json')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--device')
    parser.add_argument('--run-name')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--plots-only', action='store_true')
    args = parser.parse_args()
    spec = json.loads(path(args.config).read_text())
    for key in ('seeds', 'device', 'run_name'):
        if getattr(args, key) is not None:
            spec[key] = getattr(args, key)
    if not spec['run_name'].replace('_', '').replace('-', '').isalnum():
        raise ValueError('run-name只能含字母数字_-')
    if args.smoke:
        spec.update(epochs=1, patience=1, seeds=spec['seeds'][:1], mask_seeds=spec['mask_seeds'][:1], smoke=True)
        if not spec['run_name'].endswith('_smoke'):
            spec['run_name'] += '_smoke'
    else:
        spec['smoke'] = False
    validate(spec)
    torch.set_num_threads(4)
    base = json.loads(path(spec['base_config']).read_text())
    base['device'] = spec['device']
    unaligned = is_unaligned(base)
    root = path(results_root(base)) / spec['run_name']
    if args.plots_only:
        from problem2_retrain_v2.ablation_plots import plot_all
        plot_all(root)
        return
    code = {f: sha256(path(f)) for f in _CODE_FILES}
    identity = {
        'settings': spec, 'base_config': base, 'code': code,
        'data': {s: sha256(Path(base['data_dir']) / f'{s}.npz') for s in ('train', 'valid', 'test')},
        'bert': sha256(path('AAAmodel/bert-base-uncased/model.safetensors')),
        'scaler': sha256(Path(base['data_dir']) / 'scaler_params.npz'),
        'protocol': ('soft_alignment knockout; noC=no-span, noQ=no-time-penalty'
                     if unaligned else 'F1 component knockout; B0=full F1; CLS/SEP P_A=P_V=0'),
    }
    if not unaligned:
        identity['initialization'] = sha256(path(base['init_checkpoint']))
    if root.exists():
        if not args.resume:
            raise FileExistsError(f'{root}已存在；请换run-name或使用--resume')
        previous = json.loads((root / 'protocol.json').read_text())
        if previous['identity'] != identity:
            raise ValueError('代码/数据/配置与保存协议不同，禁止混合实验')
    else:
        root.mkdir(parents=True)
        design = {
            'B0': '完整 soft_alignment（连续窗+时间惩罚+门控+边界+β）' if unaligned else '完整F1（补全+可靠性q+边界λ+类别平衡β）',
            'noC': '拆除连续局部窗（全局软注意）' if unaligned else '拆除观测条件补全',
            'noQ': '拆除窗内时间惩罚' if unaligned else '保留补全，拆除方差可靠性门控',
            'noB': '拆除中性边界损失',
            'noBeta': '拆除类别频数平衡',
        }
        json_write(root / 'protocol.json', {
            'identity': identity,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'environment': environment(),
            'design': design,
            'selection': 'valid only; 0.7 full ACC + 0.3 TAV30 ACC',
            'automatic_adoption': False,
            'sandbox': 'problem2_retrain_v2',
            'branch': 'unaligned' if unaligned else 'aligned',
        })
    data = {
        s: subset(load_split(base['data_dir'], s),
                  48 if spec['smoke'] and s == 'train' else 16 if spec['smoke'] else 0)
        for s in ('train', 'valid', 'test')
    }
    if not unaligned:
        I0 = data['train']['I'][0].numpy()
        P0 = data['train']['P'][0].numpy()
        for tid in (101, 102):
            m = I0 == tid
            if m.any() and (P0[1, m].any() or P0[2, m].any()):
                raise RuntimeError('problem2_retrain_v2 要求 CLS/SEP 上 P_A=P_V=0')

    train_views, valid_view, records = make_views(data, unaligned, spec['main_rate'])
    json_write(root / 'valid_selection_masks.json', records)
    from problem2_retrain_v2.ablation_analysis import analyze_checkpoint, analyze_full_only, finalize
    scenario_list = [] if unaligned else scenarios(spec)
    json_write(root / 'scenarios.json', scenario_list)
    for seed in spec['seeds']:
        for variant in VARIANTS:
            cfg = variant_config(base, variant, seed, spec, root)
            run = path(cfg['output'])
            summary = run / 'summary.json'
            if summary.exists():
                saved = json.loads(summary.read_text())
                if sha256(path(saved['checkpoint'])) != saved['checkpoint_sha256']:
                    raise ValueError('权重身份不一致')
            else:
                print(f'TRAIN {variant} seed={seed} smoke={spec["smoke"]}', flush=True)
                json_write(run / 'config.json', cfg)
                saved = train_candidate(cfg, data['train'], data['valid'], train_views, valid_view)
            if not (run / 'analysis_done.json').exists():
                model, _ = restore(saved['checkpoint'], spec['device'])
                if unaligned:
                    analyze_full_only(model, data, variant, seed, run)
                else:
                    analyze_checkpoint(model, data, scenario_list, spec, variant, seed, run)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print(f'DONE {variant} seed={seed}', flush=True)
    finalize(root, spec)
    try:
        from problem2_retrain_v2.ablation_plots import plot_all
        plot_all(root)
    except Exception as error:
        print(f'绘图跳过：{error}', flush=True)
    json_write(root / 'completed.json', {
        'smoke': spec['smoke'], 'seeds': spec['seeds'],
        'trained_checkpoints': len(VARIANTS) * len(spec['seeds']),
        'variants': list(VARIANTS),
        'branch': 'unaligned' if unaligned else 'aligned',
        'full_input_primary': True,
    })
    print(f'完成：{root}' + ('（仅流程检查，不是论文结果）' if spec['smoke'] else ''), flush=True)


if __name__ == '__main__':
    main()
