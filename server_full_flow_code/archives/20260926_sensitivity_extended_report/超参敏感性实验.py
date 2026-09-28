"""当前架构的配对超参实验：只加载train/valid，固定seed，支持断点续跑。"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from problem2_retrain_v2.data import sha256

OUT = HERE / 'sensitivity_results'
SEEDS = {'aligned': [2026], 'unaligned': [2026]}
GRIDS = {'balance_beta': [0., .125, .25, .5],
         'boundary_weight': [0., .05, .1, .2],
         'bert_lr': [5e-6, 1e-5, 2e-5]}
SOFT_GRIDS = {'align_max_half_frac': [.1, .2, .3], 'align_time_penalty': [1., 2., 4.]}
CODE = ['model.py', 'training.py', 'data.py', 'local_missingness.py', 'output_contract.py', 'evaluation.py', 'ablation.py']


def write_json(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(p)


def base_config(layout):
    cfg = json.loads((ROOT / f'configs/problem2_retrain_v2_{layout}.json').read_text())
    # 清理从旧实验派生的统计量；每次只由本次train重新计算权重。
    for key in ('class_weights', 'train_class_counts', 'initialized_shared_keys'):
        cfg.pop(key, None)
    cfg.update(ablation_disable_completion=False, ablation_disable_reliability=False,
               ablation_disable_span=False, ablation_disable_time_penalty=False,
               special_tokens='none', f3_cross=False, f3_head='concat', reset_training_rng=True,
               epochs=12, patience=4, smoke_test=False, precision='float32')
    return cfg


def candidates(layout):
    base = base_config(layout)
    result = [('baseline', {})]
    for key, values in (GRIDS | (SOFT_GRIDS if layout == 'unaligned' else {})).items():
        for value in values:
            if value != base[key]:
                result.append((f'{key}_{value:g}', {key: value}))
    return result


def prepare():
    identity = {'code': {f: sha256(ROOT / 'problem2_retrain_v2' / f) for f in CODE},
                'seeds': SEEDS, 'grids': GRIDS, 'soft_grids': SOFT_GRIDS,
                'bases': {layout: base_config(layout) for layout in SEEDS},
                'data': {}, 'selection': '.7*valid_full_accuracy+.3*valid_local30_accuracy',
                'test_access': False, 'reuse_historical_runs': False,
                'tie_break': 'higher valid score; then higher full Macro-F1; then baseline',
                'design': 'single seed 2026 hyperparameter search; one factor at a time; baseline current code'}
    for layout in SEEDS:
        folder = Path(identity['bases'][layout]['data_dir'])
        identity['data'][layout] = {s: sha256(folder / f'{s}.npz') for s in ('train', 'valid')}
    p = OUT / 'protocol.json'
    if p.exists() and json.loads(p.read_text()) != identity:
        raise ValueError('实验代码、数据或协议改变，禁止混入已有结果')
    write_json(p, identity)
    return identity


def run_worker(layout, worker, workers, device, joint=False):
    import torch
    from problem2_retrain_v2.data import load_split
    from problem2_retrain_v2.ablation import make_views
    from problem2_retrain_v2.training import train_candidate
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    base = base_config(layout)
    data = {s: load_split(base['data_dir'], s) for s in ('train', 'valid')}
    views, valid_view, records = make_views(data, layout == 'unaligned', .3)
    write_json(OUT / layout / f'mask_records_worker{worker}.json', records)
    cs = [('joint', json.loads((OUT / layout / 'joint_parameters.json').read_text()))] if joint else candidates(layout)
    jobs = [(name, changes, seed) for name, changes in cs for seed in SEEDS[layout]]
    for index, (name, changes, seed) in enumerate(jobs):
        if index % workers != worker:
            continue
        run = OUT / layout / f'{name}_seed{seed}'
        cfg = base | changes | {'seed': seed, 'device': device, 'name': f'{layout}_{name}_seed{seed}',
              'output': str(run),
              'checkpoint': str(ROOT / 'AAAmodel/checkpoints/sensitivity_current' / layout / f'{name}_seed{seed}.pt')}
        config_file = run / 'config.json'
        if config_file.exists() and json.loads(config_file.read_text()) != cfg:
            raise ValueError(f'配置变化: {run}')
        write_json(config_file, cfg)
        summary_file = run / 'summary.json'
        if summary_file.exists():
            result = json.loads(summary_file.read_text())
            if result['checkpoint_sha256'] != sha256(ROOT / result['checkpoint']):
                raise ValueError(f'检查点变化: {run}')
            print(f'RESUME {run.name}', flush=True)
            continue
        print(f'START {layout} {name} seed={seed}', flush=True)
        train_candidate(cfg, data['train'], data['valid'], views, valid_view)
        gc.collect()
        torch.cuda.empty_cache()
        print(f'DONE {layout} {name} seed={seed}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout', choices=list(SEEDS))
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--aligned-device', default='cuda:2')
    parser.add_argument('--unaligned-device', default='cuda:3')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--joint-only', action='store_true')
    parser.add_argument('--joint', action='store_true')
    args = parser.parse_args()
    if args.layout:
        run_worker(args.layout, args.worker, args.workers, args.device, args.joint)
        return
    if not args.report_only:
        prepare()
        def execute(job):
            layout, worker, device = job
            log = OUT / f'{layout}_worker{worker}.log'
            with log.open('a') as stream:
                result = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
                    '--layout', layout, '--worker', str(worker), '--workers', str(args.workers),
                    '--device', device], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    env=os.environ | {'TOKENIZERS_PARALLELISM': 'false'})
            if result.returncode:
                raise RuntimeError(f'训练失败，请检查 {log}')
        jobs = [(layout, i, args.aligned_device if layout == 'aligned' else args.unaligned_device)
                for layout in SEEDS for i in range(args.workers)]
        if not args.joint_only:
            with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                list(pool.map(execute, jobs))
        from 超参敏感性绘图 import collect, preferred_parameters
        frame = collect(include_joint=False)
        for layout in SEEDS:
            write_json(OUT / layout / 'joint_parameters.json', preferred_parameters(frame, layout))
        def execute_joint(job):
            layout, worker, device = job
            with (OUT / f'{layout}_joint_worker{worker}.log').open('a') as stream:
                result = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
                    '--layout', layout, '--worker', str(worker), '--workers', str(args.workers),
                    '--device', device, '--joint'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    env=os.environ | {'TOKENIZERS_PARALLELISM': 'false'})
            if result.returncode:
                raise RuntimeError(f'联合复核失败: {layout} worker{worker}')
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            list(pool.map(execute_joint, jobs))
    from 超参敏感性绘图 import report
    report()


if __name__ == '__main__':
    main()
