"""生成可审查的实验配置；只有 --execute 才依序启动训练和 valid 评价。"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import copy
import subprocess
import pandas as pd
from shared.common import config, path, json_write, ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/default.json')
    p.add_argument('--suite', choices=['core','sensitivity'], default='core')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args(); base = config(args.config)
    changes = [('main', {}), ('no_completion', {'ablation': 'no_completion'}),
               ('no_reliability', {'ablation': 'no_reliability'})]
    if args.suite == 'sensitivity':
        changes = [('hidden32', {'hidden':32}), ('hidden128', {'hidden':128}),
                   ('rec005', {'lambda_rec':.05}), ('rec02', {'lambda_rec':.2}),
                   ('tau05', {'tau':[.5]*3}), ('tau2', {'tau':[2.]*3}),
                   ('window3', {'window':3}), ('window10', {'window':10}),
                   ('attention_extent', {'mask_rule':'attention_extent'}),
                   ('all_slots', {'mask_rule':'all_slots'})]
    runs = []
    for name, patch in changes:
        for seed in ([2026,2027,2028] if args.suite=='core' else [2026]):
            cfg = copy.deepcopy(base); cfg.update(patch); cfg['seed'] = seed
            run = f'{name}_seed{seed}'
            cfg['checkpoint'] = f'AAAmodel/checkpoints/{run}.pt'
            cfg['output'] = f'problem2/results/{run}'
            cfg['explanation_output'] = f'problem3/results/{run}'
            filename = path(f'configs/generated/{run}.json'); json_write(filename,cfg)
            runs.append({'run':run, 'config':str(filename), 'split':'valid', 'status':'planned'})
            if args.execute:
                for action in ('train','evaluate'):
                    subprocess.run([sys.executable, str(ROOT/'problem2/problem2.py'), action,
                                    '--config', str(filename)], cwd=ROOT, check=True)
                runs[-1]['status'] = 'completed'
    dest = path('problem2/results/experiment_plans'); dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(runs).to_csv(dest / f'{args.suite}.csv', index=False)
    if args.execute:
        frames = []
        for row in runs:
            f = pd.read_csv(path(f'problem2/results/{row["run"]}/valid/结果表格.csv'))
            f['variant'] = row['run'].split('_seed')[0]
            frames.append(f)
        all_results = pd.concat(frames, ignore_index=True)
        summary = all_results.groupby(['variant','modalities','rate','position'])[
            ['accuracy','macro_f1','weighted_f1','mae','pearson']].agg(['mean','std'])
        summary.to_csv(dest / f'{args.suite}_mean_std.csv')


if __name__ == '__main__':
    main()
