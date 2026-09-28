"""仅按预先定义的valid指标冻结候选；消融不参与主方案选择。"""
import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
import pandas as pd
import torch
from shared.common import ROOT, path, json_write

CANDIDATES = ('baseline', 'algorithm_v1', 'task_adaptation', 'semantic_readout',
    'bias_tuning', 'last_layers', 'mixup', 'full_task_tuning', 'full_depth_lora',
    'sentiment_transfer', 'sentiment_global', 'context_fusion', 'context_sentiment',
    'neutral_transfer', 'neutral_adaptation', 'neutral_global', 'neutral_ema',
    'neutral_seed2027', 'neutral_seed2028', 'neutral_temporal', 'neutral_contrastive',
    'neutral_triple')


def archive_previous():
    archive=path('AAAmodel/checkpoints/archive/main_round1.pt')
    if archive.exists(): return
    archive.parent.mkdir(exist_ok=True)
    shutil.copyfile(path('AAAmodel/checkpoints/main.pt'),archive)
    previous=json.loads(path('configs/final.json').read_text())
    previous.update(checkpoint='AAAmodel/checkpoints/archive/main_round1.pt',
                    output='problem2/results/round1',explanation_output='problem3/results/round1')
    json_write('configs/archive/round1.json',previous)
    for root in ('problem2/results','problem3/results','problem2/figures','problem3/figures'):
        source=path(root)/'main';destination=path(root)/'round1'
        if source.exists():source.rename(destination)
    source=path('problem2/results/optimization/frozen_selection.json')
    shutil.copyfile(source,source.with_name('frozen_selection_round1.json'))
    report=path('运行与优化报告.md')
    if report.exists():
        path('运行与优化报告_第一轮.md').write_text(
            '第一轮历史记录：其中main结果现归档至problem2/3/results/round1；当前结果见运行与优化报告.md。\n\n'+report.read_text())



def freeze():
    rows = []
    for name in CANDIDATES:
        history = pd.read_csv(path(f'problem2/results/{name}/training_history.csv'))
        best = history.loc[history.selection_score.idxmin()]
        run = json.loads(path(f'problem2/results/{name}/run_config.json').read_text())
        if len(history) < run['config']['epochs'] and len(history)-1-int(best.epoch) < run['config']['patience']:
            raise RuntimeError(f'{name} 尚未结束，不能冻结')
        rows.append({'run': name, 'epochs_completed':len(history), 'best_epoch':int(best.epoch),
                     'trainable_parameters':run['trainable_parameters'], **best.to_dict()})
    table = pd.DataFrame(rows)
    chosen = table.sort_values('selection_score').iloc[0]['run']
    source = path(f'AAAmodel/checkpoints/{chosen}.pt')
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    cfg = dict(checkpoint['config'], checkpoint='AAAmodel/checkpoints/main.pt',
               output='problem2/results/main', explanation_output='problem3/results/main')
    checkpoint['config'] = cfg
    archive_previous()
    target = path(cfg['checkpoint']); torch.save(checkpoint, target)
    json_write('configs/default.json', cfg)
    json_write('configs/final.json', cfg)
    directory = path('problem2/results/optimization'); directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(directory/'comparison.csv', index=False)
    report = {'frozen_at_utc':datetime.now(timezone.utc).isoformat(), 'selected':chosen,
              'criterion':'valid complete three-class accuracy; missing accuracy breaks ties',
              'candidates':list(CANDIDATES), 'test_used_for_selection':False,
              'previous_round_test_result_known':True, 'round':2,
              'checkpoint_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
              'valid_accuracy':float(table.set_index('run').loc[chosen,'full_accuracy']),
              'target_accuracy':.7, 'target_met':bool(table.set_index('run').loc[chosen,'full_accuracy']>=.7)}
    json_write(directory/'frozen_selection.json', report)
    output = path(cfg['output']); output.mkdir(exist_ok=True)
    for name in ('training_history.csv','run_config.json'):
        shutil.copyfile(path(f'problem2/results/{chosen}/{name}'),output/f'selected_{name}')
    with zipfile.ZipFile(output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for folder in ('problem1','problem2','problem3','shared','tests'):
            for f in (ROOT/folder).glob('*.py'): z.write(f,f.relative_to(ROOT))
        z.write(path('configs/final.json'),'configs/final.json')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',action='store_true',required=True,help='所有候选训练已结束后冻结')
    parser.parse_args(); freeze()
