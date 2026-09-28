"""六组研究消融入口：python -m problem2.ablation --config configs/problem2_ablation.json。"""
import argparse
import json
from pathlib import Path
import sys
from datetime import datetime,timezone
import numpy as np
import torch
from shared.common import path,json_write,environment
from problem2.data import load_split,sha256
from problem2.local_missingness import synchronized_view
from problem2.training import train_candidate,restore

VARIANTS={
    'Bm':(False,False,False), 'G':(True,False,False),
    'C':(False,True,False), 'R':(False,False,True), 'F':(True,True,True),
}


def variant_config(base, variant, seed, spec, root):
    gate,consistency,reconstruction=VARIANTS[variant]
    cfg=base|{'name':f'{variant}_seed{seed}','architecture':'ablation','seed':seed,'device':spec['device'],
        'ablation_gate':gate,'ablation_reconstruction':reconstruction,
        'consistency_weight':spec['consistency_weight'] if consistency else 0.,
        'reconstruction_weight':spec['reconstruction_weight'] if reconstruction else 0.,
        'special_tokens':'none','f3_cross':False,'f3_head':'concat','reset_training_rng':True,
        'epochs':spec['epochs'],'patience':spec['patience'],'smoke_test':spec['smoke'],
        'output':str(root/'runs'/f'{variant}_seed{seed}'),
        'checkpoint':str(path('AAAmodel/checkpoints/ablations')/root.name/f'{variant}_seed{seed}.pt')}
    return cfg


def scenarios(spec):
    """主图1/3：30%全部组合；主图2：单模态的跨度曲线。随机位置重复掩码。"""
    rows=[]
    for name in ('T','A','V','TA','TV','AV','TAV'):
        rates=spec['rates'] if len(name)==1 else [spec['main_rate']]
        for rate in sorted(set(rates+[spec['main_rate']])):
            positions=('start','middle','end','random') if rate==spec['main_rate'] else ('random',)
            for position in positions:
                for seed in (spec['mask_seeds'] if position=='random' else spec['mask_seeds'][:1]):
                    rows.append({'modalities':name,'rate':rate,'position':position,'mask_seed':seed,
                                 'key':f'{name}_{rate:g}_{position}_{seed}'})
    return rows


def subset(data, limit):
    if not limit:return data
    return {k:v[:limit] if torch.is_tensor(v) or k=='id' else v for k,v in data.items()}


def validate(spec):
    if len(set(spec['seeds']))!=len(spec['seeds']) or not spec['seeds']:raise ValueError('seed必须非空且唯一')
    if any(not isinstance(s,int) or not 0<=s<2**32 for s in spec['seeds']+spec['mask_seeds']):raise ValueError('非法seed')
    if not spec['mask_seeds']:raise ValueError('必须提供缺失掩码seed')
    if any(not 0<r<1 for r in spec['rates']+[spec['main_rate']]):raise ValueError('缺失比例必须在(0,1)内')
    for k in ('consistency_weight','reconstruction_weight'):
        if not np.isfinite(spec[k]) or spec[k]<=0:raise ValueError(f'{k}必须为正数')
    if spec['epochs']<1 or spec['patience']<1:raise ValueError('训练预算必须为正')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/problem2_ablation.json')
    parser.add_argument('--seeds',nargs='+',type=int)
    parser.add_argument('--device');parser.add_argument('--run-name')
    parser.add_argument('--resume',action='store_true',help='继续同一协议，仅跳过身份核验通过的完成项')
    parser.add_argument('--smoke',action='store_true',help='小样本一轮端到端检查；输出明确标注不可用于论文')
    parser.add_argument('--plots-only',action='store_true')
    args=parser.parse_args();spec=json.loads(path(args.config).read_text())
    for key in ('seeds','device','run_name'):
        if getattr(args,key) is not None:spec[key]=getattr(args,key)
    if not spec['run_name'].replace('_','').replace('-','').isalnum():raise ValueError('run-name只能含字母数字_-')
    if args.smoke:
        spec.update(epochs=1,patience=1,seeds=spec['seeds'][:1],mask_seeds=spec['mask_seeds'][:1],smoke=True)
        if not spec['run_name'].endswith('_smoke'):spec['run_name']+='_smoke'
    else:spec['smoke']=False
    validate(spec);torch.set_num_threads(4)
    root=path('problem2/results/ablations')/spec['run_name']
    if args.plots_only:
        from problem2.ablation_plots import plot_all
        plot_all(root);return
    base=json.loads(path(spec['base_config']).read_text());base['device']=spec['device']
    code={f:sha256(path(f)) for f in ['problem2/model.py','problem2/training.py','problem2/data.py',
          'problem2/local_missingness.py','problem2/ablation.py','problem2/ablation_analysis.py']}
    identity={'settings':spec,'base_config':base,'code':code,
              'data':{s:sha256(Path(base['data_dir'])/f'{s}.npz') for s in ('train','valid','test')},
              'initialization':sha256(path(base['init_checkpoint'])),
              'bert':sha256(path('AAAmodel/bert-base-uncased/model.safetensors')),
              'scaler':sha256(Path(base['data_dir'])/'scaler_params.npz')}
    if root.exists():
        if not args.resume:raise FileExistsError(f'{root}已存在；请换run-name或使用--resume')
        previous=json.loads((root/'protocol.json').read_text())
        if previous['identity']!=identity:raise ValueError('代码/数据/配置与保存协议不同，禁止混合实验')
    else:
        root.mkdir(parents=True)
        json_write(root/'protocol.json',{'identity':identity,'created_at':datetime.now(timezone.utc).isoformat(),
                   'environment':environment(),'B0_reuses_Bm_checkpoint':True,
                   'formal_checkpoint_sha256':sha256(path('AAAmodel/checkpoints/problem2_aligned.pt')),
                   'selection':'valid only; 0.7 full ACC + 0.3 TAV30 ACC','automatic_adoption':False})
    data={s:subset(load_split(base['data_dir'],s),48 if spec['smoke'] and s=='train' else 16 if spec['smoke'] else 0)
          for s in ('train','valid','test')}
    train_views=[]
    for name in ('T','A','V','TA','TV','AV','TAV'):
        for rate,seed in zip((.1,.3,.5),(3026,3027,3028)):
            train_views.append(synchronized_view(data['train'],rate,tuple('TAV'.index(m) for m in name),'random',seed)[0])
    valid_view,records=synchronized_view(data['valid'],spec['main_rate'],(0,1,2),'random',4026)
    json_write(root/'valid_selection_masks.json',records)
    from problem2.ablation_analysis import analyze_checkpoint,finalize
    scenario_list=scenarios(spec);json_write(root/'scenarios.json',scenario_list)
    for seed in spec['seeds']:
        for variant in VARIANTS:
            cfg=variant_config(base,variant,seed,spec,root);run=path(cfg['output'])
            summary=run/'summary.json'
            if summary.exists():
                saved=json.loads(summary.read_text())
                if sha256(path(saved['checkpoint']))!=saved['checkpoint_sha256']:raise ValueError('权重身份不一致')
            else:
                print(f'TRAIN {variant} seed={seed} smoke={spec["smoke"]}',flush=True)
                json_write(run/'config.json',cfg)
                saved=train_candidate(cfg,data['train'],data['valid'],train_views,valid_view)
            if not (run/'analysis_done.json').exists():
                model,_=restore(saved['checkpoint'],spec['device'])
                analyze_checkpoint(model,data,scenario_list,spec,variant,seed,run)
                del model
                if torch.cuda.is_available():torch.cuda.empty_cache()
            print(f'DONE {variant} seed={seed}',flush=True)
    finalize(root,spec)
    from problem2.ablation_plots import plot_all
    plot_all(root)
    protocol=json.loads((root/'protocol.json').read_text())
    assert sha256(path('AAAmodel/checkpoints/problem2_aligned.pt'))==protocol['formal_checkpoint_sha256']
    json_write(root/'completed.json',{'smoke':spec['smoke'],'seeds':spec['seeds'],
               'trained_checkpoints':5*len(spec['seeds']),'experimental_groups':6,'formal_unchanged':True})
    print(f'完成：{root}'+('（仅流程检查，不是论文结果）' if spec['smoke'] else ''),flush=True)


if __name__=='__main__':main()
