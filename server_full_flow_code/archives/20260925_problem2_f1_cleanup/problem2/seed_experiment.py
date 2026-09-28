"""float32五随机种子对照；固定数据与局部缺失条件，逐次记录训练及test评价。"""
import argparse
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
from shared.common import path,json_write
from problem2.data import ROOT_DATA,load_split,masked_view,MissingTextEncoder,sha256
from problem2.training import train_candidate,restore,evaluate


def typed(data,dtype):
    return {k:v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v for k,v in data.items()}


def prepare(directory,split,encoder,cfg):
    original=load_split(directory,split,np.float64)
    reference=typed(original,torch.float32)
    if split=='train':
        specs=zip(cfg['train_missing_rates'],cfg['train_missing_seeds'])
    else:specs=[(.3,cfg['evaluation_missing_seed'])]
    views=[masked_view(reference,encoder,rate,(0,1,2),'random',seed)[0] for rate,seed in specs]
    return original,views


def view_as(data,view,dtype):
    return {**data,'O':view['O'],'XT':view['XT'].to(dtype)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/precision_five_seeds.json')
    args=parser.parse_args();spec=json.loads(path(args.config).read_text())
    root=path('problem2/results/five_seeds');root.mkdir(parents=True,exist_ok=True)
    base=json.loads(path(spec['base_config']).read_text())
    protocol=root/'protocol.json'
    if not protocol.exists():
        if base.get('balance_beta',0.)!=0. or base.get('neutral_weight',1.)!=1.:
            raise ValueError('此历史入口复用未加权的2026结果；类别加权实验请使用run_problem2.py，避免混用旧结果')
        json_write(protocol,{'specification':spec,'base_config_snapshot':base,
             'started_at':datetime.now(timezone.utc).isoformat(),'test_used_for_epoch_selection':False,
             'test_reported_for_all_seeds':True,'variable':'training seed: initialization, shuffle, dropout',
             'held_constant':'train/valid/test split and continuous missing masks',
             'original_production_checkpoint_sha256':sha256(path(base['checkpoint']))})
    else:
        previous=json.loads(protocol.read_text())
        if previous['specification']!=spec:raise ValueError('恢复实验时协议不一致')
        base=previous['base_config_snapshot']
    if base.get('balance_beta',0.)!=0. or base.get('neutral_weight',1.)!=1.:
        raise ValueError('历史五种子记录仅适用于未加权目标；类别加权实验请使用run_problem2.py')
    original=json.loads(path('problem2/results/precision/comparison.json').read_text())
    torch.set_num_threads(4);encoder=MissingTextEncoder(spec['device']);records=[]
    # 先完成全部train/valid阶段，再读取额外种子的test。
    for dataset in spec['datasets']:
        directory=path(ROOT_DATA)/dataset
        raw,views=prepare(directory,'train',encoder,spec)
        vr,vv=prepare(directory,'valid',encoder,spec)
        for precision in spec['precisions']:
            dtype=getattr(torch,precision);train=typed(raw,dtype);valid=typed(vr,dtype)
            train_views=[view_as(train,v,dtype) for v in views];valid_view=view_as(valid,vv[0],dtype)
            for seed in spec['seeds']:
                name=f'{dataset}_{precision}_seed{seed}';summary=root/name/'summary.json'
                if seed==2026 and spec['reuse_verified_seed_2026']:
                    result=next(r for r in original if r['dataset']==dataset and r['precision']==precision)
                    assert sha256(path(result['checkpoint']))==result['checkpoint_sha256']
                    result={**result,'seed':2026,'reused':True}
                elif summary.exists():
                    result=json.loads(summary.read_text())
                    assert sha256(path(result['checkpoint']))==result['checkpoint_sha256']
                else:
                    cfg=base|{'name':name,'seed':seed,'precision':precision,'device':spec['device'],
                              'epochs':spec['epochs'],'patience':spec['patience'],'data_dir':str(directory),
                              'output':str((root/name).relative_to(path('.'))),
                              'checkpoint':f'AAAmodel/checkpoints/five_seeds/{name}.pt'}
                    result=train_candidate(cfg,train,valid,train_views,valid_view)
                result={**result,'run_id':name};records.append(result)
                json_write(root/'training_comparison.json',records)
        del raw,views,vr,vv,train,valid,train_views,valid_view
    old_test=json.loads(path('problem2/results/precision/test_comparison/comparison.json').read_text())
    outputs=[]
    for dataset in spec['datasets']:
        torch.set_float32_matmul_precision('highest')
        data64,views=prepare(path(ROOT_DATA)/dataset,'test',encoder,spec)
        for result in (r for r in records if r['dataset']==dataset):
            dtype=getattr(torch,result['precision']);data=typed(data64,dtype)
            partial=view_as(data,views[0],dtype)
            output=root/result['run_id'];output.mkdir(parents=True,exist_ok=True)
            if result['seed']==2026:
                test=next(r for r in old_test if r['dataset']==dataset and r['precision']==result['precision'])
                assert test['checkpoint_sha256']==result['checkpoint_sha256']
                full,missing=test['complete'],test['local30']
                correct=test['full_correct'];missing_correct=test['missing_correct']
                source=f"problem2/results/precision/test_comparison/{test['name']}"
            else:
                model,_=restore(result['checkpoint'],spec['device'])
                full,frame=evaluate(model,data,spec['device']);missing,mf=evaluate(model,partial,spec['device'])
                frame.to_csv(output/'test_full.csv',index=False);mf.to_csv(output/'test_local30.csv',index=False)
                correct=int((frame.true_class==frame['class']).sum());missing_correct=int((mf.true_class==mf['class']).sum())
                source=str(output.relative_to(path('.')));del model
            row={'dataset':dataset,'precision':result['precision'],'seed':result['seed'],
                 'valid_accuracy':result['complete']['accuracy'],'test_accuracy':full['accuracy'],
                 'test_local30_accuracy':missing['accuracy'],'test_macro_f1':full['macro_f1'],
                 'test_mae':full['mae'],'test_correct':correct,'test_local30_correct':missing_correct,
                 'checkpoint':result['checkpoint'],'checkpoint_sha256':result['checkpoint_sha256'],
                 'prediction_directory':source,'best_epoch':result['best_epoch'],
                 'epochs_run':result['epochs_run'],'elapsed_seconds':result['elapsed_seconds'],
                 'peak_gpu_bytes':result['peak_gpu_bytes']}
            outputs.append(row);json_write(root/'per_seed.json',outputs)
            print('TEST '+json.dumps(row,ensure_ascii=False),flush=True)
    frame=pd.DataFrame(outputs);frame.to_csv(root/'per_seed.csv',index=False)
    summary=frame.groupby(['dataset','precision']).agg(
        n=('seed','count'),test_mean=('test_accuracy','mean'),test_std=('test_accuracy','std'),
        test_min=('test_accuracy','min'),test_max=('test_accuracy','max'),
        missing_mean=('test_local30_accuracy','mean'),missing_std=('test_local30_accuracy','std'),
        valid_mean=('valid_accuracy','mean'),mae_mean=('test_mae','mean'),
        elapsed_mean=('elapsed_seconds','mean')).reset_index()
    assert (summary.n==5).all();summary.to_csv(root/'summary.csv',index=False)
    print('DONE',flush=True)


def report():
    root=path('problem2/results/five_seeds')
    frame=pd.read_csv(root/'per_seed.csv');summary=pd.read_csv(root/'summary.csv')
    assert len(frame)==10 and set(frame.precision)=={'float32'}
    for _,g in frame.groupby('dataset'):
        assert sorted(g.seed.tolist())==[2026,2027,2028,2029,2030]
    lines=['# float32五随机种子实验','','按用户要求停止float64扩展，只比较float32。每套输入均使用2026—2030五个训练种子；2026复用已核验记录，其余补训。缺失掩码固定，以观察训练初始化、批次顺序和dropout的波动。','','| 数据 | 完整test均值±样本标准差 | 最低～最高 | 局部30%test均值±样本标准差 |','|---|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {r.dataset} | {100*r.test_mean:.2f}% ± {100*r.test_std:.2f}个百分点 | {100*r.test_min:.2f}%～{100*r.test_max:.2f}% | {100*r.missing_mean:.2f}% ± {100*r.missing_std:.2f}个百分点 |')
    lines+=['','标准差使用ddof=1，不是置信区间。processed_float64是原数据目录名，本次输入和网络均采用float32计算；其数值已核对与processed版本等价。','','| 种子 | processed等价数据/float32 | processed_po/float32 |','|---|---:|---:|']
    for seed,g in frame.groupby('seed'):
        cells=[]
        for dataset in ('processed_float64','processed_po'):
            r=g[g.dataset==dataset].iloc[0]
            cells.append(f'{r.test_accuracy:.2%}（{int(r.test_correct)}/727）')
        lines.append(f'| {seed} | '+ ' | '.join(cells)+' |')
    best=frame.loc[frame.test_accuracy.idxmax()]
    lines+=['',f'单次最高：{best.dataset}，seed={int(best.seed)}，test={best.test_accuracy:.2%}（{int(best.test_correct)}/727）。这是单次最好值，不是五次均值，也不是集成成绩。','','训练仅使用train，epoch按valid选择；随后按用户要求报告所有test结果。test已用于方案比较，不称为全新独立外部验证。','','![五种子的测试成绩](figures/five_seeds/test_by_seed.png)','','配置：`configs/precision_five_seeds.json`。复现：`python -m problem2.seed_experiment`，支持复用已完成的训练摘要。逐次结果、均值标准差和源预测路径见 `results/five_seeds/per_seed.csv`、`summary.csv`。']
    path('problem2/五随机种子实验结果.md').write_text('\n'.join(lines)+'\n')
    from shared.plotting import pyplot
    plt=pyplot();fig,ax=plt.subplots(figsize=(8,4))
    for dataset,g in frame.groupby('dataset'):
        ax.plot(g.seed,100*g.test_accuracy,marker='o',label=dataset+' / float32')
    ax.axhline(70,color='gray',linestyle='--',label='70% target')
    ax.set(xlabel='Training seed',ylabel='Test accuracy (%)',xticks=[2026,2027,2028,2029,2030]);ax.legend();fig.tight_layout()
    dest=path('problem2/figures/five_seeds');dest.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest/'test_by_seed.png',dpi=200);plt.close(fig)


if __name__=='__main__':
    main()
    report()
