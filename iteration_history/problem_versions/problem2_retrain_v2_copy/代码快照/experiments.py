"""正式架构训练实验模块；由根目录run_problem2.py --mode train调用。"""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import torch
from shared.common import path, json_write, environment
from problem2_retrain_v2.data import ALLOWED, ROOT_DATA, load_split, masked_view, MissingTextEncoder, sha256
from problem2_retrain_v2.training import train_candidate, restore, evaluate
from problem2_retrain_v2.model import validate_neutral_weight, validate_balance_beta, boundary_settings


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.stream.write(value); self.log.write(value); self.log.flush()
        return len(value)

    def flush(self):
        self.stream.flush(); self.log.flush()


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', '--seed', nargs='+', type=int, default=[2026,2027,2028,2029,2030])
    parser.add_argument('--datasets', nargs='+', choices=ALLOWED, default=None)
    parser.add_argument('--config', default='configs/problem2_bert_finetune.json', help='架构/训练基础配置，不加载已有训练权重')
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--neutral-weight', type=float, default=None,
                        help='频数平衡后的额外中性倍率；覆盖config，默认1.0')
    parser.add_argument('--balance-beta', type=float, default=None,
                        help='频数平衡强度：0关闭、1逆频数；覆盖config，新实验默认1')
    for key in boundary_settings({}):
        parser.add_argument('--'+key.replace('_','-'),type=int if key=='boundary_warmup_epochs' else float,
                            default=None,help='中性边界训练参数；覆盖config，boundary-weight默认0关闭')
    parser.add_argument('--bert-lr',type=float,default=None,help='微调BERT的独立学习率')
    parser.add_argument('--bert-trainable-layers',type=int,default=None,help='微调顶部层数，BERT-base为1至12层')
    parser.add_argument('--run-name', help='可选实验名；相同名字已存在时拒绝覆盖')
    parser.add_argument('--resume', help='恢复已有实验目录，例如problem2_retrain_v2/results/aligned/runs/my_run 或 .../unaligned/runs/my_run')
    args = parser.parse_args(argv)
    if args.bert_lr is not None and (not np.isfinite(args.bert_lr) or args.bert_lr<=0):parser.error('bert-lr必须为有限正数')
    if args.bert_trainable_layers is not None and not 1<=args.bert_trainable_layers<=12:parser.error('bert-trainable-layers须为1至12')
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= s < 2**32 for s in args.seeds):
        parser.error('seed必须互不重复且位于0到4294967295')
    if args.datasets is not None and len(set(args.datasets)) != len(args.datasets): parser.error('数据版本不能重复')
    if any(v is not None and v<1 for v in (args.epochs,args.patience,args.batch_size)) or (args.lr is not None and (not np.isfinite(args.lr) or args.lr<=0)):
        parser.error('epochs/patience/batch-size/lr必须为正')
    if args.run_name and not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_name):
        parser.error('run-name仅允许英文字母、数字、下划线和连字符')
    if args.resume and args.run_name: parser.error('resume不能同时指定run-name')
    if args.neutral_weight is not None:
        try: validate_neutral_weight(args.neutral_weight)
        except ValueError as error: parser.error(str(error))
    if args.balance_beta is not None:
        try: validate_balance_beta(args.balance_beta)
        except ValueError as error: parser.error(str(error))
    try: boundary_settings({k:getattr(args,k) for k in boundary_settings({}) if getattr(args,k) is not None})
    except ValueError as error: parser.error(str(error))
    return args


def training_settings(args, base):
    """预算与输入版本遵循CLI > 配置 > 历史缺省，防止微调配置被旧默认值覆盖。"""
    defaults={'epochs':40,'patience':10,'batch_size':64,'lr':.0005}
    result={k:getattr(args,k) if getattr(args,k) is not None else base.get(k,v) for k,v in defaults.items()}
    result['datasets']=args.datasets if args.datasets is not None else [Path(base.get('data_dir','processed_all_zscore')).name]
    if any(d not in ALLOWED for d in result['datasets']):raise ValueError('配置数据版本不在允许范围内')
    if any(not isinstance(result[k],int) or result[k]<1 for k in ('epochs','patience','batch_size')):
        raise ValueError('训练轮数、patience和batch_size必须为正整数')
    if not np.isfinite(result['lr']) or result['lr']<=0:raise ValueError('学习率必须为有限正数')
    return result


def prepare(directory, split, encoder, protocol):
    data = load_split(directory, split, dtype=np.float32)
    specs = zip(protocol['missing_rates'],protocol['missing_seeds']) if split=='train' else [(.3,protocol['evaluation_missing_seed'])]
    views = [masked_view(data,encoder,rate,(0,1,2),'random',seed,
                         reencode=not protocol['base_config'].get('finetune_bert',False))[0] for rate,seed in specs]
    return data, views


def balance_fields(record):
    """历史实验缺省为无频数平衡，不把未知训练类数伪造为当前数据类数。"""
    beta=validate_balance_beta(record.get('balance_beta',0.0))
    neutral=validate_neutral_weight(record.get('neutral_weight',1.0))
    if beta>0 and ('class_weights' not in record or 'train_class_counts' not in record):
        raise ValueError('平衡训练记录缺少实际类别权重或训练类数')
    return {'balance_beta':beta,'neutral_weight':neutral,
            'train_class_counts':record.get('train_class_counts'),
            'class_weights':record.get('class_weights',[1.,neutral,1.])}


def check_balance(record, spec):
    if record.get('neutral_weight',1.0)!=spec['neutral_weight']:
        raise ValueError('恢复记录与中性权重不匹配')
    if record.get('balance_beta',0.0)!=spec['balance_beta']:
        raise ValueError('恢复记录与类别平衡强度不匹配')
    if boundary_settings(record)!=boundary_settings(spec):
        raise ValueError('恢复记录与中性边界训练参数不匹配')
    return balance_fields(record)


def summarize(rows, root):
    # 旧实验未设置该超参数，兼容读取时明确标为1.0。
    rows=[row|balance_fields(row)|boundary_settings(row) for row in rows]
    frame = pd.DataFrame(rows)
    for i,label in enumerate(('negative','neutral','positive')):
        frame[f'weight_{label}']=[row['class_weights'][i] for row in rows]
    frame.to_csv(root/'results.csv',index=False)
    json_write(root/'results.json',rows)
    summary_kwargs=dict(seed_count=('seed','count'),test_accuracy_mean=('test_accuracy','mean'),
        test_accuracy_std=('test_accuracy','std'),test_accuracy_max=('test_accuracy','max'))
    if 'test_local30_accuracy' in frame.columns and frame['test_local30_accuracy'].notna().any():
        summary_kwargs['local30_accuracy_mean']=('test_local30_accuracy','mean')
        report_local30=True
    else:
        report_local30=False
    summary = frame.groupby(['dataset','precision','balance_beta','neutral_weight',*boundary_settings({})],as_index=False).agg(**summary_kwargs)
    # 单seed不能估计样本标准差：CSV留空，不伪造为0。
    summary.to_csv(root/'summary.csv',index=False)
    if report_local30:
        header='| 数据 | seed | beta | 中性倍率 | 边界λ | 实际三类权重 | 最佳epoch | 验证Acc | 测试Acc | 测试中性F1 | 局部30%测试Acc |'
        sep='|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|'
    else:
        header='| 数据 | seed | beta | 中性倍率 | 边界λ | 实际三类权重 | 最佳epoch | 验证Acc | 测试Acc | 测试中性F1 |'
        sep='|---|---:|---:|---:|---:|---|---:|---:|---:|---:|'
    lines=['# 本次实验结果','','权重顺序为负、中、正。仅由train标签统计；test在按valid选好epoch后计算。边界λ是配置最大值，最佳轮次的有效值见结果JSON。','',header,sep]
    for row in rows:
        neutral_f1='—' if row.get('test_neutral_f1') is None else f"{row['test_neutral_f1']:.2%}"
        weights=', '.join(f'{w:.4f}' for w in row['class_weights'])
        base=f"| {row['dataset']} | {row['seed']} | {row['balance_beta']:g} | {row['neutral_weight']:g} | {row['boundary_weight']:g} | {weights} | {row['best_epoch']} | {row['valid_accuracy']:.2%} | {row['test_accuracy']:.2%} | {neutral_f1} |"
        if report_local30:
            local=row.get('test_local30_accuracy')
            base+=f" {'—' if local is None else f'{local:.2%}'} |"
        lines.append(base)
    lines += ['', '完整指标与对应检查点路径见 results.csv；跨seed样本标准差使用ddof=1，单seed留空。此脚本不自动替换正式模型。']
    (root/'结果说明.md').write_text('\n'.join(lines)+'\n')


def run(args):
    torch.set_num_threads(4)
    if args.resume:
        root=path(args.resume)
        protocol=json.loads((root/'protocol.json').read_text())
        if protocol.get('runner') not in ('run_problem2_v1','run_problem2_v2','run_problem2_v3'):
            raise ValueError('只能恢复本入口生成的实验')
        base=protocol['base_config'];spec=protocol['settings']
        if protocol['runner']=='run_problem2_v1': spec=spec|{'neutral_weight':1.0}
        if protocol['runner']!='run_problem2_v3': spec=spec|{'balance_beta':0.0}
        spec['neutral_weight']=validate_neutral_weight(spec['neutral_weight'])
        spec['balance_beta']=validate_balance_beta(spec['balance_beta'])
        spec=spec|boundary_settings(spec)
        print('恢复保存的seed/数据/设备/训练协议；忽略本次其他训练参数。',flush=True)
    else:
        name=args.run_name or datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        base=json.loads(path(args.config).read_text())
        for key in ('bert_lr','bert_trainable_layers'):
            if getattr(args,key) is not None:base[key]=getattr(args,key)
        neutral_weight=validate_neutral_weight(
            args.neutral_weight if args.neutral_weight is not None else base.get('neutral_weight',1.0))
        balance_beta=validate_balance_beta(
            args.balance_beta if args.balance_beta is not None else base.get('balance_beta',1.0))
        boundary=boundary_settings(base|{k:getattr(args,k) for k in boundary_settings({}) if getattr(args,k) is not None})
        root=path('problem2_retrain_v2/results/unaligned/runs' if (
            base.get('architecture')=='soft_alignment' or base.get('data_layout')=='unaligned_50_500_500')
            else 'problem2_retrain_v2/results/aligned/runs')/name
        root.mkdir(parents=True,exist_ok=False)
        spec={k:getattr(args,k) for k in ('seeds','device')}|training_settings(args,base)
        spec['neutral_weight']=neutral_weight
        spec['balance_beta']=balance_beta
        spec.update(boundary)
        protocol={'runner':'run_problem2_v3','run_id':name,'created_at':datetime.now(timezone.utc).isoformat(),
             'settings':spec,'base_config':base,'precision':'float32',
             'missing_rates':[.1,.3,.5],'missing_seeds':[3026,3027,3028],'evaluation_missing_seed':4026,
             'epoch_selection':'0.7 valid full accuracy + 0.3 valid TAV30 accuracy',
             'automatic_production_replacement':False,'environment':environment()}
        json_write(root/'protocol.json',protocol)
    device=spec['device'];rows=[]
    neutral_weight=spec['neutral_weight']
    checkpoint_root=path('AAAmodel/checkpoints/runs')/protocol['run_id']
    encoder=None
    with (root/'run.log').open('a',encoding='utf-8') as log, redirect_stdout(Tee(sys.stdout,log)):
        print(f"实验目录：{root}\n训练seeds：{spec['seeds']}；balance_beta={spec['balance_beta']:g}；中性额外倍率：{neutral_weight:g}；边界参数：{boundary_settings(spec)}；计算精度：float32",flush=True)
        for dataset in spec['datasets']:
            directory=path(ROOT_DATA)/dataset
            train=valid=train_views=valid_views=test=test_views=None
            for seed in spec['seeds']:
                name=f'{dataset}_seed{seed}';out=root/name;finished=out/'experiment_result.json'
                if finished.exists():
                    result=json.loads(finished.read_text())
                    if result['seed']!=seed or result['dataset']!=dataset:raise ValueError('恢复记录与seed/数据不匹配')
                    balance=check_balance(result,spec)
                    saved_balance=protocol.get('class_balance',{}).get(dataset)
                    if saved_balance is not None and balance!=saved_balance:
                        raise ValueError('完成记录与实验协议的实际类别权重/类数不匹配')
                    result.update(balance)
                    if sha256(path(result['checkpoint']))!=result['checkpoint_sha256']:raise ValueError('恢复模型哈希不匹配')
                    print(f'已完成，复用 {name}',flush=True)
                else:
                    if encoder is None and not base.get('finetune_bert',False): encoder=MissingTextEncoder(device)
                    saved_summary=out/'summary.json'
                    if saved_summary.exists():
                        fitted=json.loads(saved_summary.read_text())
                        if fitted['seed']!=seed or sha256(path(fitted['checkpoint']))!=fitted['checkpoint_sha256']:
                            raise ValueError('训练摘要身份不匹配')
                        check_balance(fitted,spec)
                    else:
                        if train is None:
                            train,train_views=prepare(directory,'train',encoder,protocol)
                            valid,valid_views=prepare(directory,'valid',encoder,protocol)
                        cfg=base|{'name':name,'seed':seed,'data_dir':str(directory),'precision':'float32',
                             'device':device,'epochs':spec['epochs'],'patience':spec['patience'],
                             'batch_size':spec['batch_size'],'lr':spec['lr'],
                             'neutral_weight':neutral_weight,
                             'balance_beta':spec['balance_beta'],
                             **boundary_settings(spec),
                             'output':str(out.relative_to(path('.'))),'checkpoint':str((checkpoint_root/(name+'.pt')).relative_to(path('.')))}
                        fitted=train_candidate(cfg,train,valid,train_views,valid_views[0])
                    balance=check_balance(fitted,spec)
                    saved_balance=protocol.get('class_balance',{}).get(dataset)
                    if saved_balance is not None and balance!=saved_balance:
                        raise ValueError('训练摘要与实验协议的实际类别权重/类数不匹配')
                    protocol.setdefault('class_balance',{})[dataset]=balance
                    json_write(root/'protocol.json',protocol)
                    torch.set_float32_matmul_precision('highest')
                    if test is None:test,test_views=prepare(directory,'test',encoder,protocol)
                    model,trained_config=restore(fitted['checkpoint'],device)
                    if check_balance(trained_config,spec)!=balance:
                        raise ValueError('检查点与训练摘要的类别权重/类数不匹配')
                    full,frame=evaluate(model,test,device)
                    soft=trained_config.get('architecture')=='soft_alignment'
                    if soft:
                        from problem2_retrain_v2.inference import evaluate_scenarios,export_soft_alignment
                        evaluate_scenarios(model,test,None,device,out,figures=out/'figures',checkpoint=fitted['checkpoint'])
                        export_soft_alignment(model,test,out/'figures',device)
                        shared=path('problem2_retrain_v2/figures/unaligned')/protocol['run_id']
                        export_soft_alignment(model,test,shared,device)
                        exports=[('test_full',frame)]; missing=None
                    else:
                        missing,mf=evaluate(model,test_views[0],device)
                        exports=[('test_full',frame),('test_local30',mf)]
                    for label,predictions in exports:
                        predictions.insert(0,'seed',seed);predictions.insert(1,'dataset',dataset)
                        predictions.insert(2,'neutral_weight',neutral_weight)
                        predictions['balance_beta']=balance['balance_beta']
                        for key,value in boundary_settings(spec).items():predictions[key]=value
                        for i,weight_label in enumerate(('negative','neutral','positive')):
                            predictions[f'weight_{weight_label}']=balance['class_weights'][i]
                        predictions.to_csv(out/f'{label}_seed{seed}.csv',index=False)
                    # 训练器生成的valid逐样本表也补充seed，便于单独导出后辨认。
                    valid_files=(out/'valid_full.csv',) if soft else (out/'valid_full.csv',out/'valid_local30.csv')
                    for file in valid_files:
                        if file.exists():
                            v=pd.read_csv(file);v['seed']=seed;v['dataset']=dataset
                            v['neutral_weight']=neutral_weight;v['balance_beta']=balance['balance_beta']
                            for key,value in boundary_settings(spec).items():v[key]=value
                            for i,weight_label in enumerate(('negative','neutral','positive')):
                                v[f'weight_{weight_label}']=balance['class_weights'][i]
                            v.to_csv(file,index=False)
                    result={'run_id':protocol['run_id'],'dataset':dataset,'seed':seed,'precision':'float32',
                       **balance,
                       'finetune_bert':base.get('finetune_bert',False),'bert_lr':base.get('bert_lr'),
                       'bert_trainable_layers':base.get('bert_trainable_layers',12) if base.get('finetune_bert',False) else 0,
                       **boundary_settings(spec),'effective_boundary_weight':fitted.get('effective_boundary_weight',0.),
                       'best_epoch':fitted['best_epoch'],'epochs_run':fitted['epochs_run'],
                       'valid_accuracy':fitted['complete']['accuracy'],'test_accuracy':full['accuracy'],
                       'test_macro_f1':full['macro_f1'],
                       **{f'test_{key}':full[key] for key in ('neutral_precision','neutral_recall','neutral_f1')},
                       'test_mae':full['mae'],'test_pearson':full['pearson'],
                       'test_correct':int((frame.true_class==frame['class']).sum()),'test_samples':len(frame),
                       'checkpoint':fitted['checkpoint'],'checkpoint_sha256':fitted['checkpoint_sha256']}
                    if missing is not None:
                        result.update({'test_local30_accuracy':missing['accuracy'],
                                       **{f'test_local30_{key}':missing[key] for key in ('neutral_precision','neutral_recall','neutral_f1')}})
                    json_write(finished,result);del model
                rows.append(result);summarize(rows,root)
                msg=f"RESULT dataset={dataset} seed={seed} beta={balance['balance_beta']:g} neutral_weight={neutral_weight:g} class_weights={balance['class_weights']} test_acc={result['test_accuracy']:.2%}"
                if 'test_local30_accuracy' in result:
                    msg+=f" local30_acc={result['test_local30_accuracy']:.2%}"
                print(msg,flush=True)
            del train,valid,train_views,valid_views,test,test_views
        print(f'全部完成：{root}/results.csv',flush=True)
    return root


