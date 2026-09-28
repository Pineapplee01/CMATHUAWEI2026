"""冻结模型的局部缺失场景分析与附件3推理。"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from shared.common import path, json_write, LABELS
from shared.data import load_fields
from problem2.data import masked_view, MissingTextEncoder, sha256
from problem2.training import evaluate, restore
from problem2.local_missingness import scenario_grid
from problem2.output_contract import project_intensity


def evaluate_scenarios(model,data,encoder,device,output,figures=None,checkpoint=None):
    """test完整252场景；valid保留168场景。不同划分独立保存，防止互相覆盖。"""
    split=data['split']
    if split not in ('test','valid'):raise ValueError('场景分析仅支持valid/test')
    rates=(.1,.2,.3,.4,.5,.7) if split=='test' else (.1,.3,.5,.7)
    seeds=(2026,2027,2028);grid=list(scenario_grid(seeds=seeds,rates=rates))
    out=Path(output)/f'{split}_scenarios';root=out/'predictions';root.mkdir(parents=True,exist_ok=True)
    identity=sha256(path(checkpoint)) if checkpoint else None
    protocol={'status':'running','scenarios_expected':len(grid),'samples_per_scenario':len(data['id']),
              'split':split,'rates':list(rates),'mask_seeds':list(seeds),
              'model_seed':model.cfg.get('seed'),'checkpoint':checkpoint,'checkpoint_sha256':identity,
              'text_missingness':'before BERT encoding; original slot indices retained',
              'finetune_bert':model.cfg.get('finetune_bert',False),'strict_partial':True,
              'positions':'slot intervals; not seconds',
              'joint_interval_policy':'per-modality valid axes; random starts sampled independently',
              'aggregation':'mean random repeats first; equal weight for four positions and seven modality combinations',
              'selection':'frozen checkpoint; no epoch selection from scenario scores'}
    json_write(out/'local_protocol.json',protocol);rows=[]
    for index,(name,modalities,rate,position,seed) in enumerate(grid,1):
        partial,records=masked_view(data,encoder,rate,modalities,position,seed,reencode=not model.cfg.get('finetune_bert',False))
        score,frame=evaluate(model,partial,device)
        key=f'{name}_{rate:.1f}_{position}_{seed}'
        frame.to_csv(root/f'{key}.csv',index=False)
        masks=pd.DataFrame(records)
        if not (masks.loc[masks.observed_before>0,'observed_after']>=1).all():
            raise ValueError(f'{key}:局部缺失删空了原非空模态')
        masks.to_csv(root/f'{key}_mask.csv',index=False)
        rows.append({'split':split,'modalities':name,'rate':rate,'position':position,
                     'seed':seed,'model_seed':model.cfg.get('seed'),
                     'actual_removed_fraction':float(masks.actual_removed_fraction.mean()),**score})
        if index%12==0 or index==len(grid):
            print(f'{split} scenarios {index}/{len(grid)}: {key} ACC={score["accuracy"]:.2%}',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(out/'local_scenarios.csv',index=False)
    metric_names=list(score)
    columns=metric_names+['actual_removed_fraction']
    # 不把三个随机位置重复当成三个独立位置，也不当作三个训练seed。
    positions=frame.groupby(['modalities','rate','position'],as_index=False)[columns].mean()
    positions.to_csv(out/'position_effects.csv',index=False)
    duration=positions.groupby(['modalities','rate'],as_index=False)[columns].mean()
    duration.to_csv(out/'duration_effects.csv',index=False)
    duration.groupby('rate',as_index=False)[columns].mean().to_csv(out/'overall_by_rate.csv',index=False)
    from shared.plotting import pyplot
    plt=pyplot();fig,axes=plt.subplots(1,3,figsize=(13,3.7))
    for ax,metric,title in zip(axes,('accuracy','macro_f1','mae'),('Accuracy','Macro-F1','Intensity MAE')):
        for name,g in duration.groupby('modalities'):ax.plot(g.rate,g[metric],marker='o',label=name)
        ax.set(xlabel='Missing span / valid span',ylabel=title,title=f'{split}: {title}');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.tight_layout()
    dest=path(figures or 'problem2/figures/aligned')/split;dest.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(dest/f'local_missingness.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    if checkpoint and sha256(path(checkpoint))!=identity:raise ValueError('评估期间权重文件发生改变')
    json_write(out/'local_protocol.json',protocol|{'status':'completed','scenarios':len(rows)})
    (out/'README.md').write_text(
        f'# {split}连续缺失场景评估\n\n共{len(rows)}组，每组{len(data["id"])}条样本。'
        f'训练seed={model.cfg.get("seed")}；随机缺失掩码seed={list(seeds)}。\n\n'
        'local_scenarios.csv保存各场景全部指标；position_effects.csv先合并随机位置重复；'
        'duration_effects.csv对四种位置等权平均；overall_by_rate.csv再对七种模态组合等权平均。\n\n'
        'predictions/保存逐样本预测与实际连续缺失区间。跨度是槽位比例；短/稀疏序列会缩短区间以保留观测，'
        '实际删去的观测比例另存，不能等同于标称跨度。多模态随机起点独立抽取，不宣称同步缺失。\n\n'
        '完整输入及原30%参考指标位于上级目录，未参与本文件的缺失场景平均。'
        '指标基于冻结模型，未根据test场景重新选择epoch。\n',encoding='utf-8')
    return frame


def raw_to_inputs(fields, directory, encoder):
    """仅对没有标准化文件的专项原始数据应用一次已保存的train scaler。"""
    ids=np.asarray(fields['text_bert'])[:,0].astype(np.int64)
    attention=np.asarray(fields['text_bert'])[:,1].astype(bool)
    oa=np.any(np.asarray(fields['audio'])!=0,axis=-1)
    ov=np.any(np.asarray(fields['vision'])!=0,axis=-1)
    if Path(directory).name=='processed_po':
        support=attention|oa|ov
        extent=np.arange(50)[None]<=np.where(support,np.arange(50),-1).max(-1,keepdims=True)
    else:extent=attention.copy()
    P=np.repeat(extent[:,None],3,axis=1)
    P[:,0]&=~np.isin(ids,[0,101,102])
    O=np.stack([attention&~np.isin(ids,[0,101,102]),oa,ov],1)&P
    out={'P':torch.from_numpy(P),'O':torch.from_numpy(O),'I':torch.from_numpy(ids)}
    scaler=Path(directory)/'scaler_params.npz'
    out['XT']=(encoder.encode(torch.from_numpy(ids),torch.from_numpy(O[:,0]),scaler) if encoder is not None
               else torch.zeros(len(ids),50,768))
    with np.load(scaler) as z:
        for m,(name,key) in enumerate((('audio','A'),('vision','V')),1):
            standardized=(np.asarray(fields[name],dtype=np.float64)-z[f'mu_{key}'])/z[f'sigma_{key}']
            standardized[~O[:,m]]=0
            out['X'+key]=torch.from_numpy(standardized.astype(np.float32))
    return out


@torch.no_grad()
def predict_appendix3(cfg,device):
    model,trained_cfg=restore(cfg['checkpoint'],device)
    cfg=cfg|{'data_dir':trained_cfg['data_dir']}
    encoder=None if trained_cfg.get('finetune_bert',False) else MissingTextEncoder(device)
    files=sorted(path('AAAdata/Appendix_3/对齐版本').glob('*.pkl'))
    if len(files)!=30:raise ValueError(f'附件3应为30文件，实际{len(files)}')
    rows=[];audit=[]
    for file in files:
        fields=load_fields(file)
        if len(fields['text_bert'])!=1:raise ValueError('专项文件必须单样本')
        inputs=raw_to_inputs(fields,cfg['data_dir'],encoder)
        dtype=next(model.parameters()).dtype
        out=model({k:v.to(device=device,dtype=dtype if v.is_floating_point() else v.dtype) for k,v in inputs.items()})
        c=int(out['logits'].argmax(-1)[0]);raw=float(out['intensity'][0])
        intensity=float(project_intensity(np.array([c]),np.array([raw]))[0])
        sid=str(fields.get('id',[file.stem])[0])
        rows.append({'sample_id':sid,'source_file':file.name,'polarity':LABELS[c],'intensity':intensity})
        audit.append({'source_file':file.name,'raw_intensity':raw,'input_sha256':sha256(file),
                      'observed_slots':inputs['O'].sum(-1)[0].tolist()})
    frame=pd.DataFrame(rows)
    if frame.source_file.duplicated().any() or not np.isfinite(frame.intensity).all():raise ValueError('提交结果非法')
    outdir=path(cfg['output']);outdir.mkdir(parents=True,exist_ok=True)
    target=outdir/'附件3_预测结果.csv';frame.to_csv(target,index=False)
    json_write(outdir/'附件3_提交核验.json',{'samples':len(rows),'checkpoint_sha256':sha256(path(cfg['checkpoint'])),
                    'csv_sha256':sha256(target),'labels_available':False,'accuracy':None,
                    'source':'official aligned raw -> saved train scaler once; general BERT per-slot',
                    'audit':audit})
    print(f'附件3预测已保存: {target}',flush=True)


@torch.no_grad()
def export_soft_alignment(model,data,output,device):
    """保存可复核50×500注意力矩阵；图仅用于诊断，不重新计算任务指标。"""
    from problem2.data import batch
    eligible=data['P_T'].any(-1)&data['O_A'].any(-1)&data['O_V'].any(-1)
    candidates=torch.where(eligible)[0]
    if not len(candidates):raise ValueError('没有三模态均可观测的注意力展示样本')
    index=candidates[:1];b=batch(data,index,device,dtype=next(model.parameters()).dtype)
    model.eval();out=model(b,return_attention=True)
    audio=out['attention_audio'][0].cpu().numpy();vision=out['attention_vision'][0].cpu().numpy()
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(root/'soft_alignment_weights.npz',audio=audio,vision=vision,
                        P_T=b['P_T'][0].cpu().numpy(),P_A=b['P_A'][0].cpu().numpy(),P_V=b['P_V'][0].cpu().numpy())
    json_write(root/'soft_alignment_example.json',{'id':data['id'][int(index[0])],
       'selection':'first test sample with observations in all three modalities',
       'seed':model.cfg['seed'],'matrix_shape':list(audio.shape),
       'interpretation':'learned attention, not ground-truth word timestamps or causal importance'})
    from shared.plotting import pyplot
    plt=pyplot();fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,weights,label in zip(axes,(audio,vision),('Audio','Vision')):
        im=ax.imshow(weights.mean(0),aspect='auto',origin='upper',cmap='viridis')
        ax.set(xlabel=f'{label} native slot (500)',ylabel='Text query slot (50)',title=f'Text → {label} soft alignment')
        fig.colorbar(im,ax=ax,label='Mean attention weight')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(root/f'soft_alignment.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
