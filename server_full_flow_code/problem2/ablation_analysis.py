"""冻结后评估与机制诊断；不读取test结果选择epoch或改写正式模型。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from shared.common import json_write
from problem2.data import batch
from problem2.local_missingness import synchronized_view
from problem2.evaluation import metrics
from problem2.output_contract import project_intensity


@torch.no_grad()
def outputs(model,data,extras=(),override=None):
    model.eval();result={k:[] for k in ('logits','intensity',*extras)}
    device=next(model.parameters()).device
    for lo in range(0,len(data['id']),32):
        idx=torch.arange(lo,min(lo+32,len(data['id'])))
        b=batch(data,idx,device,dtype=next(model.parameters()).dtype)
        out=model(b,gate_override=override[idx].to(device)) if override is not None else model(b)
        for key in result:result[key].append(out[key].detach().cpu())
    return {k:torch.cat(v) for k,v in result.items()}


def frame_of(out,data):
    raw=out['intensity'].numpy();c=out['logits'].argmax(-1).numpy()
    p=out['logits'].softmax(-1).numpy()
    return pd.DataFrame({'id':data['id'],'true_class':data['c'].numpy(),'true_intensity':data['y'].numpy(),
        'class':c,'intensity':project_intensity(c,raw),'raw_intensity':raw,
        **{f'p_{j}':p[:,j] for j in range(3)}})


def stability(full,missing):
    p=full['logits'].softmax(-1).clamp_min(1e-8);q=missing['logits'].softmax(-1).clamp_min(1e-8)
    mid=(p+q)/2
    js=.5*((p*(p/mid).log()).sum(-1)+(q*(q/mid).log()).sum(-1))
    return {'js':float(js.mean()),'flip_rate':float((p.argmax(-1)!=q.argmax(-1)).float().mean()),
            'raw_intensity_shift':float((full['intensity']-missing['intensity']).abs().mean())}


def gate_profile(gates,records,data,meta):
    rows=[];values={}
    for record in records:
        i=values.setdefault(record['id'],len(values))
        start,end=record['start_slot'],record['end_slot_exclusive'];width=end-start
        if width==0:continue
        missing='TAV'.index(record['modality'])
        for t in range(max(0,start-width),min(50,end+width)):
            relative=(t-start)/width;bin_id=min(29,max(0,int((relative+1)*10)))
            for m in range(3):
                if not data['P'][i,m,t]:continue
                rows.append({'relative_bin':bin_id,'relative_position':(bin_id+.5)/10-1,
                    'role':'missing modality' if m==missing else 'other modality','gate':float(gates[i,m,t])})
    if not rows:return []
    table=pd.DataFrame(rows).groupby(['relative_bin','relative_position','role'],as_index=False).gate.mean()
    return [meta|row for row in table.to_dict('records')]


def analyze_checkpoint(model,data,scenario_list,spec,variant,seed,run):
    test=data['test'];rec_enabled=model.reconstruct_enabled;gate_enabled=model.gate_enabled
    extra=('observed_embedding',) if rec_enabled else ()
    full=outputs(model,test,extra);full_frame=frame_of(full,test)
    full_frame.to_csv(run/'test_full.csv',index=False)
    metric_rows=[{'variant':variant,'seed':seed,'modalities':'none','rate':0.,'position':'none','mask_seed':0,
                  **metrics(full_frame)}]
    stable_rows=[];rec_rows=[];gate_rows=[];interventions=[]
    scenario_root=run/'scenarios';scenario_root.mkdir(exist_ok=True)
    fixed=None
    if gate_enabled:
        # 固定门控只从训练输入估计，不使用test分布或标签。
        train_gate=outputs(model,data['train'],('gates',))['gates'];P=data['train']['P']
        fixed=(train_gate*P).sum((0,2))/P.sum((0,2)).clamp_min(1)
        json_write(run/'train_fixed_gate.json',fixed)
    for scenario in scenario_list:
        name=scenario['modalities'];modalities=tuple('TAV'.index(m) for m in name)
        view,records=synchronized_view(test,scenario['rate'],modalities,scenario['position'],scenario['mask_seed'])
        extras=(('reconstruction',) if rec_enabled else ())+(('gates',) if gate_enabled else ())
        missing=outputs(model,view,extras);frame=frame_of(missing,test)
        frame.to_csv(scenario_root/f'{scenario["key"]}.csv',index=False)
        pd.DataFrame(records).to_csv(scenario_root/f'{scenario["key"]}_mask.csv',index=False)
        meta={'variant':variant,'seed':seed,**{k:v for k,v in scenario.items() if k!='key'}}
        metric_rows.append(meta|metrics(frame));stable_rows.append(meta|stability(full,missing))
        if rec_enabled:
            removed=test['P']&test['O']&~view['O'];target=full['observed_embedding']
            visible=view['O']&view['P']
            # 对照也只用缺失视图重新编码的观测，防止完整文本上下文泄漏。
            observed=outputs(model,view,('observed_embedding',))['observed_embedding']
            mean=(observed*visible[...,None]).sum(2)/visible.sum(2,keepdim=True).clamp_min(1)
            for m in modalities:
                mask=removed[:,m]
                if not mask.any():continue
                truth=target[:,m][mask]
                for kind,estimate in [('reconstructed',missing['reconstruction']),
                                      ('observed_mean',mean[:,:,None].expand_as(target))]:
                    pred=estimate[:,m][mask];cos=1-F.cosine_similarity(pred,truth,dim=-1)
                    rec_rows.append(meta|{'target_modality':'TAV'[m],'kind':kind,'cosine_distance':float(cos.mean()),
                         'target_norm':float(truth.norm(dim=-1).mean()),'estimate_norm':float(pred.norm(dim=-1).mean()),
                         'slots':int(mask.sum())})
            if name=='T' and scenario['rate']==spec['main_rate'] and scenario['position']=='random' and scenario['mask_seed']==spec['mask_seeds'][0]:
                dist=1-F.cosine_similarity(missing['reconstruction'][:,0],target[:,0],dim=-1)
                mask=removed[:,0];eligible=mask.any(-1);indices=torch.where(eligible)[0]
                if len(indices):
                    score=(dist*mask).sum(-1)/mask.sum(-1).clamp_min(1)
                    idx=int(indices[torch.argsort(score[indices])[len(indices)//2]])
                    np.savez_compressed(run/'latent_example.npz',reference=target[idx,0].numpy(),
                        observed=(observed[idx,0]*visible[idx,0,:,None]).numpy(),
                        reconstructed=missing['reconstruction'][idx,0].numpy(),mask=mask[idx].numpy(),valid=test['P'][idx,0].numpy())
                    json_write(run/'latent_example.json',{'id':test['id'][idx],'rule':'median per-sample cosine error',
                        'true_class':int(test['c'][idx]),'predicted_class':int(frame.iloc[idx]['class']),
                        'full_class':int(full_frame.iloc[idx]['class']),'seed':seed})
        if gate_enabled and scenario['rate']==spec['main_rate'] and scenario['position']=='random':
            if len(name)==1:gate_rows.extend(gate_profile(missing['gates'],records,test,meta))
            if name=='TAV':
                interventions.append(meta|{'intervention':'dynamic',**metrics(frame)})
                rng=torch.Generator().manual_seed(scenario['mask_seed'])
                shuffled=missing['gates'][torch.randperm(len(test['id']),generator=rng)]
                for kind,value in [('fixed_train_mean',fixed[None,:,None].expand_as(missing['gates'])),('shuffled',shuffled)]:
                    pred=frame_of(outputs(model,view,override=value),test)
                    interventions.append(meta|{'intervention':kind,**metrics(pred)})
    for filename,rows in [('metrics',metric_rows),('stability',stable_rows),('reconstruction',rec_rows),
                           ('gate_profile',gate_rows),('gate_interventions',interventions)]:
        if rows:pd.DataFrame(rows).to_csv(run/f'{filename}.csv',index=False)
    json_write(run/'analysis_done.json',{'seed':seed,'variant':variant,'scenarios':len(scenario_list),
               'smoke':spec['smoke'],'test_samples':len(test['id'])})


def finalize(root,spec):
    for name in ('metrics','stability','reconstruction','gate_profile','gate_interventions'):
        files=sorted((root/'runs').glob(f'*/{name}.csv'))
        if files:pd.concat([pd.read_csv(f) for f in files],ignore_index=True).to_csv(root/f'{name}.csv',index=False)
    scores=pd.read_csv(root/'metrics.csv')
    # B0和Bm是同一模型，区别仅在评估视图。
    primary=scores[(scores.rate==spec['main_rate'])&(scores.position=='random')]
    metric_cols=['accuracy','macro_f1','neutral_f1','neutral_recall','mae','pearson']
    primary=primary.groupby(['variant','seed','modalities'],as_index=False)[metric_cols].mean()
    primary=primary.groupby(['variant','seed'],as_index=False)[metric_cols].mean()
    reference=scores[(scores.variant=='Bm')&(scores.rate==0)].copy();reference['variant']='B0'
    primary=pd.concat([reference[['variant','seed',*metric_cols]],primary],ignore_index=True)
    primary.to_csv(root/'primary_by_seed.csv',index=False)
    summary=primary.groupby('variant')[metric_cols].agg(['mean','std','count'])
    summary.to_csv(root/'primary_summary.csv')
    summary_lines=['# 六组连续缺失研究实验', '',
      '**仅小样本一轮流程检查，不可用于论文结论。**' if spec['smoke'] else '完整训练结果；epoch仅依据valid选择。',
      '', '正式问题二正式模型保持不变。F表示G+C+R研究扩展，不是已采用问题二正式模型的消融等价物。B0/Bm共用检查点，分别评估完整/缺失输入。',
      '', '| 组 | 种子数 | ACC均值 | Macro-问题二正式模型均值 | MAE均值 |','|---|---:|---:|---:|---:|']
    for v in ('B0','Bm','G','C','R','F'):
        s=primary[primary.variant==v]
        summary_lines.append(f'| {v} | {len(s)} | {s.accuracy.mean():.2%} | {s.macro_f1.mean():.2%} | {s.mae.mean():.4f} |')
    summary_lines+=['','按随机位置重复先平均，再对七种模态组合等权平均。标准差跨训练seed计算，单seed不估计标准差。',
      '训练均同时使用完整/缺失监督；仅G/C/R模块开关不同。B0不是额外完整输入专训模型，也不是严格理论上限。',
      '机制诊断属于推理干预；没有将干预结果用于选epoch。六组设计不支持严格估计模块交互协同，需额外两两组合实验。']
    (root/'实验报告.md').write_text('\n'.join(summary_lines)+'\n')
