"""同条件候选比较：只用train拟合、valid选模，冻结后一次test核验。"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
from problem2_retrain_v2.data import load_split, batch, masked_view, sha256
from problem2_retrain_v2.model import build_model, objective, class_balance_metadata, boundary_settings
from problem2_retrain_v2.evaluation import metrics
from problem2_retrain_v2.output_contract import project_intensity
from shared.common import path, json_write, seed_all


@torch.no_grad()
def evaluate(model, data, device, batch_size=128):
    model.eval();rows=[]
    for lo in range(0,len(data['id']),batch_size):
        indices=torch.arange(lo,min(lo+batch_size,len(data['id'])))
        b=batch(data,indices,device,dtype=next(model.parameters()).dtype);out=model(b)
        c=out['logits'].argmax(-1);raw=out['intensity']
        intensity=project_intensity(c.cpu().numpy(),raw.cpu().numpy())
        for i,sid in enumerate(b['id']):
            rows.append({'id':sid,'true_class':int(b['c'][i]),'true_intensity':float(b['y'][i]),
                         'class':int(c[i]),'intensity':float(intensity[i]),'raw_intensity':float(raw[i])})
    frame=pd.DataFrame(rows)
    return metrics(frame),frame


def train_candidate(cfg, train, valid, train_views, valid_view):
    # 缺少新参数的历史配置仍按旧目标训练；新总入口/配置显式设置beta=1。
    balance=class_balance_metadata(train['c'],cfg.get('balance_beta',0.0),cfg.get('neutral_weight',1.0))
    boundary=boundary_settings(cfg)
    cfg=cfg|balance|boundary
    class_weights=cfg['class_weights']
    print(json.dumps({'training_class_balance':balance},ensure_ascii=False),flush=True)
    seed_all(cfg['seed'])
    device=cfg['device']
    dtype={'float32':torch.float32,'float64':torch.float64}[cfg.get('precision','float32')]
    model=build_model(cfg).to(device=device,dtype=dtype)
    if cfg.get('architecture')=='soft_alignment' and cfg.get('init_checkpoint'):
        raise ValueError('未对齐 soft_alignment 不从对齐F1投影初始化；请去掉 init_checkpoint')
    if cfg.get('init_checkpoint'):
        initial=torch.load(path(cfg['init_checkpoint']),map_location='cpu',weights_only=False)
        if initial.get('format')!='aligned_v4':raise ValueError('初始化仅允许合规aligned_v4检查点')
        if initial['data_identity']!={s:sha256(Path(cfg['data_dir'])/f'{s}.npz') for s in ('train','valid')}:
            raise ValueError('初始化模型的数据划分与本次不一致')
        if cfg.get('architecture') in ('f1','f3'):
            cfg['initialized_shared_keys']=model.initialize_shared(initial['model'])
        else:
            missing,unexpected=model.load_state_dict(initial['model'],strict=False)
            if unexpected or any(not (key.startswith('bert.') or key in ('text_mu','text_sigma')) for key in missing):
                raise ValueError('初始化模型架构不兼容')
        cfg['init_checkpoint_sha256']=sha256(path(cfg['init_checkpoint']))
    if cfg.get('finetune_bert',False):
        cfg['bert_source_sha256']=sha256(path('AAAmodel/bert-base-uncased/model.safetensors'))
    parameters=model.parameters()
    if cfg.get('finetune_bert',False):
        parameters=[{'params':[p for n,p in model.named_parameters() if not n.startswith('bert.') and p.requires_grad],
                     'lr':cfg.get('lr',.0001)},
                    {'params':[p for p in model.bert.parameters() if p.requires_grad],'lr':cfg.get('bert_lr',1e-5)}]
    optimizer=torch.optim.AdamW(parameters,lr=cfg.get('lr',.0005),weight_decay=.02)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs'],eta_min=0. if cfg.get('finetune_bert',False) else 5e-5)
    best=-float('inf');bad=0;history=[];begin=time.monotonic()
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
    run=path(cfg['output']);run.mkdir(parents=True,exist_ok=True)
    checkpoint=path(cfg['checkpoint']);checkpoint.parent.mkdir(parents=True,exist_ok=True)
    accumulation=int(cfg.get('gradient_accumulation',1))
    if accumulation<1:raise ValueError('gradient_accumulation必须为正整数')
    if cfg.get('finetune_bert',False):
        complete,_=evaluate(model,valid,device,batch_size=32)
        missing,_=evaluate(model,valid_view,device,batch_size=32)
        json_write(run/'initial_validation.json',{'complete':complete,'local30':missing})
    if cfg.get('reset_training_rng',False):seed_all(cfg['seed'])
    for epoch in range(cfg['epochs']):
        boundary_weight=boundary['boundary_weight']*min(1.,(epoch+1)/boundary['boundary_warmup_epochs'])
        generator=torch.Generator().manual_seed(cfg['seed']+epoch) if cfg.get('reset_training_rng',False) else None
        model.train();order=torch.randperm(len(train['id']),generator=generator);total=0
        for lo in range(0,len(order),cfg['batch_size']):
            idx=order[lo:lo+cfg['batch_size']]
            step=lo//cfg['batch_size']
            group_start=(step//accumulation)*accumulation*cfg['batch_size']
            group_samples=min(accumulation*cfg['batch_size'],len(order)-group_start)
            full=batch(train,idx,device,dtype=dtype)
            partial=batch(train_views[(epoch+lo//cfg['batch_size'])%len(train_views)],idx,device,dtype=dtype)
            if step%accumulation==0:optimizer.zero_grad(set_to_none=True)
            # 边界排序只监督完整视图，避免缺失已删除判别证据时仍强制排序。
            loss=objective(model(full),full,class_weights=class_weights,
                           boundary_weight=boundary_weight,boundary_margin=boundary['boundary_margin'],
                           boundary_max_intensity=boundary['boundary_max_intensity'])
            partial_output=model(partial)
            loss=loss+.4*min(1,(epoch+1)/5)*objective(partial_output,partial,class_weights=class_weights)
            (loss*len(idx)/group_samples).backward()
            if (step+1)%accumulation==0 or lo+len(idx)==len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
            total+=float(loss.detach())*len(idx)
        scheduler.step()
        complete,_=evaluate(model,valid,device)
        missing,_=evaluate(model,valid_view,device)
        # 预先声明，分类完整/局部权重70/30；强度单独报告，不用test挑架构。
        score=.7*complete['accuracy']+.3*missing['accuracy']
        record={'seed':cfg['seed'],**balance,**boundary,'effective_boundary_weight':boundary_weight,'name':cfg['name'],'epoch':epoch+1,'loss':total/len(order),'selection_score':score,
                'complete':complete,'local30':missing,'elapsed_seconds':time.monotonic()-begin}
        history.append(record)
        if score>best+1e-8:
            best=score;bad=0
            torch.save({'format':'unaligned_v1' if cfg.get('architecture')=='soft_alignment' else 'aligned_v4','config':cfg,'model':model.state_dict(),
                        'best_epoch':epoch+1,'validation':record,
                        'data_identity':{s:sha256(Path(cfg['data_dir'])/f'{s}.npz') for s in ('train','valid')},
                        'scaler_sha256':sha256(Path(cfg['data_dir'])/'scaler_params.npz')},checkpoint)
        else:bad+=1
        print(json.dumps({'run':cfg['name'],'seed':cfg['seed'],**balance,'boundary_weight':boundary_weight,'epoch':epoch+1,'acc':round(complete['accuracy'],4),
                          'local30':round(missing['accuracy'],4),'best':round(best,4)},ensure_ascii=False),flush=True)
        if bad>=cfg['patience']:break
    json_write(run/'history.json',history)
    saved=torch.load(checkpoint,map_location=device,weights_only=False)
    model.load_state_dict(saved['model'])
    complete,cf=evaluate(model,valid,device);missing,mf=evaluate(model,valid_view,device)
    cf.to_csv(run/'valid_full.csv',index=False);mf.to_csv(run/'valid_local30.csv',index=False)
    result={'name':cfg['name'],'dataset':Path(cfg['data_dir']).name,'architecture':cfg['architecture'],
            'seed':cfg['seed'],**balance,**boundary,'effective_boundary_weight':saved['validation']['effective_boundary_weight'],'precision':cfg.get('precision','float32'),'best_epoch':saved['best_epoch'],'complete':complete,'local30':missing,
            'selection_score':best,'checkpoint':str(checkpoint.relative_to(path('.'))),
            'checkpoint_sha256':sha256(checkpoint),'parameters':sum(p.numel() for p in model.parameters()),
            'device':device,'gpu_name':torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
            'peak_gpu_bytes':torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None,
            'elapsed_seconds':time.monotonic()-begin,'epochs_run':len(history)}
    json_write(run/'summary.json',result)
    return result


def restore(filename,device):
    saved=torch.load(path(filename),map_location='cpu',weights_only=False)
    if saved.get('format') not in ('aligned_v4','unaligned_v1'):
        raise ValueError('仅支持 aligned_v4 / unaligned_v1 检查点')
    cfg=saved['config']
    if (saved['format']=='unaligned_v1')!=(cfg.get('architecture')=='soft_alignment'):
        raise ValueError('检查点格式与对齐/未对齐架构不一致')
    if sha256(Path(cfg['data_dir'])/'scaler_params.npz')!=saved['scaler_sha256']:
        raise ValueError('数据缩放参数身份已改变')
    model=build_model(cfg,pretrained=False).to(device=device,dtype={'float32':torch.float32,'float64':torch.float64}[cfg.get('precision','float32')]);model.load_state_dict(saved['model'],strict=True);model.eval()
    return model,cfg
