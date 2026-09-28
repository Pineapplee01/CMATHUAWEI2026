"""问题二统一入口：audit / compare / evaluate / predict，运行于项目根目录。"""
import argparse
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import shutil
from datetime import datetime, timezone
import torch
import numpy as np
from shared.common import path, json_write
from problem2.data import ROOT_DATA, audit_data, load_split, MissingTextEncoder, masked_view, sha256
from problem2.training import train_candidate, restore, evaluate


def compare(args):
    output=path('problem2/results/aligned');output.mkdir(parents=True,exist_ok=True)
    candidates=audit_data(output/'data_audit.json')
    encoder=MissingTextEncoder(args.device);results=[]
    for name in candidates:
        directory=path(ROOT_DATA)/name
        train=load_split(directory,'train');valid=load_split(directory,'valid')
        train_views=[]
        for rate,seed in ((.1,3026),(.3,3027),(.5,3028)):
            view,_=masked_view(train,encoder,rate,(0,1,2),'random',seed)
            train_views.append(view)
        valid_view,records=masked_view(valid,encoder,.3,(0,1,2),'random',4026)
        json_write(output/f'{name}_valid_mask.json',records)
        for architecture in ('pooled','cross_attention'):
            run_name=f'{name}_{architecture}'
            cfg={'name':run_name,'architecture':architecture,'hidden':96,'seed':2026,
                 'device':args.device,'epochs':args.epochs,'patience':10,'batch_size':64,'lr':.0005,
                 'data_dir':str(directory),'features_prestandardized':True,'text_representation':'per_slot',
                 'text_encoder':'bert-base-uncased','selection_rule':'0.7 valid full acc + 0.3 valid TAV30 acc',
                 'checkpoint':f'AAAmodel/checkpoints/aligned/{run_name}.pt',
                 'output':f'problem2/results/aligned/{run_name}'}
            result=train_candidate(cfg,train,valid,train_views,valid_view)
            results.append(result);json_write(output/'comparison.json',results)
        del train,valid,train_views,valid_view
    winner=max(results,key=lambda r:(r['selection_score'],r['complete']['macro_f1'],-r['complete']['mae']))
    dest=path('AAAmodel/checkpoints/problem2_aligned.pt');shutil.copy2(path(winner['checkpoint']),dest)
    saved=torch.load(dest,map_location='cpu',weights_only=False)
    cfg=saved['config']|{'checkpoint':str(dest.relative_to(path('.'))),'output':'problem2/results/aligned/selected'}
    json_write('configs/problem2_aligned.json',cfg)
    json_write(output/'frozen_selection.json',{'selected':winner,'frozen_at':datetime.now(timezone.utc).isoformat(),
                                             'test_used_for_selection':False,'checkpoint_sha256':sha256(dest)})
    print('FROZEN '+json.dumps(winner,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['audit','compare','evaluate','predict'])
    parser.add_argument('--device',default='cuda:3')
    parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--config',default='configs/problem2_aligned.json')
    parser.add_argument('--split',choices=['valid','test'],default='valid')
    parser.add_argument('--frozen-test',action='store_true')
    parser.add_argument('--scenarios',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.action=='audit':
        print(audit_data('problem2/results/aligned/data_audit.json'));return
    if args.action=='compare':compare(args);return
    cfg=json.loads(path(args.config).read_text())
    if args.action=='predict':
        from problem2.inference import predict_appendix3
        predict_appendix3(cfg,args.device);return
    if args.split=='test' and not args.frozen_test:parser.error('test仅在冻结后评估，需--frozen-test')
    if args.split=='test':
        frozen=json.loads(path('problem2/results/aligned/frozen_selection.json').read_text())
        if sha256(path(cfg['checkpoint']))!=frozen['checkpoint_sha256']:raise ValueError('冻结权重身份不符')
    model,trained_cfg=restore(cfg['checkpoint'],args.device)
    cfg=cfg|{'data_dir':trained_cfg['data_dir']}
    data=load_split(cfg['data_dir'],args.split,dtype=getattr(np,cfg.get('precision','float32')))
    metrics,frame=evaluate(model,data,args.device)
    out=path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    frame.to_csv(out/f'{args.split}_full.csv',index=False)
    json_write(out/f'{args.split}_full.json',metrics)
    from problem2.evaluation import error_plots
    error_plots(frame,path('problem2/figures/aligned')/args.split)
    online=trained_cfg.get('finetune_bert',False)
    encoder=None if online else MissingTextEncoder(args.device)
    partial,records=masked_view(data,encoder,.3,(0,1,2),'random',4026,reencode=not online)
    pm,pf=evaluate(model,partial,args.device)
    pf.to_csv(out/f'{args.split}_local30.csv',index=False)
    json_write(out/f'{args.split}_local30.json',pm)
    json_write(out/f'{args.split}_local30_mask.json',records)
    print(json.dumps({'full':metrics,'local30':pm},ensure_ascii=False),flush=True)
    if args.scenarios:
        from problem2.inference import evaluate_scenarios
        evaluate_scenarios(model,data,encoder,args.device,out)


if __name__=='__main__':main()
