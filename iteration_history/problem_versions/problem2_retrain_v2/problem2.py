"""问题二统一入口：audit / evaluate / predict，运行于项目根目录。"""
import argparse
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import torch
import numpy as np
from shared.common import path, json_write
from problem2_retrain_v2.data import ROOT_DATA, audit_data, load_split, MissingTextEncoder, masked_view, sha256
from problem2_retrain_v2.training import restore, evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['audit','evaluate','predict'])
    parser.add_argument('--device',default='cuda:3')
    parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--config',default='configs/problem2_aligned.json')
    parser.add_argument('--split',choices=['valid','test'],default='valid')
    parser.add_argument('--frozen-test',action='store_true')
    parser.add_argument('--scenarios',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.action=='audit':
        print(audit_data('problem2_retrain_v2/results/aligned/data_audit.json'));return
    cfg=json.loads(path(args.config).read_text())
    if args.action=='predict':
        from problem2_retrain_v2.inference import predict_appendix3
        predict_appendix3(cfg,args.device);return
    if args.split=='test' and not args.frozen_test:parser.error('test仅在冻结后评估，需--frozen-test')
    if args.split=='test':
        frozen=json.loads(path('problem2_retrain_v2/results/aligned/frozen_selection.json').read_text())
        if sha256(path(cfg['checkpoint']))!=frozen['checkpoint_sha256']:raise ValueError('冻结权重身份不符')
    model,trained_cfg=restore(cfg['checkpoint'],args.device)
    cfg=cfg|{'data_dir':trained_cfg['data_dir']}
    data=load_split(cfg['data_dir'],args.split,dtype=getattr(np,cfg.get('precision','float32')))
    metrics,frame=evaluate(model,data,args.device)
    out=path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    frame.to_csv(out/f'{args.split}_full.csv',index=False)
    json_write(out/f'{args.split}_full.json',metrics)
    from problem2_retrain_v2.evaluation import error_plots
    error_plots(frame,path('problem2_retrain_v2/figures/aligned')/args.split)
    online=trained_cfg.get('finetune_bert',False)
    encoder=None if online else MissingTextEncoder(args.device)
    partial,records=masked_view(data,encoder,.3,(0,1,2),'random',4026,reencode=not online)
    pm,pf=evaluate(model,partial,args.device)
    pf.to_csv(out/f'{args.split}_local30.csv',index=False)
    json_write(out/f'{args.split}_local30.json',pm)
    json_write(out/f'{args.split}_local30_mask.json',records)
    print(json.dumps({'full':metrics,'local30':pm},ensure_ascii=False),flush=True)
    if args.scenarios:
        from problem2_retrain_v2.inference import evaluate_scenarios
        evaluate_scenarios(model,data,encoder,args.device,out)


if __name__=='__main__':main()
