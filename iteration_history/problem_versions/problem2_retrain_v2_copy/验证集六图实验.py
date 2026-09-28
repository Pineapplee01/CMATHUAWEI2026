"""对齐验证集六图：冻结原消融检查点，复用测试图场景，绝不读取test。"""
import argparse,json,sys,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;sys.path.insert(0,str(ROOT))
OUT=HERE/'valid_experiments';SEEDS=[2027,2028,2035];VAR=['B0','noC','noQ','noB','noBeta']


def checkpoint_folder(seed,variant):
    suffix='five_seeds' if seed<=2030 else 's2031_2035'
    return ROOT/f'problem2_retrain_v2/results/aligned/ablations/f1_gate_component_ablation_{suffix}/runs/{variant}_seed{seed}'


def run(seed,device):
    import torch,numpy as np,pandas as pd
    from torch.nn import functional as F
    from problem2_retrain_v2.training import restore
    from problem2_retrain_v2.data import load_split,sha256,batch
    from problem2_retrain_v2.ablation import scenarios
    from problem2_retrain_v2.local_missingness import synchronized_view
    from problem2_retrain_v2.ablation_analysis import frame_of,stability
    from problem2_retrain_v2.evaluation import metrics
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    @torch.no_grad()
    def outputs(model, data, extras=()):
        model.eval();result={k:[] for k in ('logits','intensity',*extras)}
        for lo in range(0,len(data['id']),64):
            idx=torch.arange(lo,min(lo+64,len(data['id'])))
            out=model(batch(data,idx,device,dtype=next(model.parameters()).dtype))
            for key in result:result[key].append(out[key].detach().cpu())
        return {k:torch.cat(v) for k,v in result.items()}
    spec={'rates':[.1,.2,.3,.4,.5],'main_rate':.3,'mask_seeds':[5026,5027,5028]}
    scene=scenarios(spec);data=None
    for variant in VAR:
        src=checkpoint_folder(seed,variant);summary=json.loads((src/'summary.json').read_text())
        ckpt=ROOT/summary['checkpoint'];folder=OUT/'runs'/f'{variant}_seed{seed}';folder.mkdir(parents=True,exist_ok=True)
        identity={'split':'valid','seed':seed,'variant':variant,'checkpoint_sha256':sha256(ckpt),'scenarios':scene,'inference':{'dtype':'float32','matmul':'highest','batch_size':64}}
        done=folder/'done.json'
        if done.exists():
            assert json.loads(done.read_text())==identity
            print('SKIP',variant,seed,flush=True);continue
        assert identity['checkpoint_sha256']==summary['checkpoint_sha256']
        model,cfg=restore(ckpt,device)
        if data is None:data=load_split(cfg['data_dir'],'valid')
        assert data['split']=='valid' and len(data['id'])==728
        # 补全图仅需要B0；同一次前向同时取预测和隐表示，避免多跑一次BERT。
        full=outputs(model,data,('observed_embedding',) if variant=='B0' else ())
        frame=frame_of(full,data);frame.to_csv(folder/'valid_full.csv',index=False)
        reference=pd.read_csv(src/'valid_full.csv').set_index('id').loc[frame.id]
        assert np.array_equal(reference.true_class.to_numpy(),frame.true_class.to_numpy())
        changed=reference['class'].to_numpy()!=frame['class'].to_numpy()
        probabilities=full['logits'].softmax(-1).topk(2,dim=-1).values
        audit={'reference_accuracy':summary['complete']['accuracy'],'current_accuracy':metrics(frame)['accuracy'],
               'changed_predictions':int(changed.sum()),'changed_ids':frame.id[changed].tolist(),
               'current_top2_probability_gaps':(probabilities[:,0]-probabilities[:,1]).numpy()[changed].tolist()}
        (folder/'reference_comparison.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
        if changed.any():print('REFERENCE_DIFFERENCE',variant,seed,audit,flush=True)
        rows=[{'variant':variant,'seed':seed,'modalities':'none','rate':0.,'position':'none','mask_seed':0,**metrics(frame)}]
        st=[];rec=[];(folder/'scenarios').mkdir(exist_ok=True)
        for index,s in enumerate(scene):
            mods=tuple('TAV'.index(m) for m in s['modalities'])
            view,records=synchronized_view(data,s['rate'],mods,s['position'],s['mask_seed'])
            need_rec=variant=='B0' and len(mods)==1 and s['position']=='random'
            pred=outputs(model,view,('observed_embedding','reconstruction') if need_rec else ())
            frame=frame_of(pred,data);frame.to_csv(folder/'scenarios'/f'{s["key"]}.csv',index=False)
            pd.DataFrame(records).to_csv(folder/'scenarios'/f'{s["key"]}_mask.csv',index=False)
            meta={'variant':variant,'seed':seed,**{k:v for k,v in s.items() if k!='key'}}
            rows.append(meta|metrics(frame));st.append(meta|stability(full,pred))
            if need_rec:
                m=mods[0];removed=(data['P']&data['O']&~view['O'])[:,m]
                visible=view['O']&view['P'];target=full['observed_embedding'][:,m][removed]
                observed=pred['observed_embedding']
                means=(observed*visible[...,None]).sum(2)/visible.sum(2,keepdim=True).clamp_min(1)
                for kind,estimate in [('reconstructed',pred['reconstruction'][:,m]),('observed_mean',means[:,m,None].expand_as(observed[:,m]))]:
                    if removed.any():rec.append(meta|{'kind':kind,'target_modality':'TAV'[m],'slots':int(removed.sum()),'cosine_distance':float((1-F.cosine_similarity(estimate[removed],target,dim=-1)).mean())})
            if index%10==0:print(variant,seed,index+1,len(scene),flush=True)
        for name,values in [('metrics',rows),('stability',st),('reconstruction',rec)]:
            if values:pd.DataFrame(values).to_csv(folder/f'{name}.csv',index=False)
        done.write_text(json.dumps(identity,ensure_ascii=False,indent=2))
        del model;torch.cuda.empty_cache();print('DONE',variant,seed,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--device',default='cuda:3');p.add_argument('--plots-only',action='store_true');a=p.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.seed:run(a.seed,a.device);return
    if not a.plots_only:
        def worker(job):
            seed,device=job
            with (OUT/f'seed{seed}.log').open('a') as log:
                subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--seed',str(seed),'--device',device],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(worker,[(2027,'cuda:2'),(2028,'cuda:3'),(2035,'cuda:3')]))
    import pandas as pd
    for name in ['metrics','stability','reconstruction']:
        paths=list((OUT/'runs').glob(f'*/{name}.csv'))
        pd.concat([pd.read_csv(x) for x in paths],ignore_index=True).to_csv(OUT/f'{name}.csv',index=False)
    from 验证集六图绘制 import report
    report()

if __name__=='__main__':main()
