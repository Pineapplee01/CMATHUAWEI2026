"""第五轮：同一热启动盆地的贪心参数平均，仅valid选模，单模型推理。
参考 https://arxiv.org/abs/2203.05482
"""
import copy
import json
import torch
import pandas as pd
from shared.common import path,json_write
from problem2.runtime import restore,dataset
from problem2.validation import evaluate
from problem2.evaluation import metrics

NAMES=['five_rounds_baseline','opt5_r1_rdrop','opt5_r2_ordinal',
       'opt5_r3_domainhead','opt5_r4_sam']


def mean_state(current, candidate, count):
    if current.keys()!=candidate.keys():raise ValueError('平均模型状态键不一致')
    result={}
    for key,a in current.items():
        b=candidate[key]
        if a.shape!=b.shape:raise ValueError(f'参数形状不同: {key}')
        if a.is_floating_point(): result[key]=(count*a+b)/(count+1)
        else:
            if not torch.equal(a,b):raise ValueError(f'离散状态不同: {key}')
            result[key]=a.clone()
    return result


def run():
    torch.set_num_threads(4)
    checkpoints={name:torch.load(path(f'AAAmodel/checkpoints/{name}.pt'),map_location='cpu',weights_only=False)
                 for name in NAMES}
    model,cfg=restore('AAAmodel/checkpoints/five_rounds_baseline.pt','cuda')
    # 原基线分类头冻结未入checkpoint；补回同一个本地骨干的原始分类头。
    original={k:v.detach().cpu().clone() for k,v in model.state_dict().items()
              if k.startswith('text.backbone.classifier.')}
    states={}
    for name,cp in checkpoints.items():
        if cp['encoder_sha256']!=checkpoints[NAMES[0]]['encoder_sha256']:
            raise ValueError('骨干身份不一致，不可平均')
        if cp['stats']!=checkpoints[NAMES[0]]['stats']:
            raise ValueError('标准化统计不一致，不可平均')
        states[name]={**original,**cp['model']}
        if not torch.equal(states[name]['head.background'],states[NAMES[0]]['head.background']):
            raise ValueError('背景坐标不同，不可直接平均')
    data=dataset(cfg,'valid')
    def assess(state,missing=False):
        model.load_state_dict(state,strict=False)
        model.text.cache.clear()
        frame,_=evaluate(model,data,cfg,.3 if missing else 0,reconstruction=False)
        return metrics(frame),frame
    ranking=sorted(NAMES,key=lambda n:checkpoints[n]['selection_score'])
    selected=[ranking[0]];current=copy.deepcopy(states[selected[0]])
    best,frame=assess(current);rows=[]
    for name in ranking[1:]:
        proposal=mean_state(current,states[name],len(selected))
        value,pred=assess(proposal)
        # 完整Accuracy严格增加才接受；不搜索任意连续混合系数。
        accepted=value['accuracy']>best['accuracy']
        rows.append({'candidate':name,'proposed_members':'+'.join(selected+[name]),
                     'accuracy':value['accuracy'],'macro_f1':value['macro_f1'],'accepted':accepted})
        print(rows[-1],flush=True)
        if accepted:
            current,best,frame=proposal,value,pred
            selected.append(name)
        if best['accuracy']>.7:break
    output=path('problem2/results/opt5_r5_soup');output.mkdir(exist_ok=True)
    cfg.update(checkpoint='AAAmodel/checkpoints/opt5_r5_soup.pt',output=str(output.relative_to(path('.'))),
               train_text_classifier=True,soup_members=selected)
    payload=copy.deepcopy(checkpoints[selected[0]])
    payload.update(model=current,config=cfg,epoch=-1,selection_score=-best['accuracy'])
    torch.save(payload,path(cfg['checkpoint']))
    json_write('configs/opt5_r5_soup.json',cfg)
    frame.to_csv(output/'predictions.csv',index=False)
    missing,_=assess(current,missing=True)
    report={'round':5,'name':'opt5_r5_soup','algorithm':'Greedy parameter averaging in shared warm-start basin',
            'members':selected,'proposals_evaluated':len(rows),'valid_accuracy':best['accuracy'],
            'valid_missing30_accuracy':missing['accuracy'],'target_met':best['accuracy']>.7,
            'metrics':best}
    pd.DataFrame(rows).to_csv(output/'averaging_trials.csv',index=False)
    json_write(output/'summary.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':run()
