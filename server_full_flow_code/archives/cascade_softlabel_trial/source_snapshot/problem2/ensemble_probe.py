"""验证集上比较固定均匀logit平均，只作方案诊断，不读取test。"""
import itertools
import numpy as np
import pandas as pd
import torch
from shared.common import path, to_device
from problem2.runtime import restore, dataset, loader

NAMES=['mixup','semantic_readout','task_adaptation','bias_tuning','full_task_tuning',
       'full_depth_lora','sentiment_transfer','sentiment_global',
       'neutral_transfer','neutral_adaptation','context_fusion','context_sentiment']


@torch.no_grad()
def collect(name):
    checkpoint=path(f'AAAmodel/checkpoints/{name}.pt')
    output=path('problem2/results/ensemble_probe');output.mkdir(exist_ok=True)
    filename=output/f'{name}.npz'
    if filename.exists():
        with np.load(filename) as d: return d['logits'],d['y']
    model,cfg=restore(checkpoint,'cuda');ds=dataset(cfg,'valid');values=[]
    for batch in loader(ds,cfg): values.append(model(to_device(batch,'cuda'))['logits'].cpu().numpy())
    logits=np.concatenate(values);y=np.asarray(ds.fields['classification_labels']).reshape(-1)
    np.savez_compressed(filename,logits=logits,y=y,ids=np.asarray(ds.ids))
    print(name,float((logits.argmax(1)==y).mean()),flush=True)
    return logits,y


def run():
    all_logits=[]
    for name in NAMES:
        logits,y=collect(name);all_logits.append(logits)
    rows=[]
    for count in (1,2,3):
        for combo in itertools.combinations(range(len(NAMES)),count):
            scores=np.mean([all_logits[i] for i in combo],axis=0)
            rows.append({'models':'+'.join(NAMES[i] for i in combo),'count':count,
                         'accuracy':float((scores.argmax(1)==y).mean())})
    frame=pd.DataFrame(rows).sort_values(['accuracy','count'],ascending=[False,True])
    frame.to_csv(path('problem2/results/ensemble_probe/uniform_comparison.csv'),index=False)
    print(frame.head(12).to_string(index=False),flush=True)


if __name__=='__main__':run()
