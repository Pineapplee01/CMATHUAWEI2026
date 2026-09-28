"""只用train拟合的固定表示核分类诊断，不读取test。"""
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from shared.common import config, path, to_device, seed_all
from problem2.runtime import dataset, loader
from problem2.text_encoder import TextEncoder


@torch.no_grad()
def representations(ds, cfg, encoder):
    features=[];labels=[]
    for raw in loader(ds,cfg):
        batch=to_device(raw,cfg['device']); masks=batch['O']
        text=encoder(batch['tokens'],masks[:,0])
        parts=[]
        for m,x in enumerate([text,batch['audio'],batch['vision']]):
            count=masks[:,m].sum(1,keepdim=True).clamp_min(1)
            mean=(x*masks[:,m,:,None]).sum(1)/count
            if m==0: part=mean
            else:
                variance=((x-mean[:,None]).square()*masks[:,m,:,None]).sum(1)/count
                part=torch.cat([mean,variance.sqrt()],-1)
            parts.append(part.cpu().numpy())
        features.append(parts);labels.append(raw['c'].numpy())
    return [np.concatenate([b[m] for b in features]) for m in range(3)],np.concatenate(labels)


def run(cfg):
    seed_all(cfg['seed']);cache=path('AAAdata/processed/kernel_probe_sst2.npz')
    if cache.exists():
        with np.load(cache) as d:
            train=[d[f'train{m}'] for m in range(3)];valid=[d[f'valid{m}'] for m in range(3)]
            y,z=d['y'],d['z']
    else:
        encoder=TextEncoder(cfg['bert_path']).to(cfg['device']).eval();encoder.cls_mix=1.
        train,y=representations(dataset(cfg,'train'),cfg,encoder)
        valid,z=representations(dataset(cfg,'valid'),cfg,encoder)
        cache.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(cache,**{f'train{m}':train[m] for m in range(3)},
                            **{f'valid{m}':valid[m] for m in range(3)},y=y,z=z)
    transformed=[[],[]]
    for m in range(3):
        scaler=StandardScaler().fit(train[m]);pca=PCA(n_components=min(128,train[m].shape[1]),random_state=cfg['seed'])
        a=pca.fit_transform(scaler.transform(train[m]));b=pca.transform(scaler.transform(valid[m]))
        transformed[0].append(normalize(a));transformed[1].append(normalize(b))
    rows=[];output=path('problem2/results/kernel_probe');output.mkdir(exist_ok=True)
    for av_weight in [0.,.25,.5,1.]:
        x=np.concatenate([transformed[0][0],*[av_weight*a for a in transformed[0][1:]]],-1)
        v=np.concatenate([transformed[1][0],*[av_weight*a for a in transformed[1][1:]]],-1)
        for gamma in [.25,1.,4.]:
            for C in [.1,1.,10.]:
                clf=SVC(C=C,gamma=gamma).fit(x,y);pred=clf.predict(v)
                row={'av_weight':av_weight,'gamma':gamma,'C':C,'accuracy':accuracy_score(z,pred),
                     'macro_f1':f1_score(z,pred,average='macro')}
                rows.append(row);print(row,flush=True)
                pd.DataFrame(rows).to_csv(output/'diagnostics.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/sentiment_transfer.json')
    run(config(p.parse_args().config))
