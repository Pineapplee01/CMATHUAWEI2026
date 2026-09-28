"""三分类预训练表示的正则化线性读出诊断，拟合仅使用train。"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from shared.common import config,path,seed_all
from problem2.runtime import dataset
from problem2.sentiment_encoder import SentimentEncoder
from problem2.kernel_probe import representations


def run():
    cfg=config('configs/neutral_transfer.json');seed_all(cfg['seed'])
    cache=path('AAAdata/processed/prior_probe_roberta.npz')
    if cache.exists():
        with np.load(cache) as d:
            train=[d[f'train{m}'] for m in range(3)];valid=[d[f'valid{m}'] for m in range(3)];y,z=d['y'],d['z']
    else:
        encoder=SentimentEncoder(cfg['bert_path'],cfg['input_tokenizer_path']).to('cuda').eval()
        train,y=representations(dataset(cfg,'train'),cfg,encoder)
        valid,z=representations(dataset(cfg,'valid'),cfg,encoder)
        np.savez_compressed(cache,**{f'train{m}':train[m] for m in range(3)},
            **{f'valid{m}':valid[m] for m in range(3)},y=y,z=z)
    out=path('problem2/results/prior_probe');out.mkdir(exist_ok=True)
    rows=[{'features':'zero_shot','C':0,'accuracy':accuracy_score(z,valid[0][:,-3:].argmax(-1))}]
    print(rows[0],flush=True)
    for name,indices in [('prior',[0]),('prior_av',[0,1,2]),('embedding_av',[0,1,2]),('embedding',[0])]:
        x=np.concatenate([train[m][:,-3:] if m==0 and name.startswith('prior') else train[m] for m in indices],-1)
        v=np.concatenate([valid[m][:,-3:] if m==0 and name.startswith('prior') else valid[m] for m in indices],-1)
        scaler=StandardScaler().fit(x);x=scaler.transform(x);v=scaler.transform(v)
        for C in [.001,.01,.1,1.,10.]:
            model=LogisticRegression(C=C,max_iter=1000).fit(x,y);pred=model.predict(v)
            row={'features':name,'C':C,'accuracy':accuracy_score(z,pred)}
            rows.append(row);print(row,flush=True)
            np.save(out/f'logits_{name}_{C}.npy',model.decision_function(v))
    pd.DataFrame(rows).to_csv(out/'diagnostics.csv',index=False)


if __name__=='__main__':run()
