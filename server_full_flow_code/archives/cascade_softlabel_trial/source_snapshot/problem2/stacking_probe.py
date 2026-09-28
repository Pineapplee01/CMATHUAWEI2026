"""分组交叉验证元校准诊断；验证标签拟合的全量得分不当作泛化指标。"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax
from shared.common import path
from problem2.ensemble_probe import NAMES


def run():
    arrays=[]
    for name in NAMES:
        with np.load(path(f'problem2/results/ensemble_probe/{name}.npz')) as d:
            arrays.append(d['logits']);y=d['y'];ids=d['ids']
    groups=np.array([i.split('$_$')[0] for i in ids])
    out=path('problem2/results/stacking_probe');out.mkdir(exist_ok=True)
    rows=[]
    for mode in ['logits','probabilities','uncertainty']:
        a=np.stack(arrays,1);p=softmax(a,axis=-1)
        x=a.reshape(len(a),-1) if mode=='logits' else p.reshape(len(a),-1)
        if mode=='uncertainty':
            x=np.concatenate([x,-(p*np.log(p+1e-8)).sum(-1),p.std(1)],-1)
        for C in [.001,.01,.1,1.]:
            pred=np.zeros_like(y)
            for train,held in GroupKFold(5).split(x,y,groups):
                model=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=1000))
                model.fit(x[train],y[train]);pred[held]=model.predict(x[held])
            model.fit(x,y)
            row={'features':mode,'C':C,'group_cv_accuracy':float((pred==y).mean()),
                 'fit_accuracy_not_holdout':float((model.predict(x)==y).mean())}
            rows.append(row);print(row,flush=True)
    pd.DataFrame(rows).to_csv(out/'diagnostics.csv',index=False)


if __name__=='__main__':run()
