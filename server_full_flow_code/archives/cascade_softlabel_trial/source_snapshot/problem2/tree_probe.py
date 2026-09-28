"""固定表示的树模型诊断，训练只用train，valid只做早停。"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from shared.common import path


def run():
    with np.load(path('AAAdata/processed/kernel_probe_sst2.npz')) as d:
        x=np.concatenate([d[f'train{m}'] for m in range(3)],-1)
        v=np.concatenate([d[f'valid{m}'] for m in range(3)],-1)
        y,z=d['y'],d['z']
    out=path('problem2/results/tree_probe');out.mkdir(exist_ok=True)
    rows=[]
    for depth in [3,5,7]:
        model=CatBoostClassifier(iterations=1800,depth=depth,learning_rate=.03,
            l2_leaf_reg=15,loss_function='MultiClass',eval_metric='Accuracy',
            random_seed=2026,thread_count=4,allow_writing_files=False,verbose=False)
        model.fit(x,y,eval_set=(v,z),early_stopping_rounds=120)
        pred=model.predict(v).reshape(-1)
        row={'depth':depth,'trees':model.tree_count_,'accuracy':float((pred==z).mean())}
        rows.append(row);print(row,flush=True)
        model.save_model(str(path(f'AAAmodel/checkpoints/tree_probe_d{depth}.cbm')))
        np.save(out/f'logits_d{depth}.npy',model.predict(v,prediction_type='RawFormulaVal'))
        pd.DataFrame(rows).to_csv(out/'diagnostics.csv',index=False)


if __name__=='__main__':run()
