"""只用train/valid诊断词汇证据的可分性；不读取test或专项集。"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from shared.common import config,path
from problem2.runtime import dataset


def token_documents(ds):
    return [' '.join(str(int(x)) for x in t[0,t[1].astype(bool)] if x not in (0,101,102))
            for t in ds.fields['text_bert']]


def main():
    cfg=config('configs/default.json'); train=dataset(cfg,'train'); valid=dataset(cfg,'valid')
    rows=[]
    for ngram in [(1,1),(1,2)]:
        v=TfidfVectorizer(token_pattern=r'\b\d+\b',ngram_range=ngram,min_df=3,sublinear_tf=True)
        x=v.fit_transform(token_documents(train));z=v.transform(token_documents(valid))
        for strength in [.1,1.,10.]:
            model=LogisticRegression(C=strength,max_iter=1000)
            model.fit(x,train.fields['classification_labels']);pred=model.predict(z)
            rows.append({'ngram':str(ngram),'C':strength,'features':x.shape[1],
                         'accuracy':accuracy_score(valid.fields['classification_labels'],pred),
                         'macro_f1':f1_score(valid.fields['classification_labels'],pred,average='macro')})
    dest=path('problem2/results/diagnostics');dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(dest/'lexical_probe.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
