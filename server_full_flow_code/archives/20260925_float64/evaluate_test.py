"""按用户要求比较已冻结候选的test成绩；不训练、不挑epoch，记录test参与选择。"""
import json
from datetime import datetime,timezone
import numpy as np
import torch
from shared.common import path,json_write
from problem2.data import load_split,masked_view,MissingTextEncoder,sha256
from problem2.training import restore,evaluate


def main():
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    rows=json.loads(path('problem2/results/precision/comparison.json').read_text())
    root=path('problem2/results/precision/test_comparison');root.mkdir(parents=True,exist_ok=True)
    current=json.loads(path('configs/problem2_aligned.json').read_text())
    protocol={'requested_by_user':True,'test_used_for_comparison':True,
              'selection_criterion':'test complete-observation three-class accuracy; ties retain current model',
              'training_or_epoch_selection_in_this_run':False,'started_at':datetime.now(timezone.utc).isoformat(),
              'current_checkpoint_sha256':sha256(path(current['checkpoint'])),
              'candidates':[{k:r[k] for k in ('name','checkpoint','checkpoint_sha256','best_epoch')} for r in rows]}
    json_write(root/'protocol.json',protocol)
    encoder=MissingTextEncoder('cuda:3');results=[]
    for row in rows:
        checkpoint=path(row['checkpoint'])
        assert sha256(checkpoint)==row['checkpoint_sha256']
        model,cfg=restore(checkpoint,'cuda:3')
        dtype=getattr(np,row['precision'])
        data=load_split(cfg['data_dir'],'test',dtype=dtype)
        full,frame=evaluate(model,data,'cuda:3')
        dest=root/row['name'];dest.mkdir(parents=True,exist_ok=True)
        frame.to_csv(dest/'test_full.csv',index=False)
        partial,masks=masked_view(data,encoder,.3,(0,1,2),'random',4026)
        missing,mf=evaluate(model,partial,'cuda:3')
        mf.to_csv(dest/'test_local30.csv',index=False);json_write(dest/'mask.json',masks)
        result={'name':row['name'],'dataset':row['dataset'],'precision':row['precision'],
                'checkpoint':row['checkpoint'],'checkpoint_sha256':row['checkpoint_sha256'],
                'complete':full,'local30':missing,'samples':len(frame),
                'full_correct':int((frame.true_class==frame['class']).sum()),
                'missing_correct':int((mf.true_class==mf['class']).sum())}
        results.append(result);json_write(root/'comparison.json',results)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    print('DONE',flush=True)


if __name__=='__main__':main()
