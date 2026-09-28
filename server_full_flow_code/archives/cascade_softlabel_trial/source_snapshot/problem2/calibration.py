"""验证集上的三分类决策偏置：不改强度、不读取test、不增加网络参数。
偏置含在原生贡献账本；另报告按原视频分组的两折交叉验证，暴露调阈值过拟合。
"""
import argparse
import itertools
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from shared.common import config, path, to_device, json_write
from problem2.runtime import restore, dataset, loader


def fit_bias(logits, labels):
    candidates = np.array([(a,b,0.) for a,b in itertools.product(np.linspace(-1,1,21),repeat=2)])
    accuracy = ((logits[None]+candidates[:,None]).argmax(-1)==labels).mean(-1)
    # 最优准确率同分时取最小改动；范围/网格在读结果前固定。
    best = np.flatnonzero(accuracy==accuracy.max())
    i=best[np.argmin(np.square(candidates[best]).sum(-1))]
    return candidates[i], float(accuracy[i])


@torch.no_grad()
def calibrate(cfg, destination):
    model, saved=restore(cfg['checkpoint'],cfg['device'])
    ds=dataset(cfg,'valid'); logits=[]; labels=[]
    for raw in loader(ds,cfg):
        b=to_device(raw,cfg['device'])
        logits.append(model(b)['logits'].cpu().numpy());labels.extend(raw['c'].tolist())
    logits=np.concatenate(logits);labels=np.array(labels)
    bias,fit_accuracy=fit_bias(logits,labels)
    original_accuracy=float((logits.argmax(-1)==labels).mean())
    groups=np.array([s.split('$_$')[0] for s in ds.ids]); held_predictions=np.empty_like(labels)
    folds=[]
    for train,held in GroupKFold(2).split(logits,labels,groups):
        b,_=fit_bias(logits[train],labels[train]);held_predictions[held]=(logits[held]+b).argmax(-1)
        folds.append({'bias':b.tolist(),'held_count':len(held)})
    report={'fit_split':'valid','never_reads_test':True,'grid':[-1,1,.1],
            'uncalibrated_accuracy':original_accuracy,'selected_validation_accuracy':fit_accuracy,
            'group_two_fold_accuracy':float((held_predictions==labels).mean()),
            'bias':bias.tolist(),'folds':folds,'selection_optimism_note':'validation used to fit decision thresholds'}
    checkpoint=torch.load(path(cfg['checkpoint']),map_location='cpu',weights_only=False)
    state=checkpoint['model']; old=state.get('head.decision_bias',torch.zeros(4))
    state['head.decision_bias']=old+torch.tensor([*bias,0],dtype=torch.float32)
    checkpoint['calibration']=report
    path(destination).parent.mkdir(parents=True,exist_ok=True);torch.save(checkpoint,path(destination))
    json_write(path(cfg['output'])/'decision_bias_selection.json',report)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/default.json')
    p.add_argument('--output-checkpoint',required=True)
    args=p.parse_args();print(calibrate(config(args.config),args.output_checkpoint))


if __name__=='__main__':main()
