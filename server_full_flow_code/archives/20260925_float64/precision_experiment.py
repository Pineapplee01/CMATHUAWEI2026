"""固定已提取特征，比较预测网络/损失/梯度/AdamW累积量的float32与float64。"""
from pathlib import Path
import json, hashlib
import numpy as np
import torch
from problem2.data import load_split, masked_view, MissingTextEncoder, ROOT_DATA, sha256
from problem2.training import train_candidate
from problem2.model import AlignedEmotionModel, objective
from shared.common import path, json_write, seed_all


def typed(data,dtype):
    return {k:v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v for k,v in data.items()}


def main():
    device='cuda:3';torch.set_num_threads(4)
    root=path('problem2/results/precision');root.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(path('configs/problem2_aligned.json').read_text())
    json_write(root/'protocol.json',{'selection':'0.7 valid full accuracy + 0.3 valid TAV30 accuracy',
             'test_used':False,'seed':2026,'epochs':40,'patience':10,'device':device,
             'existing_checkpoint_sha256':sha256(path(cfg['checkpoint'])),
             'scope':'同一冻结预计算BERT特征；预测网络、loss、梯度、AdamW矩全部采用指定精度。BERT不重新提取成另一组特征。',
             'datasets':['processed_float64','processed_po']})
    encoder=MissingTextEncoder(device);rows=[]
    for name in ('processed_float64','processed_po'):
        directory=path(ROOT_DATA)/name
        # 直接保留磁盘float64数据，不先转float32；po从原始float32精确升为double。
        train64=load_split(directory,'train',np.float64);valid64=load_split(directory,'valid',np.float64)
        reference=typed(train64,torch.float32);vreference=typed(valid64,torch.float32)
        views=[masked_view(reference,encoder,rate,(0,1,2),'random',seed)[0]
               for rate,seed in ((.1,3026),(.3,3027),(.5,3028))]
        vview=masked_view(vreference,encoder,.3,(0,1,2),'random',4026)[0]
        hashes=[]
        for precision in ('float32','float64'):
            dtype=getattr(torch,precision);seed_all(2026)
            initial=AlignedEmotionModel(cfg).to(dtype=dtype)
            h=hashlib.sha256()
            for value in initial.state_dict().values():h.update(value.float().numpy().tobytes())
            hashes.append(h.hexdigest())
            # 实测forward/loss/backward及优化器累积量，排除只转换文件类型。
            from problem2.data import batch
            b=batch(typed(train64,dtype),torch.arange(4),'cpu')
            output=initial(b);loss=objective(output,b);loss.backward()
            opt=torch.optim.AdamW(initial.parameters());opt.step()
            assert output['logits'].dtype==dtype and loss.dtype==dtype
            assert all(p.dtype==dtype and (p.grad is None or p.grad.dtype==dtype) for p in initial.parameters())
            assert all(st['exp_avg'].dtype==dtype and st['exp_avg_sq'].dtype==dtype for st in opt.state.values())
            del initial,opt,b,output,loss
            run=f'{name}_{precision}'
            candidate=cfg|{'name':run,'precision':precision,'data_dir':str(directory),
                  'checkpoint':f'AAAmodel/checkpoints/precision/{run}.pt','output':f'problem2/results/precision/{run}'}
            train=typed(train64,dtype);valid=typed(valid64,dtype)
            partial=[]
            for view in views:
                # 未改变的XA/XV保持与对应完整视图相同的原始精度；只复用同一XT重编码结果。
                partial.append({**train,'O':view['O'],'XT':view['XT'].to(dtype)})
            vp={**valid,'O':vview['O'],'XT':vview['XT'].to(dtype)}
            result=train_candidate(candidate,train,valid,partial,vp)
            result['initial_weights_sha256_as_float32']=hashes[-1]
            result['dtype_forward_loss_grad_optimizer_verified']=True
            rows.append(result);json_write(root/'comparison.json',rows)
        assert hashes[0]==hashes[1], '初始权重不一致'
        del train64,valid64,reference,vreference,views,vview,train,valid,partial,vp
    print('DONE',flush=True)


if __name__=='__main__':main()
