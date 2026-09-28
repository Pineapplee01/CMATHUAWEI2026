"""局部连续缺失协议：保留至少一个原可见位置，记录实际槽长而非虚构秒数。"""
import numpy as np
import torch


def local_mask(batch, rate, modalities, position, seed, offset=0):
    if not 0 <= rate < 1 or position not in ('start', 'middle', 'end', 'random'):
        raise ValueError('局部缺失要求0≤rate<1和明确位置')
    keep = torch.ones_like(batch['O'])
    records = []
    for i in range(len(keep)):
        for m in modalities:
            valid = torch.where(batch['P'][i, m])[0].cpu().numpy()
            observed = batch['O'][i, m].detach().cpu().numpy()
            before = int(observed.sum())
            start, end = 0, 0
            if len(valid) and rate and before >= 2:
                lo, hi = int(valid[0]), int(valid[-1]) + 1
                width = min(hi-lo, max(1, round(rate*(hi-lo))))
                if position == 'random':
                    rng = np.random.default_rng(np.random.SeedSequence([seed, offset+i, m]))
                    start = int(rng.integers(lo, hi-width+1))
                else:
                    start = {'start':lo, 'middle':lo+(hi-lo-width)//2, 'end':hi-width}[position]
                end = start + width
                # 极短/稀疏样本可能在小于100%的区间中包含全部观测；缩短而非整模态删除。
                while end > start and observed[start:end].sum() == before:
                    if position == 'end': start += 1
                    else: end -= 1
                keep[i, m, start:end] = False
            removed = int(observed[start:end].sum())
            records.append({'id':batch['id'][i], 'modality':'TAV'[m],
                            'start_slot':start, 'end_slot_exclusive':end,
                            'interval_slots':end-start, 'removed_observed_slots':removed,
                            'observed_before':before, 'observed_after':before-removed,
                            'actual_removed_fraction':removed/max(before, 1),
                            'too_few_observations':bool(rate and before < 2)})
    return keep, records


def scenario_grid(seeds=(2026, 2027, 2028), rates=(.1, .3, .5, .7)):
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError("掩码seed必须非空且唯一")
    if not rates or len(set(rates))!=len(rates) or any(not 0<r<1 for r in rates):
        raise ValueError("场景跨度必须互不重复且在(0,1)内")
    for name in ('T', 'A', 'V', 'TA', 'TV', 'AV', 'TAV'):
        modalities = tuple('TAV'.index(m) for m in name)
        for rate in rates:
            for position in ('start', 'middle', 'end', 'random'):
                for seed in (seeds if position == 'random' else seeds[:1]):
                    yield name, modalities, rate, position, seed


def synchronized_view(data, rate, modalities, position, seed):
    """研究实验：选定模态共用一个连续槽区间；原正式F1缺失协议不改变。"""
    if not 0<=rate<1 or position not in ('start','middle','end','random'):
        raise ValueError('非法局部缺失参数')
    if not modalities or any(m not in (0,1,2) for m in modalities):raise ValueError('非法模态组合')
    keep=torch.ones_like(data['O']);records=[]
    for i,sid in enumerate(data['id']):
        valid=torch.where(data['P'][i,list(modalities)].any(0))[0].cpu().numpy()
        start=end=0
        if len(valid) and rate:
            lo,hi=int(valid[0]),int(valid[-1])+1;width=min(hi-lo,max(1,round(rate*(hi-lo))))
            rng=np.random.default_rng(np.random.SeedSequence([seed,i]))
            start={'start':lo,'middle':lo+(hi-lo-width)//2,'end':hi-width,
                   'random':int(rng.integers(lo,hi-width+1))}[position];end=start+width
            # 对整组缩短同一区间，保证每个原非空模态至少保留一个真实观测。
            while end>start and any(int(data['O'][i,m].sum())>0 and
                    int(data['O'][i,m,start:end].sum())==int(data['O'][i,m].sum()) for m in modalities):
                if position=='end':start+=1
                else:end-=1
            for m in modalities:keep[i,m,start:end]=False
        for m in modalities:
            before=int(data['O'][i,m].sum());removed=int(data['O'][i,m,start:end].sum())
            records.append({'id':sid,'modality':'TAV'[m],'start_slot':start,'end_slot_exclusive':end,
                            'observed_before':before,'observed_after':before-removed,
                            'removed_observed_slots':removed,'actual_removed_fraction':removed/max(before,1)})
    return data|{'O':data['O']&keep},records
