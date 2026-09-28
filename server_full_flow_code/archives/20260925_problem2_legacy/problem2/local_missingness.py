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


def scenario_grid(seeds=(2026, 2027, 2028)):
    for name in ('T', 'A', 'V', 'TA', 'TV', 'AV', 'TAV'):
        modalities = tuple('TAV'.index(m) for m in name)
        for rate in (.1, .3, .5, .7):
            for position in ('start', 'middle', 'end', 'random'):
                for seed in (seeds if position == 'random' else seeds[:1]):
                    yield name, modalities, rate, position, seed
