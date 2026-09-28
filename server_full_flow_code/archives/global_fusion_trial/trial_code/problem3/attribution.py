"""精确模态子集归因、局部删除与稳定性。

推导3-1～3-13：固定完整预测类，8子集求精确来源Shapley；
账本另在logit空间验算；窗口删除必须重新编码及补全。
小扰动排序检查不是因果证明，也不是最小反事实搜索。
"""
import math
import numpy as np
import pandas as pd
import torch
from shared.common import MODALITIES, LABELS



def target(output, c):
    return torch.stack([output['logits'].softmax(-1)[0, c], output['intensity'][0]])



@torch.no_grad()
def explain_one(model, batch, cfg, mapping=None):
    base = model(batch)
    c = int(base['logits'][0].argmax())
    original = target(base, c)
    values = {}
    for subset in range(8):
        B = torch.zeros_like(batch['O'])
        for m in range(3):
            if subset & (1 << m): B[:, m] = True
        values[subset] = target(model(batch, B), c)
    phi = torch.zeros(3, 2, device=original.device)
    for m in range(3):
        for subset in range(8):
            if subset & (1 << m): continue
            size = subset.bit_count()
            weight = math.factorial(size) * math.factorial(2-size) / math.factorial(3)
            phi[m] += weight * (values[subset | (1 << m)] - values[subset])
    shapley_error = (phi.sum(0) - (values[7] - values[0])).abs()
    if shapley_error.max() > 1e-5:
        raise AssertionError('Shapley 完整性误差超限')
    denom = phi.abs().sum(0)
    degree = phi.abs() / denom.clamp_min(1e-8)
    dominant = MODALITIES[int(degree[:, 0].argmax())] if denom[0] > 1e-8 else None
    ledger = base['ledger']
    reconstructed = ledger['bias'] + ledger['single'].sum((1, 2)) + ledger['pair'].sum((1, 2))
    reconstructed = reconstructed + ledger.get('text_prior', 0)
    reconstructed = reconstructed + ledger.get('global_fusion', 0)
    if 'triple' in ledger: reconstructed = reconstructed + ledger['triple'].sum(1)
    actual = torch.cat([base['logits'], base['score'][:, None]], 1)
    ledger_error = float((reconstructed - actual).abs().max())
    if ledger_error > 1e-5:
        raise AssertionError('原生贡献账本无法重建实际预测')
    evidence = []
    evidence_window = cfg.get('evidence_window', cfg['window'])
    for m, name in enumerate(MODALITIES):
        for lo in range(0, 50, evidence_window):
            hi = min(50, lo + evidence_window)
            observed = batch['O'][0, m, lo:hi] & batch['P'][0, m, lo:hi]
            if not observed.any(): continue
            B = torch.ones_like(batch['O']); B[:, m, lo:hi] = False
            deleted = target(model(batch, B), c)
            times = [] if mapping is None else mapping.get(name, [])[lo:hi]
            has_timing = any(item is not None for item in times)
            evidence.append({'modality': name, 'start_index': lo, 'end_index_exclusive': hi,
                             'removed_observations': int(observed.sum()),
                             'probability_drop': float(original[0] - deleted[0]),
                             'intensity_change': float(original[1] - deleted[1]),
                             'timing': times, 'timing_status': 'see_mapping' if has_timing else 'unavailable',
                             'recomputed_completion': True})
    # 等宽单块随机核验，与最大正向删除效应对比；不宣称全局最小。
    fidelity = {}
    rng = np.random.default_rng(cfg['seed'])
    for m, name in enumerate(MODALITIES):
        candidates = [r for r in evidence if r['modality'] == name]
        if not candidates: continue
        best = max(candidates, key=lambda r: r['probability_drop'])
        width = best['end_index_exclusive'] - best['start_index']
        count = best['removed_observations']
        starts = [lo for lo in range(51-width)
                  if int((batch['P'][0, m, lo:lo+width] & batch['O'][0, m, lo:lo+width]).sum()) == count]
        random_effects = []
        for lo in rng.choice(starts, min(10, len(starts)), replace=False):
            B = torch.ones_like(batch['O']); B[:, m, int(lo):int(lo)+width] = False
            random_effects.append(float(original[0] - target(model(batch, B), c)[0]))
        fidelity[name] = {'key_drop': best['probability_drop'], 'random_drops': random_effects,
                          'width': width, 'matched_observed_count': count}
    # 轻微音视频扰动后重算同一窗口删除排名，文本词元保持原离散语义。
    perturbed = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
    generator = torch.Generator(device=original.device).manual_seed(cfg['seed'] + 77)
    for m, name in enumerate(('audio', 'vision'), 1):
        noise = torch.randn(batch[name].shape, generator=generator, device=original.device)
        perturbed[name] += .01 * noise * getattr(model, name + '_std') * batch['O'][:, m, :, None]
    perturbed_base = target(model(perturbed), c)
    for e in evidence:
        B = torch.ones_like(batch['O'])
        m = MODALITIES.index(e['modality'])
        B[:, m, e['start_index']:e['end_index_exclusive']] = False
        e['perturbed_probability_drop'] = float(perturbed_base[0] - target(model(perturbed, B), c)[0])
    ranking = {}
    for name in MODALITIES:
        rows = [e for e in evidence if e['modality'] == name]
        if len(rows) >= 2:
            a = pd.Series([e['probability_drop'] for e in rows]).rank()
            b = pd.Series([e['perturbed_probability_drop'] for e in rows]).rank()
            r = a.corr(b)
            ranking[name] = float(r) if pd.notna(r) else None
    return {'class': c, 'polarity': LABELS[c], 'intensity': float(base['intensity'][0]),
            'head_consistent': c == int(torch.sign(base['intensity'][0])) + 1,
            'fixed_target_class': c, 'target_scales': ['class_probability', 'intensity'],
            'source_shapley': phi, 'source_degree': degree, 'dominant_modality': dominant,
            'coalition_outputs': {str(k): v for k, v in values.items()},
            'shapley_efficiency_error': shapley_error, 'low_evidence': bool(base['low_evidence'][0]),
            'ledger': ledger, 'ledger_reconstruction_max_error': ledger_error,
            'ledger_scale': ['negative_logit', 'neutral_logit', 'positive_logit', 'pre_tanh_score'],
            'ledger_pair_order': ['TA', 'TV', 'AV'],
            'text_prior_scope': 'all retained text, separate explicit additive term' if 'text_prior' in ledger else None,
            'observed_mask': base['M'][0], 'valid_mask': base['P'][0],
            'completed_mask': (base['P'] & ~base['M'])[0], 'reliability': base['q'][0],
            'completion_readable_sources': {name: torch.where(base['M'][0, m])[0]
                                            for m, name in enumerate(MODALITIES)},
            'support_scope': 'all retained observations; text encoder sees retained sentence tokens after masking',
            'evidence': evidence, 'fidelity': fidelity, 'perturbation_rank_spearman': ranking,
            'validity_uncertain': True, 'mask_rule': cfg['mask_rule']}
