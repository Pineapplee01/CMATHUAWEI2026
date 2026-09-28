"""对齐版可解释：正向 Grad×Input（支持当前预测）、多上下文模态边际贡献、连续证据、删除复核、Comp/Suff/Stab。"""
from __future__ import annotations
import numpy as np
import torch
from problem3.model_io import predict, masked

NAMES = ('text', 'audio', 'vision')
VIEWS = {
    'T': (1, 0, 0), 'A': (0, 1, 0), 'V': (0, 0, 1),
    'TA': (1, 1, 0), 'TV': (1, 0, 1), 'AV': (0, 1, 1),
    'TAV': (1, 1, 1),
}
MARGINAL_PAIRS = {
    0: [('TA', 'A'), ('TV', 'V'), ('TAV', 'AV')],
    1: [('TA', 'T'), ('AV', 'V'), ('TAV', 'TV')],
    2: [('TV', 'T'), ('AV', 'A'), ('TAV', 'TA')],
}


def _grad_input(model, batch):
    """对投影后观测表示 H 做正向 Grad×Input：只保留抬高 s_{c*} 的贡献。

    与 problem2_retrain_v2.F1EmotionModel.forward 在 complete 之后的路径一致：
    槽均值 → 样本级模态门控 → cls_head。
    """
    from problem2_retrain_v2.model import modality_gate

    model.zero_grad(set_to_none=True)
    observed, P, M = model.encode_observed(batch)
    h0 = observed.detach().requires_grad_(True)
    if getattr(model, 'disable_completion', False):
        h = h0 * M[..., None]
        q = M.to(h0.dtype)
    else:
        h, q, _ = model.complete(h0, P, M)
    avail = P.any(-1)
    denom = P.sum(-1).to(h.dtype).clamp_min(1)
    hm = (h * P[..., None]).sum(2) / denom[..., None]
    hm = hm * avail[..., None].to(h.dtype)
    z, _alpha = modality_gate(hm, avail, model.gate_W, model.gate_w)
    logits = model.cls_head(z)
    c = int(logits[0].argmax())
    logits[0, c].backward()
    # r^+ = max(0, Σ_j H_j ∂s/∂H_j)：支持当前预测的槽；抑制项置 0
    signed = (h0 * h0.grad).sum(-1)[0].detach()
    r = torch.relu(signed).cpu().numpy()
    valid = (P & M)[0].cpu().numpy().astype(bool)
    r[~valid] = 0
    return r, valid, c, logits.detach()


def modality_attribution(r, valid):
    g = r.sum(-1)
    s = float(g.sum())
    i = g / s if s > 0 else np.zeros(3)
    return {'G': g.tolist(), 'I': i.tolist()}


def _view_batch(batch, flags):
    keep = (batch['O'] & batch['P']).clone()
    for m, on in enumerate(flags):
        if not on:
            keep[:, m] = False
    return masked(batch, keep[0])


def multi_context_marginal(model, batch, c_star):
    """非空上下文条件边际贡献 φ̃_m。"""
    avail = (batch['O'] & batch['P']).any(-1)[0].cpu().numpy().astype(bool)
    cache = {}
    for name, flags in VIEWS.items():
        need = [i for i, on in enumerate(flags) if on]
        if any(not avail[i] for i in need):
            continue
        out = predict(model, _view_batch(batch, flags))
        cache[name] = {'p': float(out['p'][c_star]), 'y': out['y']}

    phi_p, phi_y, details = [], [], {}
    for m in range(3):
        terms_p, terms_y, used = [], [], []
        if not avail[m]:
            phi_p.append(float('nan')); phi_y.append(float('nan'))
            details[NAMES[m]] = {'pairs': [], 'skipped': 'modality_unavailable'}
            continue
        for with_m, without in MARGINAL_PAIRS[m]:
            if with_m not in cache or without not in cache:
                continue
            terms_p.append(cache[with_m]['p'] - cache[without]['p'])
            terms_y.append(cache[with_m]['y'] - cache[without]['y'])
            used.append({'with': with_m, 'without': without,
                         'delta_p': terms_p[-1], 'delta_y': terms_y[-1]})
        phi_p.append(float(np.mean(terms_p)) if terms_p else float('nan'))
        phi_y.append(float(np.mean(terms_y)) if terms_y else float('nan'))
        details[NAMES[m]] = {'pairs': used, 'n': len(used)}

    phi = np.asarray(phi_p, dtype=float)
    support = None
    if np.isfinite(phi).any() and np.nanmax(phi) > 0:
        support = NAMES[int(np.nanargmax(phi))]
    full = cache.get('TAV', {})
    return {
        'phi_p': phi_p, 'phi_y': phi_y,
        'main_support': support,
        'views': cache,
        'details': details,
        'full_p': full.get('p'), 'full_y': full.get('y'),
        'method': 'nonempty_context_marginal',
    }


def continuous_evidence(r, valid, eta=0.7):
    r = np.asarray(r, float).copy()
    r[~valid] = 0
    total = float(r.sum())
    if total <= 0:
        return []
    order = np.argsort(-r)
    chosen = np.zeros_like(r, dtype=bool)
    mass = 0.0
    for i in order:
        if r[i] <= 0:
            break
        chosen[i] = True
        mass += r[i] / total
        if mass >= eta:
            break
    for i in range(1, len(chosen) - 1):
        if chosen[i - 1] and chosen[i + 1]:
            chosen[i] = True
    spans, i = [], 0
    while i < len(chosen):
        if not chosen[i]:
            i += 1
            continue
        j = i
        while j < len(chosen) and chosen[j]:
            j += 1
        spans.append((i, j))
        i = j
    return spans


def verify_spans(model, batch, c_star, spans_by_mod, y0, p0):
    rows = []
    base_keep = (batch['O'] & batch['P'])[0].clone()
    for m, spans in enumerate(spans_by_mod):
        for lo, hi in spans:
            keep = base_keep.clone()
            keep[m, lo:hi] = False
            out = predict(model, masked(batch, keep))
            rows.append({
                'modality': NAMES[m], 'start': lo, 'end': hi,
                'delta_p': p0 - float(out['p'][c_star]),
                'delta_y': abs(y0 - out['y']),
            })
    return rows


def comprehensiveness_sufficiency(model, batch, c_star, spans_by_mod, p0):
    keep_rm = (batch['O'] & batch['P'])[0].clone()
    keep_only = torch.zeros_like(keep_rm)
    for m, spans in enumerate(spans_by_mod):
        for lo, hi in spans:
            keep_rm[m, lo:hi] = False
            keep_only[m, lo:hi] = batch['O'][0, m, lo:hi] & batch['P'][0, m, lo:hi]
    p_rm = float(predict(model, masked(batch, keep_rm))['p'][c_star])
    p_sf = float(predict(model, masked(batch, keep_only))['p'][c_star])
    return {'comprehensiveness': p0 - p_rm, 'sufficiency': abs(p_sf - p0)}


def stability(model, batch, eta, repeats=3, noise=0.01):
    r0, valid, _, _ = _grad_input(model, batch)
    spans0 = {m: continuous_evidence(r0[m], valid[m], eta) for m in range(3)}
    scores = []
    for _ in range(repeats):
        b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        for key in ('XA', 'XV'):
            m = 1 if key == 'XA' else 2
            noise_t = noise * torch.randn_like(b[key]) * b['O'][:, m, :, None].float()
            b[key] = b[key] + noise_t
        r, v, _, _ = _grad_input(model, b)
        for m in range(3):
            a = _span_set(spans0[m])
            bset = _span_set(continuous_evidence(r[m], v[m], eta))
            scores.append(len(a & bset) / max(1, len(a | bset)))
    return float(np.mean(scores)) if scores else 0.0


def _span_set(spans):
    s = set()
    for lo, hi in spans:
        s.update(range(lo, hi))
    return s


def explain_sample(model, batch, tokens, cfg):
    eta = float(cfg.get('evidence_eta', 0.7))
    r, valid, c_star, _ = _grad_input(model, batch)
    attr = modality_attribution(r, valid)
    marg = multi_context_marginal(model, batch, c_star)
    spans = [continuous_evidence(r[m], valid[m], eta) for m in range(3)]
    p0 = marg['full_p'] if marg['full_p'] is not None else float(predict(model, batch)['p'][c_star])
    y0 = marg['full_y'] if marg['full_y'] is not None else predict(model, batch)['y']
    checks = verify_spans(model, batch, c_star, spans, y0, p0)
    cs = comprehensiveness_sufficiency(model, batch, c_star, spans, p0)
    stab = stability(model, batch, eta, repeats=int(cfg.get('stability_repeats', 3)),
                     noise=float(cfg.get('stability_noise', 0.01)))
    key = {
        NAMES[m]: [
            {'start': lo, 'end': hi,
             'tokens': tokens[lo:hi] if m == 0 else None,
             'mass': float(r[m, lo:hi].sum())}
            for lo, hi in spans[m]
        ]
        for m in range(3)
    }
    return {
        'class': c_star,
        'polarity': ('负', '中', '正')[c_star],
        'intensity': y0,
        'internal_attribution': {NAMES[m]: attr['I'][m] for m in range(3)},
        'main_support_modality': marg['main_support'],
        'most_sensitive_modality': NAMES[int(np.argmax(attr['I']))],
        'modality_marginal': marg,
        'key_evidence': key,
        'span_verification': checks,
        **cs,
        'stability': stab,
    }
