"""未对齐可解释：正向 Grad×Input（各轴独立）+ 多上下文模态边际贡献 + 连续证据 + soft association。"""
from __future__ import annotations
import numpy as np
import torch
from problem3_unaligned.model_io import predict, masked

NAMES = ('text', 'audio', 'vision')
LETTERS = ('T', 'A', 'V')
LENS = (50, 500, 500)
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


def _encode_feats(model, batch):
    """复现 SoftAlignment 前端，返回可反传的 E_T/E_A/E_V。"""
    P = [batch[f'P_{m}'].bool() for m in 'TAV']
    M = [p & batch[f'O_{m}'].bool() for p, m in zip(P, 'TAV')]
    token = batch['I'].clone()
    attention = M[0] | (token == 101) | (token == 102)
    token[~attention] = 0
    empty = ~attention.any(1)
    token[empty, 0] = 101
    attention[empty, 0] = True
    raw = model.bert(input_ids=token, attention_mask=attention.long()).last_hidden_state
    values = [(raw - model.text_mu) / model.text_sigma, batch['XA'], batch['XV']]
    h = []
    for layer, x, m in zip(model.project, values, M):
        t = (x * m[..., None]).detach().requires_grad_(True)
        h.append(layer(t) * m[..., None])
        h[-1].retain_grad()
        # 保留叶子供 Grad×Input：对投影前 x 归因
        h[-1]._gi_input = t  # type: ignore[attr-defined]
        h[-1]._gi_mask = m  # type: ignore[attr-defined]
    return h, P, M


def _grad_input(model, batch):
    model.zero_grad(set_to_none=True)
    h, P, M = _encode_feats(model, batch)
    # 手工走完后半段（与 SoftAlignment.forward 对齐）
    query = model.query_norm(h[0] + model.positions[0])
    query_mask = P[0].clone()
    fallback = ~query_mask.any(1)
    any_obs = M[1].any(1) | M[2].any(1)
    query_mask[fallback] = any_obs[fallback, None].expand(-1, 50)
    aligned = []
    n = batch['I'].shape[0]
    for j in range(2):
        m = j + 1
        length = P[m].long().sum(-1).clamp_min(1)
        centers, lo, hi = model._monotonic_windows(query, query_mask, length)
        memory = h[m] + model.positions[m]
        null = model.null[j:j + 1].expand(n, -1, -1)
        message, _ = model._windowed_cross(
            query, memory, M[m], query_mask, lo, hi, centers, length, null, j, False)
        aligned.append(model.norm[j](message) * M[m].any(1)[:, None, None])
    ratios = torch.stack([m.sum(-1) / p.sum(-1).clamp_min(1) for m, p in zip(M, P)], -1)
    fused = model.fuse(torch.cat([h[0], *aligned, ratios[:, None].expand(-1, 50, -1)], -1))
    scores = model.pool_score(fused).squeeze(-1).masked_fill(~query_mask, -1e4)
    pool = scores.softmax(-1) * query_mask
    pool = pool / pool.sum(-1, keepdim=True).clamp_min(1e-8)
    z = (fused * pool[..., None]).sum(1)
    out = model.head(torch.cat([z, ratios], -1))
    c = int(out[0, :3].argmax())
    out[0, c].backward()

    rs, valids = [], []
    for feat, length in zip(h, LENS):
        g = feat._gi_input.grad
        x = feat._gi_input
        m = feat._gi_mask
        r = torch.relu((x * g).sum(-1)[0].detach()).cpu().numpy()
        v = m[0].detach().cpu().numpy().astype(bool)
        r[~v] = 0
        rs.append(r)
        valids.append(v)
    return rs, valids, c


def modality_attribution(rs):
    g = np.array([float(r.sum()) for r in rs])
    s = g.sum()
    i = g / s if s > 0 else np.zeros(3)
    return i


def _view_batch(batch, flags):
    b = {k: v.clone() for k, v in batch.items()}
    for m, on in enumerate(flags):
        if not on:
            b[f'O_{LETTERS[m]}'].zero_()
    return b


def multi_context_marginal(model, batch, c_star):
    """非空上下文条件边际贡献 φ̃_m（各轴整段开关观测）。"""
    avail = [bool(batch[f'O_{L}'].any()) for L in LETTERS]
    cache = {}
    for name, flags in VIEWS.items():
        if any(on and not avail[i] for i, on in enumerate(flags)):
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
    r = r[:len(valid)]
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


def soft_associate(attn_50x500, spans_av, topk=5):
    """全局：文本位置 ← 全部音/视关键段的 soft alignment 质量（兼容旧字段）。"""
    if attn_50x500 is None or not spans_av:
        return []
    q = np.zeros(attn_50x500.shape[0])
    for lo, hi in spans_av:
        q += attn_50x500[:, lo:hi].sum(-1)
    idx = np.argsort(-q)[:topk]
    return [{'text_index': int(i), 'score': float(q[i])} for i in idx if q[i] > 0]


def soft_associate_spans(attn_50x500, spans_av, topk=2, mass_cover=0.7):
    """逐段：每个音/视证据窗关联到的文本词元下标与首尾锚点。

    单槽窗：只取该槽注意力最大的一个词元。
    多槽窗：首、尾槽各自取注意力最大词元作时间锚点；质量 Top 词元用于展示。
    """
    if attn_50x500 is None or not spans_av:
        return []
    out = []
    attn = np.asarray(attn_50x500, dtype=np.float64)
    for lo, hi in spans_av:
        if hi <= lo or lo < 0 or hi > attn.shape[1]:
            continue
        if hi - lo == 1:
            col = attn[:, lo]
            i = int(np.argmax(col)) if col.max() > 0 else None
            chosen = [i] if i is not None else []
            scores = [float(col[i])] if i is not None else []
            i_lo = i_hi = i
        else:
            q = attn[:, lo:hi].sum(-1)
            total = float(q[q > 0].sum())
            chosen, scores = [], []
            if total > 0:
                acc = 0.0
                for i in np.argsort(-q):
                    if q[i] <= 0:
                        break
                    chosen.append(int(i))
                    scores.append(float(q[i]))
                    acc += float(q[i])
                    if len(chosen) >= topk or acc / total >= mass_cover:
                        break
            i_lo = int(np.argmax(attn[:, lo])) if attn[:, lo].max() > 0 else (chosen[0] if chosen else None)
            i_hi = int(np.argmax(attn[:, hi - 1])) if attn[:, hi - 1].max() > 0 else (chosen[-1] if chosen else None)
        out.append({
            'start': int(lo), 'end': int(hi),
            'text_indices': [i for i in chosen if i is not None],
            'scores': scores,
            'endpoint_text': [i_lo, i_hi],
        })
    return out



def verify_and_cs(model, batch, c_star, spans, p0, y0):
    checks = []
    keep_rm = {f'O_{L}': batch[f'O_{L}'].clone() for L in LETTERS}
    keep_only = {f'O_{L}': torch.zeros_like(batch[f'O_{L}']) for L in LETTERS}
    for m, letter in enumerate(LETTERS):
        for lo, hi in spans[m]:
            b = masked(batch, spans={letter: [(lo, hi)]})
            out = predict(model, b)
            checks.append({
                'modality': NAMES[m], 'start': lo, 'end': hi,
                'delta_p': p0 - float(out['p'][c_star]),
                'delta_y': abs(y0 - out['y']),
            })
            keep_rm[f'O_{letter}'][:, lo:hi] = False
            keep_only[f'O_{letter}'][:, lo:hi] = batch[f'O_{letter}'][:, lo:hi]
    b_rm = dict(batch)
    b_rm.update(keep_rm)
    b_sf = dict(batch)
    b_sf.update(keep_only)
    p_rm = float(predict(model, b_rm)['p'][c_star])
    p_sf = float(predict(model, b_sf)['p'][c_star])
    return checks, {'comprehensiveness': p0 - p_rm, 'sufficiency': abs(p_sf - p0)}


def _span_set(spans):
    s = set()
    for lo, hi in spans:
        s.update(range(lo, hi))
    return s


def stability(model, batch, eta, repeats=3, noise=0.01):
    """对 XA/XV 在观测位加噪声，重复抽证据窗，与基线做 Jaccard 平均。"""
    rs0, valids0, _ = _grad_input(model, batch)
    spans0 = {m: continuous_evidence(rs0[m], valids0[m], eta) for m in range(3)}
    scores = []
    for _ in range(repeats):
        b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        for key, letter in (('XA', 'A'), ('XV', 'V')):
            mask = b[f'O_{letter}'][..., None].float()
            b[key] = b[key] + noise * torch.randn_like(b[key]) * mask
        rs, valids, _ = _grad_input(model, b)
        for m in range(3):
            a = _span_set(spans0[m])
            bset = _span_set(continuous_evidence(rs[m], valids[m], eta))
            scores.append(len(a & bset) / max(1, len(a | bset)))
    return float(np.mean(scores)) if scores else 0.0


def explain_sample(model, batch, tokens, cfg):
    eta = float(cfg.get('evidence_eta', 0.7))
    topk = int(cfg.get('associate_topk', 5))
    rs, valids, c_star = _grad_input(model, batch)
    I = modality_attribution(rs)
    marg = multi_context_marginal(model, batch, c_star)
    spans = [continuous_evidence(rs[m], valids[m], eta) for m in range(3)]
    p0 = marg['full_p'] if marg['full_p'] is not None else float(predict(model, batch)['p'][c_star])
    y0 = marg['full_y'] if marg['full_y'] is not None else predict(model, batch)['y']
    checks, cs = verify_and_cs(model, batch, c_star, spans, p0, y0)
    stab = stability(model, batch, eta, repeats=int(cfg.get('stability_repeats', 3)),
                     noise=float(cfg.get('stability_noise', 0.01)))
    attn = predict(model, batch, return_attention=True)
    assoc = {
        'audio_to_text': soft_associate(attn.get('attention_audio'), spans[1], topk),
        'vision_to_text': soft_associate(attn.get('attention_vision'), spans[2], topk),
        'audio_span_links': soft_associate_spans(attn.get('attention_audio'), spans[1], topk=2),
        'vision_span_links': soft_associate_spans(attn.get('attention_vision'), spans[2], topk=2),
    }
    key = {
        NAMES[m]: [
            {'start': lo, 'end': hi,
             'tokens': tokens[lo:hi] if m == 0 else None,
             'mass': float(rs[m][lo:hi].sum())}
            for lo, hi in spans[m]
        ]
        for m in range(3)
    }
    return {
        'class': c_star,
        'polarity': ('负', '中', '正')[c_star],
        'intensity': y0,
        'internal_attribution': {NAMES[m]: float(I[m]) for m in range(3)},
        'main_support_modality': marg['main_support'],
        'most_sensitive_modality': NAMES[int(np.argmax(I))],
        'modality_marginal': marg,
        'key_evidence': key,
        'soft_association': assoc,
        'span_verification': checks,
        **cs,
        'stability': stab,
    }
