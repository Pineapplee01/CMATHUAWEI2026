#!/usr/bin/env python3
# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：GPT-6 Astra
# 版本/型号：GPT-6 Astra（gpt-6-astra）
# 开发机构/公司：OpenAI
# 版本颁布日期：2026年9月3日
# 发布信息来源：https://openai.com/index/safety-overview-gpt-6-astra/

"""问题三对齐版：加载问题二 AlignedModel，对附件4做完整可解释。

用法（在 CPMCM 根目录）:
  python AAA提交版代码及结果/问题三/代码/aligned.py [--smoke] [--device cuda:0]
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

MODULE_DIR = Path(__file__).resolve().parent
from cpmcm_huawei2026.explanation.q3_utils import resolve_data_config, path, seed_all, json_write, LABELS
from cpmcm_huawei2026.preprocessing.appendix4_aligned import appendix_batch, masked, project_intensity

from cpmcm_huawei2026.prediction.problem2_aligned import AlignedModel, load_split
from cpmcm_huawei2026.prediction.problem2_aligned import batch as batch_fn

NAMES = ('text', 'audio', 'vision')
VIEWS = {
    'T': (1, 0, 0), 'A': (0, 1, 0), 'V': (0, 0, 1),
    'TA': (1, 1, 0), 'TV': (1, 0, 1), 'AV': (0, 1, 1), 'TAV': (1, 1, 1),
}
MARGINAL_PAIRS = {
    0: [('TA', 'A'), ('TV', 'V'), ('TAV', 'AV')],
    1: [('TA', 'T'), ('AV', 'V'), ('TAV', 'TV')],
    2: [('TV', 'T'), ('AV', 'A'), ('TAV', 'TA')],
}


def sha256(filename):
    h = hashlib.sha256()
    with Path(filename).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_model(cfg):
    ckpt = path(cfg['checkpoint'])
    digest = sha256(ckpt)
    saved = torch.load(ckpt, map_location='cpu', weights_only=False)
    if saved.get('format') != 'aligned_v4':
        raise ValueError('需要对齐 aligned_v4 检查点')
    train = resolve_data_config(saved['config'], '对齐版本')
    device = cfg.get('device', 'cpu')
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'; cfg['device'] = device
    model = AlignedModel(train, pretrained=False).to(device)
    model.load_state_dict(saved['model'], strict=True); model.eval()
    return model, train, saved, digest


@torch.no_grad()
def predict(model, batch):
    out = model(batch)
    p = out['logits'].softmax(-1)[0]
    c = int(p.argmax()); raw = float(out['intensity'][0])
    y = float(project_intensity(np.array([c]), np.array([raw]))[0])
    return {'p': p.cpu().numpy(), 'c': c, 'y': y, 'raw': raw}


def _grad_input(model, batch):
    """对投影后观测 E 做正向 Grad×Input，记 g_{m,t}=ReLU(Σ_j E·∂s/∂E)；勿与补全可靠性 r 混淆。"""
    model.zero_grad(set_to_none=True)
    observed, P, M = model.encode_observed(batch)
    e0 = observed.detach().requires_grad_(True)
    h, q, _ = model.complete(e0, P, M)
    out = model.fuse_heads(h, P)
    logits = out['logits']
    c = int(logits[0].argmax()); logits[0, c].backward()
    signed = (e0 * e0.grad).sum(-1)[0].detach()
    g = torch.relu(signed).cpu().numpy()
    valid = (P & M)[0].cpu().numpy().astype(bool)
    g[~valid] = 0
    return g, valid, c, logits.detach()


def modality_attribution(g, valid):
    mass = g.sum(-1); s = float(mass.sum())
    i = mass / s if s > 0 else np.zeros(3)
    return {'G': mass.tolist(), 'I': i.tolist()}


def _view_batch(batch, flags):
    keep = (batch['O'] & batch['P']).clone()
    for m, on in enumerate(flags):
        if not on:
            keep[:, m] = False
    return masked(batch, keep[0])


def multi_context_marginal(model, batch, c_star):
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
            details[NAMES[m]] = {'pairs': [], 'skipped': 'modality_unavailable'}; continue
        for with_m, without in MARGINAL_PAIRS[m]:
            if with_m not in cache or without not in cache:
                continue
            terms_p.append(cache[with_m]['p'] - cache[without]['p'])
            terms_y.append(cache[with_m]['y'] - cache[without]['y'])
            used.append({'with': with_m, 'without': without, 'delta_p': terms_p[-1], 'delta_y': terms_y[-1]})
        phi_p.append(float(np.mean(terms_p)) if terms_p else float('nan'))
        phi_y.append(float(np.mean(terms_y)) if terms_y else float('nan'))
        details[NAMES[m]] = {'pairs': used, 'n': len(used)}
    phi = np.asarray(phi_p, float)
    support = NAMES[int(np.nanargmax(phi))] if np.isfinite(phi).any() and np.nanmax(phi) > 0 else None
    full = cache.get('TAV', {})
    return {'phi_p': phi_p, 'phi_y': phi_y, 'main_support': support, 'views': cache,
            'details': details, 'full_p': full.get('p'), 'full_y': full.get('y'),
            'method': 'nonempty_context_marginal'}


def continuous_evidence(g, valid, eta=0.7):
    g = np.asarray(g, float).copy(); g[~valid] = 0
    total = float(g.sum())
    if total <= 0:
        return []
    order = np.argsort(-g); chosen = np.zeros_like(g, dtype=bool); mass = 0.0
    for i in order:
        if g[i] <= 0: break
        chosen[i] = True; mass += g[i] / total
        if mass >= eta: break
    for i in range(1, len(chosen) - 1):
        if chosen[i - 1] and chosen[i + 1]:
            chosen[i] = True
    spans, i = [], 0
    while i < len(chosen):
        if not chosen[i]:
            i += 1; continue
        j = i
        while j < len(chosen) and chosen[j]:
            j += 1
        spans.append((i, j)); i = j
    return spans


def verify_spans(model, batch, c_star, spans_by_mod, y0, p0):
    rows = []; base = (batch['O'] & batch['P'])[0].clone()
    for m, spans in enumerate(spans_by_mod):
        for lo, hi in spans:
            keep = base.clone(); keep[m, lo:hi] = False
            out = predict(model, masked(batch, keep))
            rows.append({'modality': NAMES[m], 'start': lo, 'end': hi,
                         'delta_p': p0 - float(out['p'][c_star]), 'delta_y': abs(y0 - out['y'])})
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


def _span_set(spans):
    s = set()
    for lo, hi in spans:
        s.update(range(lo, hi))
    return s


def stability(model, batch, eta, repeats=3, noise=0.01):
    g0, valid, _, _ = _grad_input(model, batch)
    spans0 = {m: continuous_evidence(g0[m], valid[m], eta) for m in range(3)}
    scores = []
    for _ in range(repeats):
        b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        for key in ('XA', 'XV'):
            m = 1 if key == 'XA' else 2
            b[key] = b[key] + noise * torch.randn_like(b[key]) * b['O'][:, m, :, None].float()
        g, v, _, _ = _grad_input(model, b)
        for m in range(3):
            a = _span_set(spans0[m]); bset = _span_set(continuous_evidence(g[m], v[m], eta))
            scores.append(len(a & bset) / max(1, len(a | bset)))
    return float(np.mean(scores)) if scores else 0.0


def explain_sample(model, batch, tokens, cfg):
    eta = float(cfg.get('evidence_eta', 0.7))
    g, valid, c_star, _ = _grad_input(model, batch)
    attr = modality_attribution(g, valid)
    marg = multi_context_marginal(model, batch, c_star)
    spans = [continuous_evidence(g[m], valid[m], eta) for m in range(3)]
    p0 = marg['full_p'] if marg['full_p'] is not None else float(predict(model, batch)['p'][c_star])
    y0 = marg['full_y'] if marg['full_y'] is not None else predict(model, batch)['y']
    checks = verify_spans(model, batch, c_star, spans, y0, p0)
    cs = comprehensiveness_sufficiency(model, batch, c_star, spans, p0)
    stab = stability(model, batch, eta, repeats=int(cfg.get('stability_repeats', 3)),
                     noise=float(cfg.get('stability_noise', 0.01)))
    key = {NAMES[m]: [{'start': lo, 'end': hi, 'tokens': tokens[lo:hi] if m == 0 else None,
                       'mass': float(g[m, lo:hi].sum())} for lo, hi in spans[m]] for m in range(3)}
    return {
        'class': c_star, 'polarity': ('负', '中', '正')[c_star], 'intensity': y0,
        'internal_attribution': {NAMES[m]: attr['I'][m] for m in range(3)},
        'main_support_modality': marg['main_support'],
        'most_sensitive_modality': NAMES[int(np.argmax(attr['I']))],
        'modality_marginal': marg, 'key_evidence': key, 'span_verification': checks,
        **cs, 'stability': stab,
    }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


@torch.no_grad()
def evaluate_split(model, train_cfg, device, split='test'):
    data = load_split(train_cfg['data_dir'], split)
    rows = []
    for lo in range(0, len(data['id']), 64):
        idx = torch.arange(lo, min(lo + 64, len(data['id'])))
        b = batch_fn(data, idx, device, dtype=next(model.parameters()).dtype)
        out = model(b); p = out['logits'].softmax(-1); c = p.argmax(-1); raw = out['intensity']
        y = project_intensity(c.cpu().numpy(), raw.cpu().numpy())
        for i, sid in enumerate(b['id']):
            rows.append({'id': sid, 'true_class': int(b['c'][i]), 'true_intensity': float(b['y'][i]),
                         'class': int(c[i]), 'intensity': float(y[i])})
    frame = pd.DataFrame(rows)
    yt, yp = frame['true_class'], frame['class']
    summary = {'accuracy': float(accuracy_score(yt, yp)),
               'macro_f1': float(f1_score(yt, yp, labels=[0, 1, 2], average='macro', zero_division=0)),
               'mae': float(np.abs(frame['true_intensity'] - frame['intensity']).mean())}
    return frame, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(MODULE_DIR / 'config_aligned.json'))
    p.add_argument('--device'); p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    cfg = json.loads(path(args.config).read_text(encoding='utf-8'))
    if args.device:
        cfg['device'] = args.device
    seed_all(cfg.get('seed', 2026))
    model, train, saved, digest = load_model(cfg)
    files = sorted(path(cfg['appendix4']).glob('*.pkl'))
    if args.smoke:
        files = files[:1]
    out = path(cfg['results']); out.mkdir(parents=True, exist_ok=True)
    tag = cfg.get('result_tag', '对齐版本')
    rows, cards = [], []
    for i, f in enumerate(files, 1):
        batch, text, tokens, sid = appendix_batch(f, train, cfg['device'])
        card = explain_sample(model, batch, tokens, cfg)
        card.update(sample_id=sid, source_file=f.name, raw_text=text)
        cards.append(card)
        rows.append({'样本编号': f.stem, '源文件': f.name, '原文': text,
                     '极性': card['polarity'],
                     '强度': round(float(card['intensity']), 6),
                     '文本归因': round(float(card['internal_attribution']['text']), 6),
                     '语音归因': round(float(card['internal_attribution']['audio']), 6),
                     '视觉归因': round(float(card['internal_attribution']['vision']), 6),
                     '主要参考模态': {'text': '文本', 'audio': '语音', 'vision': '视觉'}.get(
                         card['main_support_modality'], card['main_support_modality']),
                     '全面性': round(float(card['comprehensiveness']), 6),
                     '充分性': round(float(card['sufficiency']), 6),
                     '稳定性': round(float(card['stability']), 6)})
        print(f'[{i}/{len(files)}] {sid} {card["polarity"]} y={card["intensity"]:.3f}', flush=True)
    pd.DataFrame(rows).to_csv(out / f'{tag}_附件4_预测与解释结果.csv', index=False)
    _, summary = evaluate_split(model, train, cfg['device'], 'test')
    print(json.dumps({'done': True, 'n': len(cards), 'test_acc': round(summary['accuracy'], 4),
                      'checkpoint_sha256': digest, 'tag': tag}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
