#!/usr/bin/env python3
# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：GPT-6 Astra
# 版本/型号：GPT-6 Astra（gpt-6-astra）
# 开发机构/公司：OpenAI
# 版本颁布日期：2026年9月3日
# 发布信息来源：https://openai.com/index/safety-overview-gpt-6-astra/

"""问题二未对齐版：连续窗软注意力 + 样本级模态门控。用法：python unaligned.py train|evaluate|predict"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SUBMISSION_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from q2_utils import resolve_data_config, path, json_write, seed_all
from q2_utils import NumpyCompatibleUnpickler
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


# ---------- 数据（50/500/500） ----------
def load_split(directory, split, dtype=np.float32):
    directory = path(directory)
    with np.load(directory / f'{split}.npz', allow_pickle=True) as z:
        f = {k: z[k] for k in z.files}
    ids = np.asarray(f['id']).astype(str).tolist(); n = len(ids); out = {}
    for m, key, length, width in zip('TAV', ('XT', 'XA', 'XV'), (50, 500, 500), (768, 74, 35)):
        x = np.ascontiguousarray(f[key], dtype=dtype)
        P = f[f'P_{m}'].astype(bool); O = f[f'O_{m}'].astype(bool)
        if x.shape != (n, length, width):
            raise ValueError(f'未对齐{m}形状错误')
        if m == 'T':
            P = P & ~np.isin(f['I'], [0, 101, 102]); O = O & P
        out[key] = torch.from_numpy(x)
        out[f'P_{m}'] = torch.from_numpy(P); out[f'O_{m}'] = torch.from_numpy(O)
    y = np.asarray(f['regression_labels'], dtype=dtype).reshape(-1)
    c = np.asarray(f['classification_labels'], dtype=np.int64).reshape(-1)
    return out | {'I': torch.from_numpy(f['I'].astype(np.int64)), 'y': torch.from_numpy(y),
                  'c': torch.from_numpy(c), 'id': ids, 'directory': str(directory), 'split': split,
                  'layout': 'unaligned_50_500_500'}


def batch(data, indices, device, dtype=None):
    return {k: (v[indices].to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
                if torch.is_tensor(v) else [v[i] for i in indices.tolist()])
            for k, v in data.items() if torch.is_tensor(v) or k == 'id'}


def local_mask_axis(P, O, rate, seed, mid, ids):
    """单轴连续缺失。P/O:(N,L)"""
    keep = torch.ones_like(O); records = []
    for i in range(len(keep)):
        valid = torch.where(P[i])[0].cpu().numpy()
        obs = O[i].cpu().numpy(); before = int(obs.sum()); start = end = 0
        if len(valid) and rate and before >= 2:
            lo, hi = int(valid[0]), int(valid[-1]) + 1
            width = min(hi - lo, max(1, round(rate * (hi - lo))))
            rng = np.random.default_rng(np.random.SeedSequence([seed, i, mid]))
            start = int(rng.integers(lo, hi - width + 1)); end = start + width
            while end > start and obs[start:end].sum() == before:
                end -= 1
            keep[i, start:end] = False
        records.append({'id': ids[i], 'modality': 'TAV'[mid], 'removed': int(obs[start:end].sum())})
    return keep, records


def masked_view(data, rate=.3, seed=4026):
    result = dict(data); records = []
    for m, letter in enumerate('TAV'):
        keep, part = local_mask_axis(data[f'P_{letter}'], data[f'O_{letter}'], rate, seed, m, data['id'])
        result[f'O_{letter}'] = data[f'O_{letter}'] & keep; records.extend(part)
    return result, records


def project_intensity(classes, raw, eps=1e-6):
    classes, raw = np.asarray(classes), np.asarray(raw, dtype=float)
    return np.where(classes == 0, np.clip(raw, -3, -eps),
                    np.where(classes == 1, 0., np.clip(raw, eps, 3)))


# ---------- 模型 ----------
def modality_gate(hm, avail, gate_W, gate_w):
    """对齐/未对齐共用样本级门控：α=softmax(wᵀ tanh(Wh))，z=Σ α_m h_m。"""
    scores = gate_w(torch.tanh(gate_W(hm))).squeeze(-1).masked_fill(~avail, -1e4)
    none = ~avail.any(-1)
    if none.any():
        scores = torch.where(none[:, None], scores.new_zeros(scores.shape), scores)
    alpha = scores.softmax(-1)
    return (alpha.unsqueeze(-1) * hm).sum(1), alpha


class UnalignedModel(nn.Module):
    """未对齐完整模型：连续窗软对齐 + 样本级模态门控（与 AlignedModel 成对）。

    内部 span 窗与窗内交叉注意对应对齐版的 AlignedCompletionBlock。
    """
    def __init__(self, cfg, pretrained=True):
        super().__init__(); self.cfg = cfg
        from transformers import AutoModel, AutoConfig
        src = path('AAAmodel/bert-base-uncased')
        self.bert = (AutoModel.from_pretrained(src, local_files_only=True) if pretrained
                     else AutoModel.from_config(AutoConfig.from_pretrained(src, local_files_only=True)))
        if getattr(self.bert, 'pooler', None) is not None:
            self.bert.pooler.requires_grad_(False)
        with np.load(path(cfg['data_dir']) / 'scaler_params.npz') as z:
            self.register_buffer('text_mu', torch.as_tensor(z['mu_T'], dtype=torch.float32))
            self.register_buffer('text_sigma', torch.as_tensor(z['sigma_T'], dtype=torch.float32))
        d = cfg.get('hidden', 96); self.heads = cfg.get('heads', 4); drop = cfg.get('dropout', .1)
        self.max_half_frac = float(cfg.get('align_max_half_frac', .2))
        self.min_half = float(cfg.get('align_min_half', 1.))
        self.time_penalty = float(cfg.get('align_time_penalty', 2.))
        self.project = nn.ModuleList([nn.Sequential(nn.Linear(w, d), nn.GELU(), nn.Dropout(drop)) for w in (768, 74, 35)])
        self.positions = nn.ParameterList([nn.Parameter(torch.randn(L, d) * .02) for L in (50, 500, 500)])
        self.query_norm = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.ModuleList([nn.Linear(d, d) for _ in range(2)])
        self.v_proj = nn.ModuleList([nn.Linear(d, d) for _ in range(2)])
        self.o_proj = nn.ModuleList([nn.Linear(d, d) for _ in range(2)])
        self.attn_drop = nn.Dropout(drop)
        self.span_step = nn.Linear(d, 1); self.span_half = nn.Linear(d, 1)
        self.null = nn.Parameter(torch.zeros(2, 1, d))
        self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])
        # 门控：α=softmax(wᵀ tanh(W h + b))，偏置在 gate_W 的仿射项中（与对齐版同构）
        self.gate_W = nn.Linear(d, d); self.gate_w = nn.Linear(d, 1, bias=False)
        self.cls_head = nn.Linear(d, 3); self.reg_head = nn.Linear(d, 1)

    def encode_observed(self, batch):
        """投影后观测 E_m（问题三 Grad×Input 挂钩点；与对齐版符号一致）。

        返回 list[E_T,E_A,E_V]、P 列表、M 列表（长度分别为 50/500/500）。
        """
        P = [batch[f'P_{m}'].bool() for m in 'TAV']
        M = [p & batch[f'O_{m}'].bool() for p, m in zip(P, 'TAV')]
        token = batch['I'].clone(); attn = M[0] | (token == 101) | (token == 102)
        token[~attn] = 0; empty = ~attn.any(1); token[empty, 0] = 101; attn[empty, 0] = True
        raw = self.bert(input_ids=token, attention_mask=attn.long()).last_hidden_state
        values = [(raw - self.text_mu) / self.text_sigma, batch['XA'], batch['XV']]
        observed = [layer(x * m[..., None]) * m[..., None]
                    for layer, x, m in zip(self.project, values, M)]
        return observed, P, M

    def span_windows(self, query, query_mask, length):
        """未对齐连续单调局部窗（对应对齐版补全块的时序组织）。"""
        step = (F.softplus(self.span_step(query).squeeze(-1)) + 1e-3) * query_mask.to(query.dtype)
        empty = ~query_mask.any(-1)
        if empty.any():
            step = torch.where(empty[:, None], query_mask.new_ones(query_mask.shape, dtype=step.dtype) / 50, step)
        total = step.sum(-1, keepdim=True).clamp_min(1e-6)
        ends = torch.cumsum(step, -1) / total; starts = ends - step / total
        span = length.to(step.dtype).clamp_min(1)[:, None] - 1
        centers = .5 * (starts + ends) * span
        half = (torch.sigmoid(self.span_half(query).squeeze(-1)) * self.max_half_frac
                * length.to(step.dtype)[:, None]).clamp(min=self.min_half)
        lo = torch.cummax(centers - half, -1).values
        hi = torch.cummax(centers + half, -1).values
        hi = torch.maximum(hi, lo + self.min_half)
        L = length.to(lo.dtype)[:, None].clamp_min(1)
        lo = torch.minimum(lo.clamp_min(0), L - 1e-3)
        hi = torch.minimum(torch.maximum(hi, lo + self.min_half), L)
        return centers, lo, torch.maximum(hi, lo + self.min_half)

    def span_cross(self, query, memory, observed, query_mask, lo, hi, centers, length, null, j):
        """窗内软交叉注意（对应对齐版 AlignedCompletionBlock 的跨模态消息）。"""
        n, t, d = query.shape; h = self.heads; dh = d // h
        q = self.q_proj(query).view(n, t, h, dh).transpose(1, 2)
        k = self.k_proj[j](memory).view(n, 500, h, dh).transpose(1, 2)
        v = self.v_proj[j](memory).view(n, 500, h, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        idx = torch.arange(500, device=query.device, dtype=lo.dtype)[None, None]
        inside = (idx >= lo[:, :, None]) & (idx <= hi[:, :, None]) & observed[:, None]
        dist = ((idx - centers[:, :, None]).abs() / length.to(lo.dtype).clamp_min(1)[:, None, None]).clamp(0, 1)
        scores = scores - self.time_penalty * dist[:, None]
        null_k = self.k_proj[j](null).view(n, 1, h, dh).transpose(1, 2)
        null_score = (q * null_k).sum(-1, keepdim=True) / math.sqrt(dh)
        empty = ~observed.any(-1)
        scores = scores.masked_fill(~inside[:, None], -1e4)
        scores = torch.cat([scores, null_score], -1)
        none = ~inside.any(-1)
        scores[..., -1] = torch.where(none[:, None] | empty[:, None, None],
                                      scores.new_zeros(scores[..., -1].shape), scores[..., -1])
        weights = self.attn_drop(scores.softmax(-1))
        null_v = self.v_proj[j](null).view(n, 1, h, dh).transpose(1, 2)
        msg = (weights @ torch.cat([v, null_v], 2)).transpose(1, 2).contiguous().view(n, t, d)
        return self.o_proj[j](msg)

    def fuse_from_observed(self, observed, P, M, return_attention=False):
        """软对齐 → 词轴均值 hm → 样本级门控（与对齐版 fuse_heads 同构出口）。"""
        query = self.query_norm(observed[0] + self.positions[0])
        qmask = P[0].clone(); fallback = ~qmask.any(1)
        qmask[fallback] = (M[1].any(1) | M[2].any(1))[fallback, None].expand(-1, 50)
        aligned = []; maps = {}; n = observed[0].shape[0]
        for j in range(2):
            length = P[j + 1].long().sum(-1).clamp_min(1)
            centers, lo, hi = self.span_windows(query, qmask, length)
            msg = self.span_cross(query, observed[j + 1] + self.positions[j + 1], M[j + 1], qmask,
                                  lo, hi, centers, length, self.null[j:j + 1].expand(n, -1, -1), j)
            aligned.append(self.norm[j](msg) * M[j + 1].any(1)[:, None, None])
            if return_attention:
                with torch.no_grad():
                    n_, t, d = query.shape; heads = self.heads; dh = d // heads
                    q = self.q_proj(query).view(n_, t, heads, dh).transpose(1, 2)
                    memory = observed[j + 1] + self.positions[j + 1]
                    k = self.k_proj[j](memory).view(n_, 500, heads, dh).transpose(1, 2)
                    scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
                    idx = torch.arange(500, device=query.device, dtype=lo.dtype)[None, None]
                    inside = (idx >= lo[:, :, None]) & (idx <= hi[:, :, None]) & M[j + 1][:, None]
                    dist = ((idx - centers[:, :, None]).abs()
                            / length.to(lo.dtype).clamp_min(1)[:, None, None]).clamp(0, 1)
                    scores = scores - self.time_penalty * dist[:, None]
                    scores = scores.masked_fill(~inside[:, None], -1e4)
                    weights = scores.softmax(-1).mean(1)
                    name = 'audio' if j == 0 else 'vision'
                    maps[f'attention_{name}'] = weights * qmask[:, :, None]
        denom = qmask.sum(-1, keepdim=True).to(observed[0].dtype).clamp_min(1)
        hm = torch.stack([(observed[0] * qmask[..., None]).sum(1) / denom,
                          (aligned[0] * qmask[..., None]).sum(1) / denom,
                          (aligned[1] * qmask[..., None]).sum(1) / denom], 1)
        avail = torch.stack([qmask.any(-1), M[1].any(-1), M[2].any(-1)], -1)
        hm = hm * avail[..., None].to(observed[0].dtype)
        z, alpha = modality_gate(hm, avail, self.gate_W, self.gate_w)
        out = {'logits': self.cls_head(z), 'intensity': 3 * self.reg_head(z).squeeze(-1).tanh(),
               'modality_alpha': alpha, 'hm': hm, 'z': z}
        if return_attention:
            out.update(maps)
        return out

    def forward(self, batch, return_attention=False):
        observed, P, M = self.encode_observed(batch)
        out = self.fuse_from_observed(observed, P, M, return_attention=return_attention)
        keys = ['logits', 'intensity', 'modality_alpha']
        if return_attention:
            keys += [k for k in out if k.startswith('attention_')]
        return {k: out[k] for k in keys}


# ---------- 损失 / 训练 / 评测（与对齐版同口径） ----------
def class_weights(labels, beta=0.25, neutral=1.0):
    counts = torch.bincount(torch.as_tensor(labels).long(), minlength=3).tolist()
    total = sum(counts)
    w = [(total / (3 * n)) ** beta for n in counts]; w[1] *= neutral
    s = sum(n * x for n, x in zip(counts, w)) / total
    return counts, [x / s for x in w]


def objective(out, batch, weights, boundary_weight=0., margin=.5, max_i=.5):
    logits, y = out['logits'], out['intensity']
    per = F.cross_entropy(logits, batch['c'], label_smoothing=.03, reduction='none')
    wt = logits.new_tensor(weights)[batch['c']]
    loss = (per * wt).sum() / wt.sum() + .2 * F.huber_loss(y, batch['y'])
    target = F.one_hot(batch['c'], 3).to(logits.dtype).cumsum(-1)[:, :2]
    loss = loss + .1 * (logits.softmax(-1).cumsum(-1)[:, :2] - target).square().mean()
    if boundary_weight:
        score = logits[:, 1] - torch.logsumexp(logits[:, [0, 2]], -1)
        neu = score[batch['c'] == 1]
        weak = score[(batch['c'] != 1) & (batch['y'].abs() > 0) & (batch['y'].abs() <= max_i)]
        if neu.numel() and weak.numel():
            loss = loss + boundary_weight * F.softplus(margin - neu[:, None] + weak[None]).mean()
    return loss


def metrics(frame):
    c, cp = frame['true_class'].to_numpy(), frame['class'].to_numpy()
    y, pred = frame['true_intensity'].to_numpy(), frame['intensity'].to_numpy()
    _, _, f, _ = precision_recall_fscore_support(c, cp, labels=[0, 1, 2], zero_division=0)
    return {'accuracy': float(accuracy_score(c, cp)),
            'macro_f1': float(f1_score(c, cp, labels=[0, 1, 2], average='macro', zero_division=0)),
            'neutral_f1': float(f[1]), 'mae': float(np.abs(y - pred).mean())}


@torch.no_grad()
def evaluate(model, data, device, batch_size=128):
    model.eval(); rows = []
    for lo in range(0, len(data['id']), batch_size):
        idx = torch.arange(lo, min(lo + batch_size, len(data['id'])))
        b = batch(data, idx, device, dtype=next(model.parameters()).dtype)
        out = model(b); c = out['logits'].argmax(-1); raw = out['intensity']
        inten = project_intensity(c.cpu().numpy(), raw.cpu().numpy())
        for i, sid in enumerate(b['id']):
            rows.append({'id': sid, 'true_class': int(b['c'][i]), 'true_intensity': float(b['y'][i]),
                         'class': int(c[i]), 'intensity': float(inten[i])})
    frame = pd.DataFrame(rows)
    return metrics(frame), frame


def restore(ckpt, device, data_dir=None):
    saved = torch.load(path(ckpt), map_location='cpu', weights_only=False)
    if saved.get('format') != 'unaligned_v1':
        raise ValueError('需要未对齐 unaligned_v1 检查点')
    cfg = dict(saved['config'])
    if data_dir is not None:
        cfg['data_dir'] = data_dir
    cfg = resolve_data_config(cfg, '未对齐版本')
    model = UnalignedModel(cfg, pretrained=False).to(device)
    model.load_state_dict(saved['model'], strict=True); model.eval()
    return model, cfg


def train(cfg):
    device = cfg.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    cfg = {**cfg, 'device': device, 'architecture': 'soft_alignment'}
    train_data = load_split(cfg['data_dir'], 'train')
    valid = load_split(cfg['data_dir'], 'valid')
    counts, weights = class_weights(train_data['c'], cfg.get('balance_beta', .25), cfg.get('neutral_weight', 1.))
    cfg['train_class_counts'] = counts; cfg['class_weights'] = weights
    views = [masked_view(train_data, .3, 4026 + i)[0] for i in range(3)]
    valid_view, _ = masked_view(valid, .3, 4026)
    seed_all(cfg['seed'])
    model = UnalignedModel(cfg).to(device)
    opt = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if not n.startswith('bert.') and p.requires_grad],
         'lr': cfg.get('lr', 1e-4)},
        {'params': [p for p in model.bert.parameters() if p.requires_grad], 'lr': cfg.get('bert_lr', 1e-5)},
    ], weight_decay=.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs'], eta_min=0.)
    out = path(cfg['output']); out.mkdir(parents=True, exist_ok=True)
    ckpt = path(cfg['checkpoint']); ckpt.parent.mkdir(parents=True, exist_ok=True)
    accum = int(cfg.get('gradient_accumulation', 4)); best = -1e9; bad = 0
    bw0 = float(cfg.get('boundary_weight', .1)); warm = int(cfg.get('boundary_warmup_epochs', 3))
    for epoch in range(cfg['epochs']):
        bw = bw0 * min(1., (epoch + 1) / warm)
        model.train(); order = torch.randperm(len(train_data['id'])); total = 0
        for step, lo in enumerate(range(0, len(order), cfg['batch_size'])):
            idx = order[lo:lo + cfg['batch_size']]
            group = min(accum * cfg['batch_size'], len(order) - (step // accum) * accum * cfg['batch_size'])
            full = batch(train_data, idx, device); part = batch(views[(epoch + step) % len(views)], idx, device)
            if step % accum == 0:
                opt.zero_grad(set_to_none=True)
            loss = objective(model(full), full, weights, bw, cfg.get('boundary_margin', .5),
                             cfg.get('boundary_max_intensity', .5))
            loss = loss + .4 * min(1, (epoch + 1) / 5) * objective(model(part), part, weights)
            (loss * len(idx) / group).backward()
            if (step + 1) % accum == 0 or lo + len(idx) == len(order):
                nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
            total += float(loss.detach()) * len(idx)
        sched.step()
        full_m, _ = evaluate(model, valid, device); loc_m, _ = evaluate(model, valid_view, device)
        score = .7 * full_m['accuracy'] + .3 * loc_m['accuracy']
        print(json.dumps({'epoch': epoch + 1, 'acc': round(full_m['accuracy'], 4),
                          'local30': round(loc_m['accuracy'], 4), 'best': round(max(best, score), 4)},
                         ensure_ascii=False), flush=True)
        if score > best + 1e-8:
            best = score; bad = 0
            torch.save({'format': 'unaligned_v1', 'config': cfg, 'model': model.state_dict(),
                        'best_epoch': epoch + 1}, ckpt)
        else:
            bad += 1
            if bad >= cfg['patience']:
                break
    saved = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(saved['model'])
    full_m, cf = evaluate(model, valid, device); loc_m, mf = evaluate(model, valid_view, device)
    cf.to_csv(out / 'valid_full.csv', index=False); mf.to_csv(out / 'valid_local30.csv', index=False)
    json_write(out / 'summary.json', {'complete': full_m, 'local30': loc_m, 'best': best, 'checkpoint': str(ckpt)})
    return full_m


def raw_to_inputs(fields, directory, tokenizer=None):
    """附件3未对齐：原生长度轴；无 text_bert 时用 raw_text 在线分词。"""
    if 'text_bert' in fields:
        tb = np.asarray(fields['text_bert'])
        if tb.ndim == 2:
            tb = tb[None]
        ids = tb[0, 0].astype(np.int64); attn = tb[0, 1].astype(bool)
    else:
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(path('AAAmodel/bert-base-uncased'), local_files_only=True)
        text = str(np.asarray(fields['raw_text']).reshape(-1)[0])
        enc = tokenizer(text, padding='max_length', truncation=True, max_length=50, return_tensors='np')
        ids = enc['input_ids'][0].astype(np.int64); attn = enc['attention_mask'][0].astype(bool)
    audio = np.asarray(fields['audio'], np.float64); vision = np.asarray(fields['vision'], np.float64)
    if audio.ndim == 2:
        audio, vision = audio[None], vision[None]
    al = int(np.asarray(fields.get('audio_lengths', [min(500, audio.shape[1])]))[0])
    vl = int(np.asarray(fields.get('vision_lengths', [min(500, vision.shape[1])]))[0])
    al, vl = max(1, min(al, 500)), max(1, min(vl, 500))
    if audio.shape[1] < 500:
        pad = np.zeros((audio.shape[0], 500 - audio.shape[1], audio.shape[2]), audio.dtype)
        audio = np.concatenate([audio, pad], 1)
    if vision.shape[1] < 500:
        pad = np.zeros((vision.shape[0], 500 - vision.shape[1], vision.shape[2]), vision.dtype)
        vision = np.concatenate([vision, pad], 1)
    audio, vision = audio[:, :500], vision[:, :500]
    P_T = attn & ~np.isin(ids, [0, 101, 102]); O_T = P_T.copy()
    P_A = np.arange(500) < al; P_V = np.arange(500) < vl
    O_A = np.any(audio[0] != 0, -1) & P_A; O_V = np.any(vision[0] != 0, -1) & P_V
    out = {'I': torch.from_numpy(ids[None]), 'P_T': torch.from_numpy(P_T[None]), 'O_T': torch.from_numpy(O_T[None]),
           'P_A': torch.from_numpy(P_A[None]), 'O_A': torch.from_numpy(O_A[None]),
           'P_V': torch.from_numpy(P_V[None]), 'O_V': torch.from_numpy(O_V[None]),
           'XT': torch.zeros(1, 50, 768)}
    with np.load(path(directory) / 'scaler_params.npz') as z:
        for arr, key, Om in ((audio, 'A', O_A), (vision, 'V', O_V)):
            x = (arr - z[f'mu_{key}']) / z[f'sigma_{key}']; x[:, ~Om] = 0
            out['X' + key] = torch.from_numpy(x.astype(np.float32))
    return out


@torch.no_grad()
def predict(cfg, device):
    model, trained = restore(cfg['checkpoint'], device, cfg.get('data_dir'))
    directory = trained['data_dir']
    root3 = path('AAAdata/Appendix_3/未对齐版本')
    files = sorted(root3.glob('*.pkl'))
    if not files:
        raise FileNotFoundError(f'附件3未对齐数据目录中没有 PKL 文件: {root3}')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path('AAAmodel/bert-base-uncased'), local_files_only=True)
    rows = []
    for file in files:
        with Path(file).open('rb') as f:
            blob = NumpyCompatibleUnpickler(f).load()
        fields = blob['test'] if isinstance(blob, dict) and 'test' in blob else blob
        inputs = raw_to_inputs(fields, directory, tokenizer=tokenizer)
        dtype = next(model.parameters()).dtype
        out = model({k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
                     for k, v in inputs.items()})
        c = int(out['logits'].argmax(-1)[0]); raw = float(out['intensity'][0])
        sid = str(np.asarray(fields.get('id', [file.stem])).reshape(-1)[0])
        rows.append({'样本编号': sid, '源文件': file.name,
                     '极性': ('负', '中', '正')[c],
                     '强度': round(float(project_intensity(np.array([c]), np.array([raw]))[0]), 6)})
    outdir = path(cfg['output']); outdir.mkdir(parents=True, exist_ok=True)
    tag = cfg.get('result_tag', '未对齐版本')
    target = outdir / f'{tag}_附件3_预测结果.csv'
    pd.DataFrame(rows).to_csv(target, index=False)
    print(f'附件3预测已保存: {target}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['train', 'evaluate', 'predict'])
    p.add_argument('--config', default=str(HERE / 'config_unaligned.json'))
    p.add_argument('--device', default=None)
    p.add_argument('--split', choices=['valid', 'test'], default='valid')
    args = p.parse_args()
    cfg = resolve_data_config(json.loads(path(args.config).read_text()), '未对齐版本')
    device = args.device or cfg.get('device', 'cuda:0')
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    if args.action == 'train':
        cfg['device'] = device; train(cfg); return
    if args.action == 'predict':
        predict(cfg, device); return
    model, trained = restore(cfg['checkpoint'], device, cfg.get('data_dir'))
    data = load_split(trained['data_dir'], args.split)
    full, frame = evaluate(model, data, device)
    view, _ = masked_view(data)
    local, _ = evaluate(model, view, device)
    print(json.dumps({'full': full, 'local30': local}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
