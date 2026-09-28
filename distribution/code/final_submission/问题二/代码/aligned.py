#!/usr/bin/env python3
# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：GPT-6 Astra
# 版本/型号：GPT-6 Astra（gpt-6-astra）
# 开发机构/公司：OpenAI
# 版本颁布日期：2026年9月3日
# 发布信息来源：https://openai.com/index/safety-overview-gpt-6-astra/

"""问题二对齐版：槽位补全 + 样本级模态门控。用法：python aligned.py train|evaluate|predict"""
from __future__ import annotations
import argparse, json, sys
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
from q2_utils import load_fields
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


# ---------- 数据 ----------
def load_split(directory, split, dtype=np.float32):
    directory = path(directory)
    with np.load(directory / f'{split}.npz', allow_pickle=True) as z:
        f = {k: z[k] for k in z.files}
    ids = np.asarray(f['id']).astype(str).tolist()
    P, O = f['P'].astype(bool), f['O'].astype(bool)
    P[:, 0] &= ~np.isin(f['I'], [0, 101, 102])
    cls = np.isin(f['I'], [101, 102]); P[:, 1] &= ~cls; P[:, 2] &= ~cls
    O &= P
    n = len(ids)
    arrays = {k: np.ascontiguousarray(f[k], dtype=dtype) for k in ('XT', 'XA', 'XV')}
    for k, w in zip(arrays, (768, 74, 35)):
        if arrays[k].shape != (n, 50, w):
            raise ValueError(f'{split} {k} 形状错误')
    y = np.asarray(f['regression_labels'], dtype=dtype).reshape(-1)
    c = np.asarray(f['classification_labels'], dtype=np.int64).reshape(-1)
    return {**{k: torch.from_numpy(v) for k, v in arrays.items()},
            'P': torch.from_numpy(P), 'O': torch.from_numpy(O),
            'I': torch.from_numpy(f['I'].astype(np.int64)),
            'y': torch.from_numpy(y), 'c': torch.from_numpy(c), 'id': ids,
            'directory': str(directory), 'split': split}


def batch(data, indices, device, dtype=None):
    return {k: (v[indices].to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
                if torch.is_tensor(v) else [v[i] for i in indices.tolist()])
            for k, v in data.items() if torch.is_tensor(v) or k == 'id'}


def local_mask(data, rate, modalities, position, seed):
    keep = torch.ones_like(data['O']); records = []
    for i in range(len(keep)):
        for m in modalities:
            valid = torch.where(data['P'][i, m])[0].cpu().numpy()
            obs = data['O'][i, m].cpu().numpy(); before = int(obs.sum()); start = end = 0
            if len(valid) and rate and before >= 2:
                lo, hi = int(valid[0]), int(valid[-1]) + 1
                width = min(hi - lo, max(1, round(rate * (hi - lo))))
                if position == 'random':
                    rng = np.random.default_rng(np.random.SeedSequence([seed, i, m]))
                    start = int(rng.integers(lo, hi - width + 1))
                else:
                    start = {'start': lo, 'middle': lo + (hi - lo - width) // 2, 'end': hi - width}[position]
                end = start + width
                while end > start and obs[start:end].sum() == before:
                    end -= 1 if position != 'end' else 0; start += 1 if position == 'end' else 0
                keep[i, m, start:end] = False
            records.append({'id': data['id'][i], 'modality': 'TAV'[m],
                            'removed': int(obs[start:end].sum()), 'before': before})
    return keep, records


def masked_view(data, rate=.3, seed=4026):
    keep, records = local_mask(data, rate, (0, 1, 2), 'random', seed)
    return {**data, 'O': data['O'] & keep}, records


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


class AlignedCompletionBlock(nn.Module):
    """对齐版槽位补全块（对偶：未对齐 UnalignedModel.span_cross）。"""
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.cross = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d); self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))

    def forward(self, query, memory, padding):
        msg, _ = self.cross(query, memory, memory, key_padding_mask=padding, need_weights=False)
        h = self.norm1(query + msg)
        return self.norm2(h + self.ff(h))


class AlignedModel(nn.Module):
    """对齐完整模型：槽位补全交互 + 样本级模态门控（与 UnalignedModel 成对）。"""
    def __init__(self, cfg, pretrained=True):
        super().__init__(); self.cfg = cfg
        from transformers import AutoConfig, AutoModel
        bert_dir = path('AAAmodel/bert-base-uncased')
        self.bert = (AutoModel.from_pretrained(bert_dir, local_files_only=True) if pretrained
                     else AutoModel.from_config(AutoConfig.from_pretrained(bert_dir, local_files_only=True)))
        if getattr(self.bert, 'pooler', None) is not None:
            self.bert.pooler.requires_grad_(False)
        with np.load(path(cfg['data_dir']) / 'scaler_params.npz') as z:
            self.register_buffer('text_mu', torch.as_tensor(z['mu_T'], dtype=torch.float32))
            self.register_buffer('text_sigma', torch.as_tensor(z['sigma_T'], dtype=torch.float32))
        d = cfg.get('hidden', 96); heads = cfg.get('heads', 4)
        drop = float(cfg.get('dropout', .1))
        self.project = nn.ModuleList([nn.Sequential(nn.Linear(w, d), nn.GELU(), nn.Dropout(.2)) for w in (768, 74, 35)])
        self.position = nn.Parameter(torch.randn(50, d) * .02)
        self.modality = nn.Parameter(torch.randn(3, d) * .02)
        self.query = nn.Parameter(torch.randn(3, d) * .02)
        self.null = nn.Parameter(torch.zeros(1, 1, d))
        self.completion = nn.ModuleList([AlignedCompletionBlock(d, heads, drop) for _ in range(2)])
        self.mean = nn.ModuleList([nn.Linear(d, w) for w in (768, 74, 35)])
        self.logvar = nn.ModuleList([nn.Linear(d, w) for w in (768, 74, 35)])
        # 门控：α=softmax(wᵀ tanh(W h + b))，偏置在 gate_W 的仿射项中
        self.gate_W = nn.Linear(d, d); self.gate_w = nn.Linear(d, 1, bias=False)
        self.cls_head = nn.Linear(d, 3); self.reg_head = nn.Linear(d, 1)

    def initialize_shared(self, state):
        own = self.state_dict()
        keys = [k for k in own if k.startswith('project.') or k in ('position', 'modality')]
        for k in keys:
            if k not in state or own[k].shape != state[k].shape:
                raise ValueError(f'共享初始化不兼容: {k}')
        self.load_state_dict({k: state[k] for k in keys}, strict=False)
        return keys

    def encode_observed(self, batch):
        """投影后观测 E（问题三 Grad×Input 挂钩点；符号与文档 E_m 一致）。"""
        P = batch['P'].bool(); M = P & batch['O'].bool(); token = batch['I'].clone()
        attn = M[:, 0] | (token == 101) | (token == 102); token[~attn] = 0
        empty = ~attn.any(1); token[empty, 0] = 101; attn[empty, 0] = True
        xt = (self.bert(input_ids=token, attention_mask=attn.long()).last_hidden_state
              - self.text_mu) / self.text_sigma
        values = [xt * M[:, 0, :, None], batch['XA'] * M[:, 1, :, None], batch['XV'] * M[:, 2, :, None]]
        observed = torch.stack([layer(x) for layer, x in zip(self.project, values)], 1)
        return observed, P, M

    def complete(self, observed, P, M):
        """槽位补全 + 可靠性门控 r→q，得到槽表示 H。"""
        n, _, length, d = observed.shape
        offset = self.position[None, None] + self.modality[None, :, None]
        memory = torch.cat([(observed + offset).reshape(n, 3 * length, d), self.null.expand(n, -1, -1)], 1)
        padding = torch.cat([~M.flatten(1), M.flatten(1).any(1)[:, None]], 1)
        query = (offset + self.query[None, :, None]).expand(n, -1, -1, -1).reshape(n, 3 * length, d)
        for layer in self.completion:
            query = layer(query, memory, padding)
        query = query.reshape(n, 3, length, d)
        mu = [layer(query[:, m]) for m, layer in enumerate(self.mean)]
        completed = torch.stack([F.gelu(layer[0](x)) for layer, x in zip(self.project, mu)], 1)
        lv = [layer(query[:, m]).clamp(-6, 4) for m, layer in enumerate(self.logvar)]
        reliability = torch.stack([torch.exp(-x.exp().mean(-1)) for x in lv], 1)
        q = torch.where(M, torch.ones_like(reliability), reliability) * P
        h = observed * M[..., None] + completed * (P & ~M)[..., None] * q[..., None]
        return h, q, completed

    def fuse_heads(self, h, P):
        """槽均值 → 样本级门控 → logits / intensity。"""
        avail = P.any(-1)
        denom = P.sum(-1).to(h.dtype).clamp_min(1)
        hm = (h * P[..., None]).sum(2) / denom[..., None] * avail[..., None].to(h.dtype)
        z, alpha = modality_gate(hm, avail, self.gate_W, self.gate_w)
        return {'logits': self.cls_head(z), 'intensity': 3 * self.reg_head(z).squeeze(-1).tanh(),
                'modality_alpha': alpha, 'hm': hm, 'z': z}

    def forward(self, batch):
        observed, P, M = self.encode_observed(batch)
        h, q, _ = self.complete(observed, P, M)
        out = self.fuse_heads(h, P)
        return {k: out[k] for k in ('logits', 'intensity', 'modality_alpha')}


# ---------- 损失 / 指标 / 训练 ----------
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
    loss = (per * wt).sum() / wt.sum()
    loss = loss + .2 * F.huber_loss(y, batch['y'])
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
    p, r, f, _ = precision_recall_fscore_support(c, cp, labels=[0, 1, 2], zero_division=0)
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
    if saved.get('format') != 'aligned_v4':
        raise ValueError('需要对齐 aligned_v4 检查点')
    cfg = dict(saved['config'])
    if data_dir is not None:
        cfg['data_dir'] = data_dir
    cfg = resolve_data_config(cfg, '对齐版本')
    model = AlignedModel(cfg, pretrained=False).to(device)
    model.load_state_dict(saved['model'], strict=True); model.eval()
    return model, cfg


def train(cfg):
    device = cfg.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    cfg = {**cfg, 'device': device}
    train_data = load_split(cfg['data_dir'], 'train')
    valid = load_split(cfg['data_dir'], 'valid')
    counts, weights = class_weights(train_data['c'], cfg.get('balance_beta', .25), cfg.get('neutral_weight', 1.))
    cfg['train_class_counts'] = counts; cfg['class_weights'] = weights
    views = [masked_view(train_data, .3, 4026 + i)[0] for i in range(3)]
    valid_view, _ = masked_view(valid, .3, 4026)
    seed_all(cfg['seed'])
    model = AlignedModel(cfg).to(device)
    if cfg.get('init_checkpoint'):
        init = torch.load(path(cfg['init_checkpoint']), map_location='cpu', weights_only=False)
        cfg['initialized_shared_keys'] = model.initialize_shared(init['model'])
    opt = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if not n.startswith('bert.') and p.requires_grad],
         'lr': cfg.get('lr', 1e-4)},
        {'params': [p for p in model.bert.parameters() if p.requires_grad], 'lr': cfg.get('bert_lr', 1e-5)},
    ], weight_decay=.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs'], eta_min=0.)
    out = path(cfg['output']); out.mkdir(parents=True, exist_ok=True)
    ckpt = path(cfg['checkpoint']); ckpt.parent.mkdir(parents=True, exist_ok=True)
    accum = int(cfg.get('gradient_accumulation', 4)); best = -1e9; bad = 0; bw0 = float(cfg.get('boundary_weight', .1))
    warm = int(cfg.get('boundary_warmup_epochs', 3))
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
            torch.save({'format': 'aligned_v4', 'config': cfg, 'model': model.state_dict(),
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


def raw_to_inputs(fields, directory):
    ids = np.asarray(fields['text_bert'])[:, 0].astype(np.int64)
    attention = np.asarray(fields['text_bert'])[:, 1].astype(bool)
    oa = np.any(np.asarray(fields['audio']) != 0, -1)
    ov = np.any(np.asarray(fields['vision']) != 0, -1)
    support = attention | oa | ov
    extent = np.arange(50)[None] <= np.where(support, np.arange(50), -1).max(-1, keepdims=True)
    P = np.repeat(extent[:, None], 3, 1)
    P[:, 0] &= ~np.isin(ids, [0, 101, 102])
    cls = np.isin(ids, [101, 102]); P[:, 1] &= ~cls; P[:, 2] &= ~cls
    O = np.stack([attention & ~np.isin(ids, [0, 101, 102]), oa, ov], 1) & P
    out = {'P': torch.from_numpy(P), 'O': torch.from_numpy(O), 'I': torch.from_numpy(ids),
           'XT': torch.zeros(len(ids), 50, 768)}
    with np.load(path(directory) / 'scaler_params.npz') as z:
        for name, key, m in (('audio', 'A', 1), ('vision', 'V', 2)):
            x = (np.asarray(fields[name], np.float64) - z[f'mu_{key}']) / z[f'sigma_{key}']
            x[~O[:, m]] = 0
            out['X' + key] = torch.from_numpy(x.astype(np.float32))
    return out


@torch.no_grad()
def predict(cfg, device):
    model, trained = restore(cfg['checkpoint'], device, cfg.get('data_dir'))
    directory = trained['data_dir']
    root3 = path('AAAdata/Appendix_3/对齐版本')
    files = sorted(root3.glob('*.pkl'))
    if not files:
        raise FileNotFoundError(f'附件3对齐数据目录中没有 PKL 文件: {root3}')
    rows = []
    for file in files:
        fields = load_fields(file)
        inputs = raw_to_inputs(fields, directory)
        dtype = next(model.parameters()).dtype
        out = model({k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
                     for k, v in inputs.items()})
        c = int(out['logits'].argmax(-1)[0]); raw = float(out['intensity'][0])
        rows.append({'样本编号': str(fields.get('id', [file.stem])[0]), '源文件': file.name,
                     '极性': ('负', '中', '正')[c],
                     '强度': round(float(project_intensity(np.array([c]), np.array([raw]))[0]), 6)})
    outdir = path(cfg['output']); outdir.mkdir(parents=True, exist_ok=True)
    tag = cfg.get('result_tag', '对齐版本')
    target = outdir / f'{tag}_附件3_预测结果.csv'
    pd.DataFrame(rows).to_csv(target, index=False)
    print(f'附件3预测已保存: {target}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['train', 'evaluate', 'predict'])
    p.add_argument('--config', default=str(HERE / 'config_aligned.json'))
    p.add_argument('--device', default=None)
    p.add_argument('--split', choices=['valid', 'test'], default='valid')
    args = p.parse_args()
    cfg = resolve_data_config(json.loads(path(args.config).read_text()), '对齐版本')
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
