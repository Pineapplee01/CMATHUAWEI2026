#!/usr/bin/env python3
# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：GPT-6 Astra
# 版本/型号：GPT-6 Astra（gpt-6-astra）
# 开发机构/公司：OpenAI
# 版本颁布日期：2026年9月3日
# 发布信息来源：https://openai.com/index/safety-overview-gpt-6-astra/

"""问题三未对齐版输入构造：附件4 pkl → 50/500/500 batch。"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer

from multimodal_emotion.explanation.q3_utils import path

SPECIAL = {0, 101, 102}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        return super().find_class(module, name)


def project_intensity(classes, raw, eps=1e-6):
    classes, raw = np.asarray(classes), np.asarray(raw, dtype=float)
    return np.where(classes == 0, np.clip(raw, -3, -eps),
                    np.where(classes == 1, 0., np.clip(raw, eps, 3)))


def _load_pkl(file):
    with Path(file).open('rb') as f:
        data = _Unpickler(f).load()
    if np.asarray(data['text_bert']).ndim == 2:
        data = {k: (np.expand_dims(v, 0) if not isinstance(v, str) else np.array([v])) for k, v in data.items()}
    return data


def appendix_batch(file, train_cfg, device):
    fields = _load_pkl(file)
    ids = np.asarray(fields['text_bert'])[0, 0].astype(np.int64)
    attn = np.asarray(fields['text_bert'])[0, 1].astype(bool)
    xa = np.asarray(fields['audio'], dtype=np.float64)
    xv = np.asarray(fields['vision'], dtype=np.float64)
    if xa.ndim == 2:
        xa, xv = xa[None], xv[None]
    la = int(np.asarray(fields.get('audio_lengths', [xa.shape[1]]))[0])
    lv = int(np.asarray(fields.get('vision_lengths', [xv.shape[1]]))[0])
    la, lv = max(1, min(la, 500)), max(1, min(lv, 500))
    p_t = attn & ~np.isin(ids, list(SPECIAL)); o_t = p_t.copy()
    p_a = np.zeros(500, dtype=bool); p_a[:la] = True
    p_v = np.zeros(500, dtype=bool); p_v[:lv] = True
    o_a = p_a & np.any(xa[0] != 0, axis=-1)
    o_v = p_v & np.any(xv[0] != 0, axis=-1)
    scaler = path(train_cfg['data_dir']) / 'scaler_params.npz'
    with np.load(scaler) as z:
        xa = (xa - z['mu_A']) / z['sigma_A']
        xv = (xv - z['mu_V']) / z['sigma_V']
    xa[0, ~o_a] = 0; xv[0, ~o_v] = 0
    batch = {
        'I': torch.from_numpy(ids[None].astype(np.int64)),
        'XA': torch.from_numpy(xa.astype(np.float32)),
        'XV': torch.from_numpy(xv.astype(np.float32)),
        'P_T': torch.from_numpy(p_t[None]), 'O_T': torch.from_numpy(o_t[None]),
        'P_A': torch.from_numpy(p_a[None]), 'O_A': torch.from_numpy(o_a[None]),
        'P_V': torch.from_numpy(p_v[None]), 'O_V': torch.from_numpy(o_v[None]),
    }
    batch = {k: v.to(device) for k, v in batch.items()}
    text = str(np.asarray(fields['raw_text']).reshape(-1)[0])
    tok = AutoTokenizer.from_pretrained(
        path('reference/models/bert-base-uncased'), local_files_only=True
    )
    tokens = tok.convert_ids_to_tokens(ids.tolist())
    sid = str(np.asarray(fields.get('id', [Path(file).stem])).reshape(-1)[0])
    return batch, text, tokens, sid


def masked(batch, modality=None, spans=None):
    b = {k: v.clone() for k, v in batch.items()}
    if modality == 'T':
        b['O_T'].zero_()
    elif modality == 'A':
        b['O_A'].zero_()
    elif modality == 'V':
        b['O_V'].zero_()
    if spans:
        for letter, segs in spans.items():
            for lo, hi in segs:
                b[f'O_{letter}'][:, lo:hi] = False
    return b
