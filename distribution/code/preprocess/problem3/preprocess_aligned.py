#!/usr/bin/env python3
# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：GPT-6 Astra
# 版本/型号：GPT-6 Astra（gpt-6-astra）
# 开发机构/公司：OpenAI
# 版本颁布日期：2026年9月3日
# 发布信息来源：https://openai.com/index/safety-overview-gpt-6-astra/

"""问题三对齐版输入构造：附件4 pkl → 模型 batch（与问题二对齐预处理口径一致）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from q3_utils import path
from q3_utils import load_fields

SPECIAL = {0, 101, 102}


def project_intensity(classes, raw, eps=1e-6):
    classes, raw = np.asarray(classes), np.asarray(raw, dtype=float)
    return np.where(classes == 0, np.clip(raw, -3, -eps),
                    np.where(classes == 1, 0., np.clip(raw, eps, 3)))


def appendix_batch(file, train_cfg, device):
    """读附件4对齐样本，建 P/O，音视用训练集 scaler 一次标准化；文本留给模型在线 BERT。"""
    fields = load_fields(file)
    ids = np.asarray(fields['text_bert'])[:, 0].astype(np.int64)
    attention = np.asarray(fields['text_bert'])[:, 1].astype(bool)
    oa = np.any(np.asarray(fields['audio']) != 0, -1)
    ov = np.any(np.asarray(fields['vision']) != 0, -1)
    support = attention | oa | ov
    extent = np.arange(50)[None] <= np.where(support, np.arange(50), -1).max(-1, keepdims=True)
    P = np.repeat(extent[:, None], 3, 1)
    P[:, 0] &= ~np.isin(ids, list(SPECIAL))
    cls = np.isin(ids, [101, 102]); P[:, 1] &= ~cls; P[:, 2] &= ~cls
    O = np.stack([attention & ~np.isin(ids, list(SPECIAL)), oa, ov], 1) & P
    out = {
        'P': torch.from_numpy(P), 'O': torch.from_numpy(O),
        'I': torch.from_numpy(ids), 'XT': torch.zeros(len(ids), 50, 768),
    }
    scaler = Path(train_cfg['data_dir']) / 'scaler_params.npz'
    with np.load(path(scaler) if not Path(scaler).is_absolute() else scaler) as z:
        for name, key, m in (('audio', 'A', 1), ('vision', 'V', 2)):
            x = (np.asarray(fields[name], np.float64) - z[f'mu_{key}']) / z[f'sigma_{key}']
            x[~O[:, m]] = 0
            out['X' + key] = torch.from_numpy(x.astype(np.float32))
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in out.items()}
    text = str(fields['raw_text'][0])
    tok = AutoTokenizer.from_pretrained(
        path('reference/models/bert-base-uncased'), local_files_only=True
    )
    tokens = tok.convert_ids_to_tokens(batch['I'][0].tolist())
    sid = str(fields.get('id', [Path(file).stem])[0])
    return batch, text, tokens, sid


def masked(batch, keep):
    b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    b['O'] = keep.bool() & batch['O'] & batch['P']
    return b
