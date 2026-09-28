#!/usr/bin/env python3
"""补题目(5)：附件2验证集基础性能 + 混淆/错类图 + 错例解释归因。

用法（仓库根目录）:
  CUDA_VISIBLE_DEVICES=0 python -m problem3.supplement_section5 --variant aligned --device cuda:0
  CUDA_VISIBLE_DEVICES=0 python -m problem3.supplement_section5 --variant unaligned --device cuda:0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from shared.common import path, seed_all, json_write
from problem2_retrain_v2.data import batch, load_split
from problem3.report import evaluate_split, plot_split_summary, _jsonable

LABELS = ('负', '中', '正')
# 优先覆盖的混淆方向（真→预）
PRIORITY = [(1, 2), (1, 0), (0, 2), (2, 0), (0, 1), (2, 1)]


def _cfg(variant: str) -> dict:
    if variant == 'aligned':
        return {
            'variant': 'aligned',
            'seed': 2027,
            'checkpoint': 'AAAmodel/checkpoints/selected/problem3/aligned_gate_B0_primary.pt',
            'results': 'problem3/results',
            'analysis_md': 'problem3/问题三结果分析_对齐版.md',
            'evidence_eta': 0.7,
            'stability_repeats': 3,
            'stability_noise': 0.01,
        }
    return {
        'variant': 'unaligned',
        'seed': 2031,
        'checkpoint': 'AAAmodel/checkpoints/selected/problem3/unaligned_gate_B0_primary.pt',
        'results': 'problem3_unaligned/results',
        'analysis_md': 'problem3_unaligned/问题三结果分析_未对齐版.md',
        'evidence_eta': 0.7,
        'associate_topk': 5,
        'stability_repeats': 3,
        'stability_noise': 0.01,
    }


def _load(variant: str, cfg: dict):
    if variant == 'aligned':
        from problem3.model_io import load_model
        from problem3.explain import explain_sample
        model, train, saved, tok, digest = load_model(cfg)
        return model, train, saved, tok, digest, explain_sample

    # 未对齐正式检查点为门控版；problem3_unaligned 旧解释仍绑 fuse 头，改走提交版实现
    import importlib.util
    aaa_py = path('AAA提交版代码及结果/问题三/代码/unaligned.py')
    spec = importlib.util.spec_from_file_location('aaa_q3_unaligned', aaa_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model, train, saved, digest = mod.load_model(cfg)
    tok = AutoTokenizer.from_pretrained(path('AAAmodel/bert-base-uncased'), local_files_only=True)
    return model, train, saved, tok, digest, mod.explain_sample


def _pick_errors(frame: pd.DataFrame, k_per=1, max_total=6):
    err = frame[frame.true_class != frame['class']].copy()
    err['conf'] = err.apply(lambda r: float(r[f'p_{int(r["class"])}']), axis=1)
    picked = []
    used = set()
    for t, p in PRIORITY:
        sub = err[(err.true_class == t) & (err['class'] == p)].sort_values('conf', ascending=False)
        for _, row in sub.head(k_per).iterrows():
            sid = row['id']
            if sid in used:
                continue
            used.add(sid)
            picked.append(row)
            if len(picked) >= max_total:
                return picked
    # 不足则按置信度补
    for _, row in err.sort_values('conf', ascending=False).iterrows():
        if row['id'] in used:
            continue
        used.add(row['id'])
        picked.append(row)
        if len(picked) >= max_total:
            break
    return picked


def _sample_batch(data, idx, device, dtype, tok, variant):
    b = batch(data, torch.tensor([idx]), device, dtype=dtype)
    # 解释链路只接受张量字段（id 等 list 会破坏 clone）
    b = {k: v for k, v in b.items() if torch.is_tensor(v)}
    ids = b['I'][0].tolist() if 'I' in b else []
    tokens = tok.convert_ids_to_tokens(ids) if ids else []
    return b, tokens


def _write_section(md_path: Path, section: str):
    text = md_path.read_text(encoding='utf-8') if md_path.is_file() else ''
    marker = '## 5. 验证集基础性能'
    if marker in text:
        pre = text.split(marker)[0].rstrip() + '\n\n'
        text = pre + section.lstrip()
    else:
        text = text.rstrip() + '\n\n' + section.lstrip()
    md_path.write_text(text, encoding='utf-8')


def run(variant: str, device: str):
    cfg = _cfg(variant)
    cfg['device'] = device
    seed_all(cfg['seed'])
    out = path(cfg['results'])
    out.mkdir(parents=True, exist_ok=True)
    fig = out / 'figures'
    fig.mkdir(exist_ok=True)

    model, train, saved, tok, digest, explain_sample = _load(variant, cfg)
    dtype = next(model.parameters()).dtype

    print(f'[{variant}] 评价验证集…', flush=True)
    frame, summary = evaluate_split(model, train, saved, device, split='valid')
    frame.to_csv(out / '验证集预测.csv', index=False)
    err_info = plot_split_summary(frame, summary, fig, split='valid')
    err_info['n_valid'] = len(frame)
    payload = {**summary, **err_info, 'checkpoint_sha256': digest}
    json_write(out / '验证集指标.json', _jsonable(payload))

    print(f'[{variant}] 评价测试集…', flush=True)
    test_frame, test_summary = evaluate_split(model, train, saved, device, split='test')
    test_frame.to_csv(out / '测试集预测.csv', index=False)
    test_err = plot_split_summary(test_frame, test_summary, fig, split='test')
    test_err['n_test'] = len(test_frame)
    json_write(out / '测试集指标.json', _jsonable({**test_summary, **test_err, 'checkpoint_sha256': digest}))
    tyt, typ = test_frame['true_class'].to_numpy(), test_frame['class'].to_numpy()
    t_neu_rec = float(((typ == 1) & (tyt == 1)).sum() / max(1, (tyt == 1).sum()))
    t_neu_prec = float(((typ == 1) & (tyt == 1)).sum() / max(1, (typ == 1).sum()))
    t_cm = np.array(test_err['confusion'])
    t_rates = test_err['error_rate_by_true_class']

    # 错例 + 解释：以测试集为主
    data = load_split(train['data_dir'], 'test', dtype=getattr(np, train.get('precision', 'float32')))
    id_to_i = {str(data['id'][i]): i for i in range(len(data['id']))}
    picked = _pick_errors(test_frame)
    cases = []
    print(f'[{variant}] 解释 {len(picked)} 条测试集错例…', flush=True)
    for row in picked:
        sid = str(row['id'])
        i = id_to_i[sid]
        b, tokens = _sample_batch(data, i, device, dtype, tok, variant)
        card = explain_sample(model, b, tokens, cfg)
        true_c, pred_c = int(row['true_class']), int(row['class'])
        cases.append({
            'id': sid,
            'true': LABELS[true_c],
            'pred': LABELS[pred_c],
            'true_intensity': float(row['true_intensity']),
            'pred_intensity': float(row['intensity']),
            'conf': float(row[f'p_{pred_c}']),
            'main_support': card.get('main_support_modality'),
            'I': card.get('internal_attribution'),
            'comp': card.get('comprehensiveness'),
            'suff': card.get('sufficiency'),
            'stab': card.get('stability'),
            'key_text': (card.get('key_evidence') or {}).get('text'),
        })
        print(f'  {sid}: {LABELS[true_c]}→{LABELS[pred_c]} support={card.get("main_support_modality")} '
              f'Comp={card.get("comprehensiveness")}', flush=True)

    json_write(out / '测试集错例解释.json', _jsonable({
        'checkpoint_sha256': digest, 'n': len(cases), 'cases': cases,
    }))

    case_rows = []
    for c in cases:
        kt = c.get('key_text')
        words = ''
        if isinstance(kt, list) and kt:
            toks = []
            for span in kt[:2]:
                if isinstance(span, dict) and span.get('tokens'):
                    toks.extend(span['tokens'])
            words = ' '.join(toks)[:48]
        elif isinstance(kt, dict):
            words = ' '.join(kt.get('tokens') or [])[:48]
        I = c.get('I') or {}
        case_rows.append(
            f"| {c['id']} | {c['true']}→{c['pred']} | {c['conf']:.2f} | "
            f"{CN(c.get('main_support'))} | "
            f"{float((I or {}).get('text', 0)):.2f}/"
            f"{float((I or {}).get('audio', 0)):.2f}/"
            f"{float((I or {}).get('vision', 0)):.2f} | "
            f"{_fmt(c.get('comp'))} / {_fmt(c.get('suff'))} / {_fmt(c.get('stab'))} | {words} |"
        )

    section = f"""## 5. 测试集上的基础性能评价、可视化分析与错误归因结论

> 本节给出附件2 **测试集**（`test`，{len(test_frame)} 条，有金标签）上的基础性能、可视化与错误归因。  
> **不是**附件4（附件4无公开金标，只做预测与解释汇总）。  
> 检查点：`{cfg['checkpoint']}`（sha256 前 12 位 `{digest[:12]}`）。解释层零训练，不改变冻结前向预测。  
> 图题写在正文；`figures/test_*.pdf` 图内不另加标题。

### 5.1 基础性能评价

路径：`{test_summary['split_path']}`。

| 指标 | 数值 |
|------|------|
| Accuracy | **{test_summary['accuracy']*100:.2f}%** |
| Macro-F1 | {test_summary['macro_f1']*100:.2f}% |
| MAE | {test_summary['mae']:.4f} |
| Pearson | {test_summary['pearson']:.4f} |
| 中性 Precision / Recall | {t_neu_prec*100:.1f}% / {t_neu_rec*100:.1f}% |
| 错误条数 | {test_err['n_error']} / {len(test_frame)} |

按真实类错误率：负 {t_rates['负']*100:.1f}%、中 **{t_rates['中']*100:.1f}%**、正 {t_rates['正']*100:.1f}%。中性类错误率最高。

（对照：同检查点验证集 ACC={summary['accuracy']*100:.2f}%，错误 {err_info['n_error']}/{len(frame)}；选 epoch 仅用验证集。）

### 5.2 可视化分析

- 图　测试集混淆矩阵：`figures/test_confusion.pdf`（行=真实，列=预测；类别序负/中/正）
- 图　测试集按真实类错误率：`figures/test_error_by_class.pdf`

混淆计数：

|  | 预负 | 预中 | 预正 |
|--|-----:|-----:|-----:|
| 真负 | {t_cm[0,0]} | {t_cm[0,1]} | {t_cm[0,2]} |
| 真中 | {t_cm[1,0]} | {t_cm[1,1]} | {t_cm[1,2]} |
| 真正 | {t_cm[2,0]} | {t_cm[2,1]} | {t_cm[2,2]} |

### 5.3 错误归因结论

从测试集错误样本中按「中→正 / 中→负 / 负→正 / 正→负 …」优先抽取高置信错例，并在**同一冻结模型**上跑 Grad×Input 解释（不改预测）。

| 样本 | 真→预 | 预置信度 | 主参考 | $I_T/I_A/I_V$ | Comp/Suff/Stab | 文本关键窗词面（节选） |
|------|-------|----------|--------|-----------------|----------------|------------------------|
{chr(10).join(case_rows)}

**结论**

1. **错因主体在分类边界，不在解释层**：问题三不更新参数，测试集预测与问题二冻结前向一致；解释只说明「为何得到当前预测」，不能把错例改对。  
2. **中性最易错**：真实中性错误率最高；错例主参考仍常落在文本，属弱极性/中性边界判偏。  
3. **解释与错误同向**：高置信错例的 Comp 往往仍为正，关键窗非空——证据支持的是**错误类别**。  
4. 与附件4区分：附件4用于交付全量预测与解释卡；本节用附件2测试集金标做对错评价与归因。
"""
    md_path = path(cfg['analysis_md'])
    text = md_path.read_text(encoding='utf-8') if md_path.is_file() else ''
    for marker in (
        '## 5. 测试集上的基础性能评价',
        '## 5. 有标签划分上的基础性能',
        '## 5. 验证集基础性能',
    ):
        if marker in text:
            text = text.split(marker)[0].rstrip() + '\n\n'
            break
    md_path.write_text(text + section.lstrip(), encoding='utf-8')
    section_file = out / '测试集结果分析_第5节.md'
    section_file.write_text(section, encoding='utf-8')
    # 清理旧文件名
    old = out / '验证集结果分析_第5节.md'
    if old.is_file():
        old.unlink()
    print(json.dumps({
        'done': True, 'variant': variant,
        'valid_acc': round(summary['accuracy'], 4),
        'test_acc': round(test_summary['accuracy'], 4),
        'n_error_valid': err_info['n_error'],
        'n_error_test': test_err['n_error'],
        'n_cases': len(cases),
        'section': str(section_file),
        'analysis': cfg['analysis_md'],
    }, ensure_ascii=False), flush=True)


def CN(m):
    return {'text': '文本', 'audio': '语音', 'vision': '视觉'}.get(m, str(m))


def _fmt(x):
    if x is None:
        return '—'
    try:
        return f'{float(x):.3f}'
    except Exception:
        return '—'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['aligned', 'unaligned'], required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    run(args.variant, args.device)


if __name__ == '__main__':
    main()
