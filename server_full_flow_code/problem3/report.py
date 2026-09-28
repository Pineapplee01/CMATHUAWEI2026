"""问题三交付：划分集评价、解释卡排版、可视化、报告。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shared.common import path, json_write
from problem2_retrain_v2.data import batch, load_split, sha256
from problem2_retrain_v2.evaluation import metrics
from problem2_retrain_v2.output_contract import project_intensity
from problem3.explain import _grad_input, continuous_evidence, NAMES

LABELS = ('负', '中', '正')
CN = {'text': '文本', 'audio': '语音', 'vision': '视觉'}


def _setup_font():
    """中文用宋体（Noto Serif CJK SC），英文用罗马体（Times New Roman）。"""
    from matplotlib import font_manager

    root = Path(__file__).resolve().parent
    song_file = root / 'fonts' / 'NotoSerifCJK-SC-Regular.otf'
    # 若本地未抽出 SC 面，从系统 TTC 抽取一次
    if not song_file.is_file():
        ttc = Path('/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc')
        if ttc.is_file():
            try:
                from fontTools.ttLib import TTCollection
                song_file.parent.mkdir(parents=True, exist_ok=True)
                TTCollection(str(ttc)).fonts[2].save(str(song_file))  # face 2 = SC
            except Exception:
                song_file = None
        else:
            song_file = None

    roman_paths = (
        '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf',
    )
    roman = 'Times New Roman'
    for p in roman_paths:
        if Path(p).is_file():
            try:
                font_manager.fontManager.addfont(p)
                roman = font_manager.FontProperties(fname=p).get_name()
                break
            except Exception:
                continue

    song = 'Noto Serif CJK SC'
    if song_file and Path(song_file).is_file():
        try:
            font_manager.fontManager.addfont(str(song_file))
            song = font_manager.FontProperties(fname=str(song_file)).get_name()
        except Exception:
            song = 'AR PL UMing CN'  # 明体衬线回退
    elif not any(f.name == song for f in font_manager.fontManager.ttflist):
        song = 'AR PL UMing CN' if any(f.name == 'AR PL UMing CN' for f in font_manager.fontManager.ttflist) else song

    # matplotlib≥3.6：family 为列表时按字符逐字回退（拉丁→罗马，汉字→宋体）
    plt.rcParams.update({
        'font.family': [roman, song],
        'font.serif': [roman, song, 'DejaVu Serif'],
        'axes.unicode_minus': False,
        'mathtext.fontset': 'stix',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


_setup_font()


def save_pdf(fig, dest):
    dest = Path(dest)
    if dest.suffix.lower() != '.pdf':
        dest = dest.with_suffix('.pdf')
    fig.savefig(dest, bbox_inches='tight')
    return dest


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


@torch.no_grad()
def evaluate_split(model, train_cfg, saved, device, split='test', dtype=None):
    """在 data_dir/{split}.npz 上评价（不强制数据文件 SHA 匹配）。"""
    directory = Path(train_cfg['data_dir'])
    npz = directory / f'{split}.npz'
    if not npz.is_file():
        raise FileNotFoundError(npz)
    digest = sha256(npz)
    data = load_split(directory, split, dtype=getattr(np, train_cfg.get('precision', 'float32')))
    dtype = dtype or next(model.parameters()).dtype
    rows = []
    for lo in range(0, len(data['id']), 64):
        idx = torch.arange(lo, min(lo + 64, len(data['id'])))
        b = batch(data, idx, device, dtype=dtype)
        out = model(b)
        p = out['logits'].softmax(-1)
        c = p.argmax(-1)
        raw = out['intensity']
        y = torch.as_tensor(project_intensity(c.cpu().numpy(), raw.cpu().numpy()),
                            dtype=raw.dtype, device=raw.device)
        for i, sid in enumerate(b['id']):
            rows.append({
                'id': sid, 'true_class': int(b['c'][i]), 'true_intensity': float(b['y'][i]),
                'class': int(c[i]), 'intensity': float(y[i]), 'raw_intensity': float(raw[i]),
                **{f'p_{j}': float(p[i, j]) for j in range(3)},
            })
    frame = pd.DataFrame(rows)
    summary = metrics(frame)
    summary['split'] = split
    summary['split_sha256'] = digest
    summary['split_path'] = str(npz)
    summary['identity_checked'] = False
    return frame, summary


def evaluate_valid(model, train_cfg, saved, device, dtype=None):
    return evaluate_split(model, train_cfg, saved, device, split='valid', dtype=dtype)


def load_text_times(mapping_dir):
    root = path(mapping_dir)
    if not root.is_dir():
        return {}
    out = {}
    for f in root.glob('[0-9][0-9].json'):
        m = json.loads(f.read_text(encoding='utf-8'))
        out[f.stem] = m.get('text') or []
    return out


def _mfa_span(times, indices):
    """由词元下标取 MFA 起止时间与可见词面。"""
    segs, words = [], []
    for i in indices:
        if times is None or i is None or i < 0 or i >= len(times) or not times[i]:
            continue
        t = times[i]
        if 'start_seconds' not in t or 'end_seconds' not in t:
            continue
        segs.append(t)
        w = (t.get('text') or '').strip()
        if w and w not in ('[CLS]', '[SEP]', '[PAD]'):
            words.append(w)
    if not segs:
        return None, None, words
    return min(t['start_seconds'] for t in segs), max(t['end_seconds'] for t in segs), words


def _av_span_time(times, link):
    """单段音/视证据的时间：一词用该词 MFA；多词按首尾锚点词元拼接时段。"""
    if not times or not link:
        return None, None, []
    ends = link.get('endpoint_text') or [None, None]
    i0, i1 = ends[0], ends[1]
    # 展示词面：优先首尾锚点（按文本顺序），否则质量 Top
    show_ids = []
    for i in (i0, i1):
        if i is not None and i not in show_ids:
            show_ids.append(i)
    if not show_ids:
        show_ids = [int(i) for i in (link.get('text_indices') or []) if i is not None]
    words = []
    for i in show_ids:
        if 0 <= i < len(times) and times[i]:
            w = (times[i].get('text') or '').strip()
            if w and w not in ('[CLS]', '[SEP]', '[PAD]'):
                words.append(w)
    if i0 is not None and (i1 is None or i0 == i1):
        a, b, _ = _mfa_span(times, [i0])
        return a, b, words
    if i1 is not None and i0 is None:
        a, b, _ = _mfa_span(times, [i1])
        return a, b, words
    if i0 is not None and i1 is not None:
        if not (0 <= i0 < len(times) and times[i0] and 0 <= i1 < len(times) and times[i1]):
            return _mfa_span(times, show_ids)
        # 按文本顺序决定起点/终点；多词 MFA 时段首尾拼接
        left, right = (i0, i1) if i0 <= i1 else (i1, i0)
        t0 = times[left].get('start_seconds')
        t1 = times[right].get('end_seconds')
        if t0 is None or t1 is None:
            return _mfa_span(times, show_ids)
        return float(t0), float(t1), words
    return _mfa_span(times, show_ids)


def format_evidence(card, times=None):
    """解释卡字段：关键文本/语音/视觉。

    文本：直接使用词元自身 MFA 时间戳。
    未对齐：每个音/视证据窗单独关联词元——一词用该词时段，多词按首尾词元 MFA 拼接。
    对齐（无 span links）：音/视与文本同槽时借用 MFA 作近似。
    """
    assoc = card.get('soft_association') or {}
    span_links = {
        'audio': { (lk['start'], lk['end']): lk for lk in assoc.get('audio_span_links') or [] },
        'vision': { (lk['start'], lk['end']): lk for lk in assoc.get('vision_span_links') or [] },
    }
    unaligned = bool(span_links['audio'] or span_links['vision']
                     or assoc.get('audio_to_text') or assoc.get('vision_to_text'))
    lines = {}
    for m in NAMES:
        spans = card.get('key_evidence', {}).get(m, [])
        parts = []
        for s in spans[:3]:
            lo, hi = s['start'], s['end']
            if m == 'text' and s.get('tokens'):
                phrase = ' '.join(t for t in s['tokens'] if t not in ('[CLS]', '[SEP]', '[PAD]'))
                tip = ''
                a, b, _ = _mfa_span(times, range(lo, hi))
                if a is not None:
                    tip = f'，约 {a:.1f}–{b:.1f} s'
                parts.append(f'“{phrase}”（槽 {lo}:{hi}{tip}）' if phrase else f'槽 {lo}:{hi}{tip}')
            elif m in ('audio', 'vision') and unaligned and times:
                tip = ''
                link = span_links[m].get((lo, hi))
                if link is None and span_links[m]:
                    # 容错：按起止匹配最近链接
                    link = next((lk for lk in span_links[m].values()
                                 if lk['start'] == lo and lk['end'] == hi), None)
                a, b, words = _av_span_time(times, link) if link else (None, None, [])
                if a is not None:
                    shown = ' '.join(words[:4]) if words else '关联词元'
                    tip = f'，对应“{shown}”，约 {a:.1f}–{b:.1f} s'
                parts.append(f'槽 {lo}:{hi}{tip}')
            else:
                tip = ''
                if times and not unaligned:
                    a, b, _ = _mfa_span(times, range(lo, min(hi, len(times))))
                    if a is not None:
                        tip = f'，约 {a:.1f}–{b:.1f} s（槽对齐近似）'
                parts.append(f'槽 {lo}:{hi}{tip}')
        lines[m] = '；'.join(parts) if parts else '无显著局部段'
    return lines


def card_table_rows(card, times=None):
    I = card['internal_attribution']
    p = card.get('modality_marginal', {}).get('views', {}).get('TAV', {})
    pc = p.get('p')
    # 预测类概率：用完整视图缓存没有三类概率时，用 intensity 侧已有字段
    pred = f"{card['polarity']}，强度 {card['intensity']:.2f}"
    if pc is not None:
        pred += f'，该类概率 {pc:.2f}'
    ev = format_evidence(card, times)
    phi = card.get('modality_marginal', {}).get('phi_p', [None, None, None])
    if all(x is not None and np.isfinite(x) for x in phi):
        modality = (f"内部归因 文本 {I['text']:.2f} / 语音 {I['audio']:.2f} / 视觉 {I['vision']:.2f}；"
                    f"边际贡献 φ̃ 文本 {phi[0]:.3f} / 语音 {phi[1]:.3f} / 视觉 {phi[2]:.3f}")
    else:
        modality = f"文本 {I['text']:.2f}，语音 {I['audio']:.2f}，视觉 {I['vision']:.2f}"
    cs = []
    if card.get('comprehensiveness') is not None and np.isfinite(card['comprehensiveness']):
        cs.append(f"Comp {card['comprehensiveness']:.3f}")
    if card.get('sufficiency') is not None and np.isfinite(card['sufficiency']):
        cs.append(f"Suff {card['sufficiency']:.3f}")
    if card.get('stability') is not None and np.isfinite(card['stability']):
        cs.append(f"Stab {card['stability']:.3f}")
    fidelity = '；'.join(cs) if cs else '—'
    return [
        ('样本编号', card.get('sample_id')),
        ('预测结果', pred),
        ('模态作用', modality),
        ('主要参考模态', CN.get(card.get('main_support_modality'), card.get('main_support_modality'))),
        ('关键文本', ev['text']),
        ('关键语音', ev['audio']),
        ('关键画面', ev['vision']),
        ('Comp / Suff / Stab', fidelity),
        ('人工核查', '（待人工对照原视频填写）'),
    ]


def write_explanation_cards_md(cards, out_path, mapping_dir='problem3/results/mapping_v3_mfa', limit=None):
    """写出解释卡 Markdown。limit=None 表示全部样本；否则只写代表性子集。"""
    times_map = load_text_times(mapping_dir)
    picks = cards if limit is None else _pick_cards(cards, k=min(limit, len(cards)))
    title = '# 附件4全部样本解释卡' if limit is None else '# 典型样本解释卡'
    lines = [title, '']
    for i, card in enumerate(picks, 1):
        stem = Path(card.get('source_file', '')).stem
        times = times_map.get(stem) or times_map.get(card.get('sample_id', '')[:2])
        lines.append(f'## 表{i}　样本 {card.get("sample_id")}')
        lines.append('')
        lines.append('| 项目 | 内容 |')
        lines.append('|---|---|')
        for k, v in card_table_rows(card, times):
            lines.append(f'| {k} | {v} |')
        lines.append('')
    Path(out_path).write_text('\n'.join(lines), encoding='utf-8')


def _pick_cards(cards, k=3):
    if len(cards) <= k:
        return cards
    scored = []
    for c in cards:
        I = c['internal_attribution']
        scored.append((abs(c['intensity']) + max(I.values()), c))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [c for _, c in scored[:k]]


def plot_figures(model, batch, card, fig_dir, eta=0.7):
    _setup_font()
    fig_dir = Path(fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    r, valid, c_star, _ = _grad_input(model, batch)
    # 1) 三模态位置重要度
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=False)
    for ax, m, name in zip(axes, range(3), NAMES):
        rr = r[m].copy(); rr[~valid[m]] = np.nan
        ax.plot(np.arange(len(rr)), rr, color='#0072B2', lw=1.2)
        for s in card['key_evidence'].get(name, []):
            ax.axvspan(s['start'], s['end'] - 1e-3, color='#E69F00', alpha=.25)
        ax.set_ylabel(CN[name])
        ax.grid(alpha=.2)
    axes[-1].set_xlabel('槽位 / 帧索引')
    axes[0].set_title(f'样本 {card["sample_id"]}  正向 Grad×Input（阴影=支持当前预测的关键证据）')
    fig.tight_layout()
    save_pdf(fig, fig_dir / f'{card["sample_id"]}_importance.pdf')
    plt.close(fig)

    # 2) 模态作用对比：内部归因 vs 边际贡献（中文与数学符号分写，避免 $-mathtext 吞掉汉字）
    I = [card['internal_attribution'][n] for n in NAMES]
    phi = card.get('modality_marginal', {}).get('phi_p', [0, 0, 0])
    phi = [0 if (x is None or not np.isfinite(x)) else x for x in phi]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(x - .18, I, .36, label='内部归因 Im', color='#0072B2')
    ax.bar(x + .18, phi, .36, label='边际贡献 φm', color='#E69F00')
    ax.set_xticks(x, [CN[n] for n in NAMES])
    ax.axhline(0, color='k', lw=.6)
    ax.legend(frameon=False)
    ax.set_title(f'样本 {card["sample_id"]}  三模态作用差异')
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    save_pdf(fig, fig_dir / f'{card["sample_id"]}_modality.pdf')
    plt.close(fig)


def plot_split_summary(frame, summary, fig_dir, split='test'):
    _setup_font()
    fig_dir = Path(fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    stem = {'train': 'train', 'valid': 'valid', 'test': 'test'}.get(split, split)
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(frame.true_class, frame['class']):
        cm[int(t), int(p)] += 1
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(3), LABELS); ax.set_yticks(range(3), LABELS)
    ax.set_xlabel('预测'); ax.set_ylabel('真实')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center')
    # 图题写在正文/图注，图内不再放标题
    fig.colorbar(im, ax=ax, fraction=.046)
    fig.tight_layout()
    save_pdf(fig, fig_dir / f'{stem}_confusion.pdf')
    plt.close(fig)

    err = frame[frame.true_class != frame['class']]
    fig, ax = plt.subplots(figsize=(5, 3))
    counts = frame.true_class.value_counts().reindex([0, 1, 2], fill_value=0)
    wrong = err.true_class.value_counts().reindex([0, 1, 2], fill_value=0)
    rate = wrong.values / np.maximum(counts.values, 1)
    ax.bar([LABELS[i] for i in range(3)], rate, color='#D55E00')
    ax.set_ylabel('错误率')
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    save_pdf(fig, fig_dir / f'{stem}_error_by_class.pdf')
    plt.close(fig)
    return {
        'n_error': int(len(err)),
        'error_rate_by_true_class': {LABELS[i]: float(rate[i]) for i in range(3)},
        'confusion': cm.tolist(),
    }


def plot_valid_summary(frame, summary, fig_dir):
    return plot_split_summary(frame, summary, fig_dir, split='valid')


def write_deliverable_report(*, cfg, digest, train_cfg, saved, cards, summary, error_info, out_dir, variant='aligned'):
    out = Path(out_dir)
    I_mean = {n: float(np.mean([c['internal_attribution'][n] for c in cards])) for n in NAMES}
    support = pd.Series([c.get('main_support_modality') for c in cards]).value_counts().to_dict()
    lines = [
        '# 问题三交付说明（对齐版）' if variant == 'aligned' else '# 问题三交付说明（未对齐版）',
        '',
        '对应要求：(1) 可解释建模；(2) 典型解释卡；(3) 重要性与模态差异可视化；'
        '(4) 附件4全量预测与解释；(5) 测试集评价与错误归因。',
        '',
        '## (1) 可解释性模型：原理、结构、目标与参数',
        '',
        '问题三**不另训网络**，复用问题二已冻结预测器作黑盒，在其内部时序表示上做事后解释。',
        '',
        '**预测主干（问题二）**：对齐版为 F1（观测条件补全 → 五槽窗池化 → Concat-MLP）；'
        '未对齐版为 SoftAlignment（单调局部窗软注意力 → 按词融合）。详见问题二模型文档。',
        '',
        '**解释层结构**（推理期）：',
        '1. 正向 Gradient×Input：'
        '$r_{m,t}=\\max\\bigl(0,\\sum_j H_{m,t,j}\\partial s_{c^*}/\\partial H_{m,t,j}\\bigr)$'
        '（只保留抬高当前预测类的槽）；',
        '2. 多上下文模态边际贡献：在非空组合 $\\{T,A,V,TA,TV,AV,TAV\\}$ 上平均 $\\tilde\\phi_m$；',
        '3. 连续证据提取：贪心覆盖归因质量 $\\eta$ 后填缝合并为 $E_T,E_A,E_V$（允许互不相同）；',
        '4. 分别删除复核，并计算 Comp / Suff / Stab（两版同构：音视观测位加噪，证据窗 Jaccard）。',
        '',
        '**目标函数**：解释层无训练损失；预测器损失仍为问题二的加权平滑 CE + Huber + 有序项 + 中性边界。',
        '',
        '**训练方案**：问题三零训练；检查点冻结后只做前向 / 反传归因。',
        '',
        f'- 检查点 SHA256：`{digest}`',
        f'- 训练 best_epoch：{saved.get("best_epoch")}',
        f'- `evidence_eta`={cfg.get("evidence_eta", 0.7)}',
        f'- `stability_repeats`={cfg.get("stability_repeats", 3)}，`stability_noise`={cfg.get("stability_noise", 0.01)}',
        f'- device=`{cfg.get("device")}`，seed=`{cfg.get("seed")}`',
        '',
        '## (2) 典型样本解释卡',
        '',
        '见同目录 [典型样本解释卡.md](典型样本解释卡.md)。字段对齐论文模板：预测结果、三模态作用、'
        '主要参考模态、关键文本/语音/画面、Comp/Suff/Stab、人工核查栏。',
        '',
        '## (3) 局部重要性与三模态作用可视化',
        '',
        '图目录：`figures/`。每个代表性样本含：',
        '- `*_importance.pdf`：三模态正向 Grad×Input 曲线（支持当前预测），阴影为关键证据窗；',
        '- `*_modality.pdf`：内部归因 $I_m$ 与边际贡献 $\\tilde\\phi_m$ 对照。',
        '',
        f'附件4共 {len(cards)} 条上，内部归因均值：文本 {I_mean["text"]:.3f}，'
        f'语音 {I_mean["audio"]:.3f}，视觉 {I_mean["vision"]:.3f}；'
        f'主要参考模态计数：{support}；'
        f'Comp/Suff/Stab 均值：'
        f'{float(np.nanmean([c.get("comprehensiveness") for c in cards])):.3f} / '
        f'{float(np.nanmean([c.get("sufficiency") for c in cards])):.3f} / '
        f'{float(np.nanmean([c.get("stability") for c in cards])):.3f}'
        f'（$\\sigma$={cfg.get("stability_noise", 0.01)}, $R$={cfg.get("stability_repeats", 3)}）。'
        f'全部 {len(cards)} 条解释卡见 [附件4_全部解释卡.md](附件4_全部解释卡.md)；'
        f'图目录 `figures/` 含每条样本的 `*_importance.pdf` 与 `*_modality.pdf`。',
        '',
        '## (4) 附件4全量预测与解释',
        '',
        '- 汇总表：[附件4_可解释结果.csv](附件4_可解释结果.csv)（含 Comp / Suff / Stab）',
        '- 全量卡片 JSON：[解释卡片.json](解释卡片.json)',
        '- 全量解释卡：[附件4_全部解释卡.md](附件4_全部解释卡.md)',
        '- 典型 3 张：[典型样本解释卡.md](典型样本解释卡.md)',
        '',
        '## (5) 测试集基础性能、可视化与错误归因',
        '',
        f'- 数据：`{summary.get("split_path", "data_dir/test.npz")}`',
        f'- 样本数：{error_info.get("n_test", error_info.get("n_valid", len(cards)))}',
        f'- 划分 SHA256：`{summary.get("split_sha256", "—")}`'
        + ('（与检查点核对一致）' if summary.get('identity_checked') else '（检查点未登记 test，仅记录当前文件指纹）'),
        f'- ACC：{summary["accuracy"]:.2%}',
        f'- Macro-F1：{summary.get("macro_f1", float("nan")):.2%}',
        f'- MAE：{summary.get("mae", float("nan")):.4f}',
        f'- Pearson：{summary.get("pearson", float("nan")):.4f}',
        '',
        '图：`figures/test_confusion.pdf`、`figures/test_error_by_class.pdf`。',
        '',
        f'错误样本数 {error_info["n_error"]}；按真实类错误率：'
        + '，'.join(f'{k} {v:.1%}' for k, v in error_info['error_rate_by_true_class'].items()) + '。',
        '',
        '结论：解释层不改变测试集预测；错误归因仍由分类头混淆结构主导（见图）。'
        '附件4无标签，不报告准确率，仅提供预测与可复核解释。',
        '',
    ]
    # fix n display
    n_valid = error_info.get('n_valid')
    if n_valid:
        lines = [ln.replace('见 metrics', str(n_valid)) if '见 metrics' in ln else ln for ln in lines]
    (out / '问题三交付说明.md').write_text('\n'.join(lines), encoding='utf-8')
