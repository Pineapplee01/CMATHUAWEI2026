"""未对齐版可视化：音视轴更长，分面板绘制。"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from problem3.report import save_pdf, _setup_font
from problem3_unaligned.explain import _grad_input, NAMES

CN = {'text': '文本', 'audio': '语音', 'vision': '视觉'}


def plot_figures(model, batch, card, fig_dir):
    _setup_font()
    fig_dir = Path(fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    rs, valids, _ = _grad_input(model, batch)
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.5))
    for ax, m, name in zip(axes, range(3), NAMES):
        rr = rs[m].copy(); rr = rr.astype(float); rr[~valids[m]] = np.nan
        ax.plot(np.arange(len(rr)), rr, color='#0072B2', lw=.9)
        for s in card['key_evidence'].get(name, []):
            ax.axvspan(s['start'], s['end'] - 1e-3, color='#E69F00', alpha=.2)
        ax.set_ylabel(CN[name]); ax.grid(alpha=.2)
    axes[-1].set_xlabel('各自轴索引（文本≤50，音视≤500）')
    axes[0].set_title(f'样本 {card["sample_id"]}  正向 Grad×Input（支持当前预测）')
    fig.tight_layout()
    save_pdf(fig, fig_dir / f'{card["sample_id"]}_importance.pdf')
    plt.close(fig)

    I = [card['internal_attribution'][n] for n in NAMES]
    phi = card.get('modality_marginal', {}).get('phi_p', [0, 0, 0])
    phi = [0 if (x is None or not np.isfinite(x)) else x for x in phi]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(x - .18, I, .36, label='内部归因 Im', color='#0072B2')
    ax.bar(x + .18, phi, .36, label='边际贡献 φm', color='#E69F00')
    ax.set_xticks(x, [CN[n] for n in NAMES]); ax.axhline(0, color='k', lw=.6)
    ax.legend(frameon=False); ax.grid(axis='y', alpha=.2)
    ax.set_title(f'样本 {card["sample_id"]}  三模态作用差异')
    fig.tight_layout()
    save_pdf(fig, fig_dir / f'{card["sample_id"]}_modality.pdf')
    plt.close(fig)
