"""问题一 v3：质量掩码图（中文标注，PDF）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import font_manager
from shared.plotting import pyplot


def _setup_cjk(plt) -> None:
    preferred = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "SimHei",
        "DejaVu Sans",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((n for n in preferred if n in available), "DejaVu Sans")
    fs = 11
    plt.rcParams.update(
        {
            "font.sans-serif": [chosen, "DejaVu Sans"],
            "font.family": "sans-serif",
            "font.size": fs,
            "axes.titlesize": fs,
            "axes.labelsize": fs,
            "xtick.labelsize": fs,
            "ytick.labelsize": fs,
            "legend.fontsize": fs,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def quality_plots(directory, output):
    """绘制并保存：

    - valid_fraction.pdf：（a）有效掩码占比；（b）支持掩码占比
    - quality_masks.pdf：（a）文本；（b）语音；（c）视觉 词槽有效掩码热图
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt = pyplot()
    _setup_cjk(plt)
    fs = float(plt.rcParams["font.size"])

    with np.load(Path(directory) / "masks.npz", allow_pickle=False) as data:
        text_m = data["text_mask"]
        audio_m = data["audio_mask"]
        vision_m = data["vision_mask"]
        audio_s = data["audio_support"] if "audio_support" in data.files else None
        vision_s = data["vision_support"] if "vision_support" in data.files else None

    # —— 图1：有效 / 支持 比例 ——
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    ax_a, ax_b = axes

    labels_valid = ["文本", "语音", "视觉"]
    vals_valid = [float(text_m.mean()), float(audio_m.mean()), float(vision_m.mean())]
    bars_a = ax_a.bar(labels_valid, vals_valid, color=["#4C78A8", "#F58518", "#54A24B"])
    ax_a.set_ylim(0, 1)
    ax_a.set_ylabel("占比")
    ax_a.set_title("（a）有效掩码占比", loc="center")
    ax_a.set_xlabel("模态")
    for bar, v in zip(bars_a, vals_valid):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.02,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=fs,
        )
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    if audio_s is not None and vision_s is not None:
        labels_sup = ["语音支持", "视觉支持"]
        vals_sup = [float(audio_s.mean()), float(vision_s.mean())]
        bars_b = ax_b.bar(labels_sup, vals_sup, color=["#F58518", "#54A24B"], alpha=0.85)
        for bar, v in zip(bars_b, vals_sup):
            ax_b.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.02,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=fs,
            )
    else:
        ax_b.bar(["无数据"], [0.0], color="#bbbbbb")
    ax_b.set_ylim(0, 1)
    ax_b.set_title("（b）音视支持掩码占比", loc="center")
    ax_b.set_xlabel("模态")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output / "valid_fraction.pdf", bbox_inches="tight")
    plt.close(fig)

    # —— 图2：样本×词槽 有效掩码 ——
    panels = [
        ("（a）文本有效掩码", text_m),
        ("（b）语音有效掩码", audio_m),
        ("（c）视觉有效掩码", vision_m),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.2), sharex=True)
    for ax, (title, mask) in zip(axes, panels):
        ax.imshow(mask, aspect="auto", vmin=0, vmax=1, interpolation="nearest", cmap="viridis")
        ax.set_title(title, loc="center")
        ax.set_ylabel("样本编号")
    axes[-1].set_xlabel("MFA 内容词槽（前 50，不含 CLS/SEP）")
    fig.tight_layout()
    fig.savefig(output / "quality_masks.pdf", bbox_inches="tight")
    plt.close(fig)

    # 可选保留旧 png 名提示：若存在旧 png，不强制删除
    return {
        "valid_fraction": str(output / "valid_fraction.pdf"),
        "quality_masks": str(output / "quality_masks.pdf"),
    }
