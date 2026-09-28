"""F1组件拆除消融图：B0=完整模型。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from shared.plotting import pyplot

ORDER = ['B0', 'noC', 'noQ', 'noB', 'noBeta']
COLORS = {
    'B0': '#0072B2', 'noC': '#E69F00', 'noQ': '#009E73',
    'noB': '#CC79A7', 'noBeta': '#D55E00',
}
CAPTIONS = [
    ('01_ablation', 'F1组件拆除主结果',
     '点区间图；横轴实验组，纵轴ACC、Macro-F1、强度MAE。',
     'B0为完整F1。noC拆除补全，noQ拆除可靠性q，noB拆除边界损失，noBeta拆除类别平衡。主结果为完整输入test（无额外人工遮挡）；各组在同一条件下对照。点为训练seed，区间为跨seed均值±标准差。',
     '检验正式F1各组件在完整输入下的独立贡献。'),
    ('02_robustness', '连续缺失跨度与鲁棒性',
     '三列T/A/V缺失；上排Macro-F1，下排MAE；横轴缺失跨度。',
     '各组共享相同缺失区间。0%点为各组完整输入性能。随机掩码重复在每个训练seed内先平均；阴影为跨seed标准差。',
     '检验拆除收益/损失是否随缺失严重程度变化。'),
    ('03_location', '缺失模态与位置',
     '热力图；行是七种模态组合，列是位置；显示B0、noC及B0−noC。',
     '固定主缺失比例。随机位置重复先合并；绝对性能共用色标，差值以0为中心，单位百分点。',
     '定位补全模块在何种场景贡献最大。'),
    ('04_stability', '完整与缺失视图稳定性',
     '点区间图；横轴模型，纵轴JS散度、极性翻转率、原始强度变化。',
     '同模型同样本配对比较完整与缺失视图。强度用未投影原始输出。先合并掩码重复再对模态等权。',
     '检验拆除组件是否加剧完整/缺失预测不一致。'),
    ('05_reconstruction', '补全质量（仅含补全的组）',
     '上排T/A/V缺失跨度与余弦距离；下排中位误差样本热力图。',
     '只评价人工遮挡且原本可观测的槽。完整参考来自同一模型。观测均值对照仅用缺失视图可见特征。样本取B0首个seed中位误差。',
     '检验补全是否比观测均值更接近完整表示；接近不自动等于任务增益。'),
]


def curve(ax, frame, x, y, variant, linestyle='-', label=None):
    seedwise = frame.groupby(['seed', x], as_index=False)[y].mean()
    agg = seedwise.groupby(x)[y].agg(['mean', 'std']).sort_index()
    if agg.empty:
        return
    xs = agg.index.to_numpy()
    mean = agg['mean'].to_numpy()
    sd = agg['std'].to_numpy()
    ax.plot(xs, mean, linestyle=linestyle, marker='o' if linestyle == '-' else 's',
            markersize=3, color=COLORS[variant], label=label or variant)
    if np.isfinite(sd).all():
        ax.fill_between(xs, mean - sd, mean + sd, color=COLORS[variant], alpha=.12)


def dots(ax, frame, metric, order):
    for i, v in enumerate(order):
        values = frame.loc[frame.variant == v, metric].dropna().to_numpy()
        if not len(values):
            continue
        ax.scatter(i + np.linspace(-.08, .08, len(values)), values, s=15, alpha=.55, color=COLORS[v])
        sd = values.std(ddof=1) if len(values) > 1 else None
        ax.errorbar(i, values.mean(), yerr=sd, fmt='o', color=COLORS[v], capsize=4, markersize=6)
    ax.set_xticks(range(len(order)), order)
    ax.grid(axis='y', alpha=.2)


def plot_all(root):
    root = Path(root)
    protocol = json.loads((root / 'protocol.json').read_text())
    spec = protocol['identity']['settings']
    unaligned = protocol.get('branch') == 'unaligned'
    required = ['primary_by_seed', 'metrics']
    if not unaligned:
        required.append('stability')
    tables = {name: pd.read_csv(root / f'{name}.csv') for name in required}
    rec_path = root / 'reconstruction.csv'
    has_rec = rec_path.exists() and len(pd.read_csv(rec_path)) > 0
    if has_rec:
        tables['reconstruction'] = pd.read_csv(rec_path)
    if unaligned:
        # 未对齐消融只评完整输入：只画主表三点图
        plt = pyplot()
        plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
        dest = root / 'figures'
        dest.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
        for ax, key, label in zip(axes, ['accuracy', 'macro_f1', 'mae'],
                                  ['Accuracy ↑', 'Macro-F1 ↑', 'Intensity MAE ↓']):
            dots(ax, tables['primary_by_seed'], key, ORDER)
            ax.set_ylabel(label)
        fig.suptitle('soft_alignment component knockout (full input)')
        fig.tight_layout()
        fig.savefig(dest / f'{CAPTIONS[0][0]}.png', dpi=300, bbox_inches='tight')
        fig.savefig(dest / f'{CAPTIONS[0][0]}.pdf', bbox_inches='tight')
        plt.close(fig)
        (root / '图表说明.md').write_text(
            '# 图表说明\n\n未对齐消融仅完整输入主表；无局部缺失/补全辅图。\n')
        return
    plt = pyplot()
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    dest = root / 'figures'
    dest.mkdir(exist_ok=True)

    def save(fig, index):
        if spec['smoke']:
            fig.text(.5, .005, 'SMOKE TEST — NOT RESEARCH RESULTS', ha='center', color='red', fontsize=10)
        fig.savefig(dest / f'{CAPTIONS[index][0]}.png', dpi=300, bbox_inches='tight')
        fig.savefig(dest / f'{CAPTIONS[index][0]}.pdf', bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    for ax, key, label in zip(axes, ['accuracy', 'macro_f1', 'mae'],
                              ['Accuracy ↑', 'Macro-F1 ↑', 'Intensity MAE ↓']):
        dots(ax, tables['primary_by_seed'], key, ORDER)
        ax.set_ylabel(label)
    fig.suptitle('F1 component knockout under continuous missingness')
    fig.tight_layout()
    save(fig, 0)

    scores = tables['metrics']
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for col, mod in enumerate('TAV'):
        for row, key in enumerate(['macro_f1', 'mae']):
            ax = axes[row, col]
            for variant in ORDER:
                frame = scores[(scores.variant == variant) & (scores.modalities == mod) & (scores.position == 'random')]
                zero = scores[(scores.variant == variant) & (scores.rate == 0)]
                curve(ax, pd.concat([frame, zero]), 'rate', key, variant)
            ax.set(xlabel='Continuous span / valid span', ylabel=key, title=f'{mod} missing')
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    names = ['T', 'A', 'V', 'TA', 'TV', 'AV', 'TAV']
    positions = ['start', 'middle', 'end', 'random']
    subset = scores[np.isclose(scores.rate, spec['main_rate'])]

    def heat(v):
        frame = subset[subset.variant == v].groupby(['seed', 'modalities', 'position']).macro_f1.mean().reset_index()
        return frame.groupby(['modalities', 'position']).macro_f1.mean().unstack().reindex(
            index=names, columns=positions).to_numpy() * 100

    b0, noc = heat('B0'), heat('noC')
    bound = max(1, float(np.nanmax(np.abs(b0 - noc))))
    for ax, array, title in zip(axes, [b0, noc, b0 - noc],
                                ['B0 Macro-F1 (%)', 'noC Macro-F1 (%)', 'B0 − noC (pp)']):
        delta = title.startswith('B0 −')
        im = ax.imshow(array, cmap='RdBu_r' if delta else 'viridis',
                       vmin=-bound if delta else 0, vmax=bound if delta else 100)
        for i in range(7):
            for j in range(4):
                ax.text(j, i, f'{array[i, j]:.1f}', ha='center', va='center', fontsize=8,
                        color='black' if delta else 'white')
        ax.set(xticks=range(4), xticklabels=positions, yticks=range(7), yticklabels=names, title=title)
        fig.colorbar(im, ax=ax, fraction=.045)
    fig.tight_layout()
    save(fig, 2)

    stable = tables['stability']
    stable = stable[np.isclose(stable.rate, spec['main_rate']) & (stable.position == 'random')]
    stable = stable.groupby(['variant', 'seed', 'modalities'], as_index=False)[
        ['js', 'flip_rate', 'raw_intensity_shift']].mean()
    stable = stable.groupby(['variant', 'seed'], as_index=False)[
        ['js', 'flip_rate', 'raw_intensity_shift']].mean()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    for ax, key in zip(axes, ['js', 'flip_rate', 'raw_intensity_shift']):
        dots(ax, stable, key, ORDER)
        ax.set_ylabel(key + ' ↓')
    fig.suptitle('Paired full/missing prediction stability')
    fig.tight_layout()
    save(fig, 3)

    if has_rec and (tables['reconstruction']['variant'] == 'B0').any():
        rec = tables['reconstruction']
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
        for col, mod in enumerate('TAV'):
            ax = axes[0, col]
            for variant in ('B0', 'noQ'):
                if variant not in set(rec.variant):
                    continue
                for kind, style in [('reconstructed', '-'), ('observed_mean', '--')]:
                    frame = rec[(rec.variant == variant) & (rec.modalities == mod)
                                & (rec.position == 'random') & (rec.kind == kind)]
                    curve(ax, frame, 'rate', 'cosine_distance', variant, style, f'{variant} {kind}')
            ax.set(xlabel='Continuous span / valid span', ylabel='Cosine distance ↓',
                   title=f'{mod} compensation')
        axes[0, 0].legend(fontsize=6)
        example = root / 'runs' / f'B0_seed{spec["seeds"][0]}' / 'latent_example.npz'
        if example.exists():
            with np.load(example) as e:
                valid = e['valid'] if 'valid' in e else np.ones(50, dtype=bool)
                stop = int(np.where(valid)[0].max()) + 1
                compensated = np.where(e['mask'][:, None], e['reconstructed'], e['observed'])
                arrays = [e['reference'], e['observed'], compensated]
                bound = max(.01, float(np.percentile(
                    np.abs(np.concatenate([a[valid].ravel() for a in arrays])), 99)))
                for ax, array, title in zip(
                        axes[1], arrays,
                        ['Full reference', 'Missing input', 'Compensated representation']):
                    display = np.ma.array(
                        array[:stop].T, mask=np.broadcast_to(~valid[:stop], (array.shape[1], stop)))
                    im = ax.imshow(display, aspect='auto', cmap='RdBu_r', vmin=-bound, vmax=bound)
                    slots = np.where(e['mask'])[0]
                    for x in (slots.min() - .5, slots.max() + .5):
                        ax.axvline(x, color='black', ls='--', lw=.8)
                    ax.set(xlabel='Aligned slot', ylabel='Latent dim', title=title)
                fig.colorbar(im, ax=axes[1, 2], fraction=.045, label='Latent activation')
        fig.tight_layout()
        save(fig, 4)

    used = CAPTIONS[:5] if has_rec else CAPTIONS[:4]
    lines = ['# 图表说明', '',
             '主图为组件拆除与鲁棒性；辅助图为稳定性与补全质量。误差区间为跨训练seed标准差。']
    for name, title, contents, caption, goal in used:
        caption = caption.replace('30%', f"{spec['main_rate']:.0%}")
        lines += ['', f'## {name}: {title}', f'图类型/内容：{contents}',
                  f'图注：{caption}', f'论证目标：{goal}', f'![{title}](figures/{name}.png)']
    if spec['smoke']:
        lines.insert(2, '**本目录全部图仅为流程测试，不能用于论文实证结论。**')
    (root / '图表说明.md').write_text('\n\n'.join(lines) + '\n')
