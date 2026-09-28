"""seed=2026验证集敏感性分析；图表不画多seed误差棒。"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from 超参敏感性实验 import HERE, ROOT, OUT, SEEDS, GRIDS, SOFT_GRIDS, candidates, base_config

NAMES = {'aligned': '对齐', 'unaligned': '未对齐'}
PARAMS = {'balance_beta': ('类别平衡强度', 'β'), 'boundary_weight': ('边界损失权重', 'λ'),
          'bert_lr': ('BERT微调学习率', 'Learning rate'),
          'align_max_half_frac': ('软对齐窗口半宽上限', 'Window fraction'),
          'align_time_penalty': ('软对齐时间惩罚', 'Time penalty')}
METRICS = ['accuracy', 'macro_f1', 'mae', 'pearson', 'neutral_precision', 'neutral_recall', 'neutral_f1']


def collect(require_complete=True, include_joint=True):
    rows = []
    for layout in SEEDS:
        cs = candidates(layout)
        joint = OUT / layout / 'joint_parameters.json'
        if include_joint and joint.exists():
            cs += [('joint', json.loads(joint.read_text()))]
        for name, changes in cs:
            for seed in SEEDS[layout]:
                run = OUT / layout / f'{name}_seed{seed}'
                if not (run / 'summary.json').exists():
                    if require_complete:
                        raise FileNotFoundError(f'实验未完成: {run}')
                    continue
                result = json.loads((run / 'summary.json').read_text())
                history = json.loads((run / 'history.json').read_text())
                best = history[0]
                for record in history[1:]:
                    if record['selection_score'] > best['selection_score'] + 1e-8:
                        best = record
                if best['epoch'] != result['best_epoch'] or abs(best['selection_score']-result['selection_score']) > 1e-8:
                    raise ValueError(f'选轮次记录不一致: {run}')
                cfg = json.loads((run / 'config.json').read_text())
                if result['seed'] != seed or any(cfg[k] != v for k, v in changes.items()):
                    raise ValueError(f'实验参数不一致: {run}')
                row = {'layout': layout, 'candidate': name, 'seed': seed,
                       'score': result['selection_score'], 'best_epoch': result['best_epoch'],
                       'epochs_run': result['epochs_run'], 'seconds': result['elapsed_seconds'],
                       'effective_boundary_weight': result['effective_boundary_weight'],
                       'checkpoint': result['checkpoint'], 'checkpoint_sha256': result['checkpoint_sha256']}
                row.update({k: cfg[k] for k in GRIDS | (SOFT_GRIDS if layout == 'unaligned' else {})})
                row.update({f'{view}_{k}': result[view][k] for view in ('complete', 'local30') for k in METRICS})
                rows.append(row)
    return pd.DataFrame(rows)


def rank(frame):
    stats = frame.groupby('candidate').agg(score=('score', 'mean'), f1=('complete_macro_f1', 'mean'))
    stats['is_baseline'] = (stats.index == 'baseline').astype(int)
    stats['score_key'] = stats['score'].round(10)
    stats['f1_key'] = stats['f1'].round(10)
    return stats.sort_values(['score_key', 'f1_key', 'is_baseline'], ascending=[False, False, False])


def preferred_parameters(frame, layout):
    selected = {}
    subset = frame[frame.layout == layout]
    for key in GRIDS | (SOFT_GRIDS if layout == 'unaligned' else {}):
        relevant = subset[(subset.candidate == 'baseline') | subset.candidate.str.startswith(key + '_')]
        name = rank(relevant).index[0]
        selected[key] = float(relevant[relevant.candidate == name][key].iloc[0])
    return selected


def style():
    font_manager.fontManager.addfont(HERE / 'fonts/simsun.ttc')
    plt.rcParams.update({'font.family': ['Times New Roman', 'SimSun'], 'font.size': 11,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#A6B2BB',
        'axes.linewidth': .7, 'axes.axisbelow': True, 'grid.color': '#E5EAED',
        'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'path', 'savefig.dpi': 360})


def save(fig, name):
    folder = OUT / 'figures'
    folder.mkdir(exist_ok=True)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(folder / f'{name}.{ext}', bbox_inches='tight', pad_inches=.12)
    plt.close(fig)


def table(frame, include_neutral=False):
    cols = [('score', 100), ('complete_accuracy', 100), ('complete_macro_f1', 100),
            ('complete_mae', 1), ('complete_pearson', 1), ('local30_accuracy', 100)]
    headings = ['Validation score (%)', 'Accuracy (%)', 'Macro-F1 (%)', 'MAE', 'Pearson r', 'Local30 Accuracy (%)']
    if include_neutral:
        cols += [('complete_neutral_precision',100), ('complete_neutral_recall',100)]
        headings += ['Neutral Precision (%)', 'Neutral Recall (%)']
    lines = ['| 取值/配置 | '+' | '.join(headings)+' |', '|---|'+'---:|'*len(cols)]
    for label, row in frame.iterrows():
        lines.append('| '+str(label)+' | '+' | '.join(f'{row[k]*scale:.3f}' for k,scale in cols)+' |')
    return lines+['']


def report():
    for layout in SEEDS:
        if not (OUT / layout / 'joint_parameters.json').exists():
            raise RuntimeError('请先运行 --joint-only 完成联合复核')
    frame = collect()
    frame.to_csv(OUT / 'all_validation_runs.csv', index=False)
    style()
    lines = ['# 当前架构的超参数敏感性分析（seed=2026）', '',
        '**结论范围：** 本次固定seed=2026，用验证集调参。多seed评估留到方案固定以后，本报告不估计跨seed标准差，不画误差棒。', '',
        '## 实验设计', '',
        '只使用附件2 train 学习参数、valid 选轮次与比较超参数，未读取test或附件3、4。对齐与未对齐分别比较；每次只改变一个参数，其余保持当前默认。对齐9组、未对齐13组，再各做1组联合复核，共24个配置。', '',
        '保持当前模型架构、通用BERT逐槽全量微调、float32、训练数据、缺失掩码与样本顺序不变；对齐版沿用同一共享投影初始化，未对齐版不从对齐检查点迁移。每次最多12轮，patience=4，batch_size=16，累积4次更新。默认组按当前代码重新训练；本轮中已完成的seed2026同配置记录直接复用。', '',
        r'检查点按 $S=0.7\,Accuracy_{valid,full}+0.3\,Accuracy_{valid,TAV30}$ 选择；候选按S排序，完全同分时比较完整Macro-F1，再优先默认配置。S是当前代码的选模规则，不是题目官方综合指标。', '',
        '四项任务指标分别报告，MAE和Pearson沿用当前模型输出协议的极性一致性投影强度。局部缺失验证视图为三模态连续30%，掩码seed=4026。正文表格给出完整视图四项指标和缺失Accuracy，完整逐项数据保存在CSV。', '',
        '## 参数范围与作用', '',
        r'类别权重 $w_c=\frac{(N/(3n_c))^\beta}{\sum_j(n_j/N)(N/(3n_j))^\beta}$。β扫描0、0.125、0.25、0.5；依次检验无平衡、更弱平衡、当前温和平衡及更强平衡。权重由3395条训练样本计算，负/中/正数量为967/758/1670。', '',
        r'损失为 $\mathcal L=\mathcal L_{CE}+0.2\mathcal L_{Huber}+0.1\mathcal L_{ordinal}+\lambda_e\mathcal L_{boundary}$，$\lambda_e=\lambda_{max}\min(1,e/3)$。λ_max扫描0、0.05、0.1、0.2，对照无约束、半强度、当前值、双强度。边界间隔与弱极性强度上限均固定0.5。最佳轮次在前三轮时，实际λ_e可能小于λ_max，CSV记录了实际值。', '',
        r'边界项令 $r_i=z_{i,N}-\log(\exp z_{i,-}+\exp z_{i,+})$，对真实中性i与弱极性j采用 $\operatorname{softplus}(m-r_i+r_j)$ 的批内均值。其对r_i的导数为负、对r_j为正，所以增大λ会加强二者分离；过大则可能压过主分类目标。参数实验用于检验这种权衡是否在验证数据中出现。', '',
        r'BERT学习率扫描 $5\times10^{-6}$、$10^{-5}$、$2\times10^{-5}$；新模块学习率固定$10^{-4}$，用于比较预训练表示的适应速度与过度更新。', '',
        r'未对齐窗内注意力为 $e_{kj}=q_k^Tk_j/\sqrt{d_h}-\gamma|j-c_k|/L_m$。原始半宽 $h_k=\max(1,L_m\rho\sigma(w_h^Tq_k+b_h))$，随后施加单调边界修正。ρ取0.1/0.2/0.3，γ取1/2/4。前者控制可读取上下文的范围，后者抑制远离中心的匹配；ρ是半宽上限，不是最终窗口比例。', '',
        '这些取值围绕默认值做局部扫描，不能证明全局最优。下面直接依据测量结果判断默认值能否保留，不为默认参数编造理由。', '']
    for idx, (key, (title, xlabel)) in enumerate(PARAMS.items(), 1):
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.7), layout='constrained')
        second = 'complete_neutral_f1' if key in ('balance_beta','boundary_weight') else 'complete_macro_f1'
        sections = []
        for layout, color, marker in [('aligned','#44789A','o'), ('unaligned','#BD5148','s')]:
            if key in SOFT_GRIDS and layout == 'aligned': continue
            f = frame[(frame.layout == layout) & ((frame.candidate == 'baseline') | frame.candidate.str.startswith(key+'_'))].sort_values(key)
            for ax, metric, ylabel in zip(axes, ['score', second], ['Validation score (%)','Neutral F1 (%)' if 'neutral' in second else 'Macro-F1 (%)']):
                ax.plot(f[key], f[metric]*100, marker+'-', color=color, ms=5, lw=1.7, label=NAMES[layout])
                ax.set(xlabel=xlabel, ylabel=ylabel, xticks=f[key])
                if key == 'bert_lr':
                    ax.set_xscale('log');ax.set_xticks(f[key], ['5e-6','1e-5','2e-5']);ax.minorticks_off()
                ax.grid(axis='y', alpha=.7);ax.margins(x=.15)
            best_name = rank(f).index[0]
            best = f[f.candidate == best_name].iloc[0]
            base = f[f.candidate == 'baseline'].iloc[0]
            value, default = float(best[key]), float(base[key])
            delta = (best.score-base.score)*100
            sections += [f'### {NAMES[layout]}', '', f'当前默认值 **{default:g}**；本次候选最优为 **{value:g}**，Validation score={best.score*100:.3f}%，相对默认变化 **{delta:+.3f}个百分点**。', '']
            sections += table(f.set_index(key), key in ('balance_beta','boundary_weight'))
            if value == default:
                sections += ['**取值依据：** 默认值在扫描范围内的验证选模分数最高，因此本次结果支持保留；不据此断言对其他seed也最优。', '']
            else:
                sections += [f'**取值依据：** 验证选模分数支持改为{value:g}，不支持把原默认{default:g}写成最优。相对默认，Macro-F1变化{(best.complete_macro_f1-base.complete_macro_f1)*100:+.3f}个百分点，中性Recall变化{(best.complete_neutral_recall-base.complete_neutral_recall)*100:+.3f}个百分点，MAE变化{best.complete_mae-base.complete_mae:+.4f}（负数更好）。最终是否采用还需看联合复核。', '']
            if best.best_epoch == 12:
                sections += ['该候选的最佳检查点位于12轮训练预算上限，尚不能据此断言训练已经收敛；当前比较是在统一预算下进行。', '']
            ordered = f[key].tolist()
            if value != 0 and value in (min(ordered),max(ordered)):
                sections += ['候选最优位于扫描边界，说明目前只能选择已测范围内更好的值，不能宣称该处是内部最优点。', '']
        axes[0].legend()
        fig.suptitle(title+'的验证集敏感性 · seed=2026', fontsize=14)
        name = f'{idx:02d}_{key}'
        save(fig,name)
        lines += [f'## {idx}. {title}', '', f'![{title}](sensitivity_results/figures/{name}.png)', '',
            '**图注：** 横轴为参数取值；左图为完整/缺失验证Accuracy的加权选模分数，右图为'+('完整验证Neutral F1' if 'neutral' in second else '完整验证Macro-F1')+'。每个点为seed=2026独立训练后按同一规则选取的检查点；连线连接实际测量点，不表示中间值也经过实验。两种颜色对应数据布局；未对齐专用参数仅绘未对齐结果。无误差棒、无阴影。', ''] + sections
    lines += ['## 联合复核与整套参数建议', '', '将各单因素候选最优值组合后重新训练，再与默认和各单因素实测配置比较。不同参数的单独改善不能直接相加，最终选择实际测量过的整套配置。', '']
    recommendation = {}
    fig, axes = plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for ax, layout in zip(axes,SEEDS):
        sub=frame[frame.layout==layout]
        chosen=rank(sub).index[0]
        cfg=json.loads((OUT/layout/f'{chosen}_seed2026'/'config.json').read_text())
        values={k:cfg[k] for k in GRIDS | (SOFT_GRIDS if layout=='unaligned' else {})}
        best=sub[sub.candidate==chosen].iloc[0]
        recommendation[layout]={'seed':2026,'candidate':chosen,'parameters':values,'validation_score':float(best.score)}
        names=list(dict.fromkeys(['baseline','joint',chosen]))
        shown=sub.set_index('candidate').loc[names]
        ax.plot(range(len(names)),shown.score*100,'o-',lw=1.6,color='#44789A')
        for i,score in enumerate(shown.score*100):
            ax.annotate(f'{score:.2f}',(i,score),xytext=(0,8),textcoords='offset points',ha='center')
        ax.set(title=NAMES[layout],ylabel='Validation score (%)',xticks=range(len(names)),xticklabels=[{'baseline':'当前默认','joint':'联合参数'}.get(n,'最终候选') for n in names])
        ax.margins(x=.22,y=.3);ax.grid(axis='y',alpha=.7)
        lines += [f'### {NAMES[layout]}', '', f'建议配置：**{chosen}**，参数 `{json.dumps(values,ensure_ascii=False)}`。', '']+table(shown)
        baseline_row = sub[sub.candidate=='baseline'].iloc[0]
        joint_row = sub[sub.candidate=='joint'].iloc[0]
        lines += [f'整套取值依据：相对当前默认，所选配置的Validation score变化{(best.score-baseline_row.score)*100:+.3f}个百分点，完整Accuracy变化{(best.complete_accuracy-baseline_row.complete_accuracy)*100:+.3f}个百分点，Macro-F1变化{(best.complete_macro_f1-baseline_row.complete_macro_f1)*100:+.3f}个百分点；中性Recall变化{(best.complete_neutral_recall-baseline_row.complete_neutral_recall)*100:+.3f}个百分点。', '']
        if chosen != 'joint':
            lines += [f'联合组合的选模分数为{joint_row.score*100:.3f}%，低于或未优于最终候选。因此没有把所有单因素最优值直接拼成正式建议，而是保留已实测更优的整套参数。这也解释了为什么最终组合中的个别参数可能不同于其单因素最优值。', '']
        else:
            lines += ['联合组合取得本次已测候选中的最高选模分数，支持采用这套组合；组件贡献仍应由前面的单因素曲线分别解释，不应将各自增益相加。', '']
    fig.suptitle('联合参数复核 · seed=2026',fontsize=14)
    save(fig,'06_joint_verification')
    lines += ['![联合参数复核](sensitivity_results/figures/06_joint_verification.png)', '', '**图注：** 展示默认组合、单因素最佳值组成的联合组合，以及全部已测配置中选模分数最高的候选。重复配置类别不重复绘制。所有点均为seed=2026的真实验证结果；该图用于检查参数交互，不表征跨seed稳定性。', '',
        '## 复现与文件', '', '```bash', 'conda run -n CPMCM python "problem2_retrain_v2 copy/超参敏感性实验.py"', '```', '',
        '默认只用seed=2026。对齐cuda:2、未对齐cuda:3，每种布局两个独立进程并行处理不同参数；`--workers`可调整并发，`--aligned-device`与`--unaligned-device`可改设备。重复执行检查配置后跳过完成项；`--report-only`仅重画图，`--joint-only`只补联合复核。', '',
        '全部结果见[all_validation_runs.csv](sensitivity_results/all_validation_runs.csv)，配置建议见[recommendation.json](sensitivity_results/recommendation.json)。各运行目录含config、history、summary及验证预测。检查点在`AAAmodel/checkpoints/sensitivity_current`。', '',
        '正式模型不自动替换。本次仅确定单seed验证依据；方案固定后再用多seed评估波动。此前被停止的多seed扫描已单独归档，不纳入本报告。', '']
    (OUT/'recommendation.json').write_text(json.dumps(recommendation,ensure_ascii=False,indent=2))
    (HERE/'超参数敏感性分析.md').write_text('\n'.join(lines))
    print(json.dumps(recommendation,ensure_ascii=False,indent=2))


if __name__=='__main__':
    report()
