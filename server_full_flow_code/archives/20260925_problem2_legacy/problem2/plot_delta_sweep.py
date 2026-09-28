"""δ 超参扫描曲线：训练区间重标注的中性带宽 vs valid/test 准确率。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from shared.common import path
from shared.plotting import pyplot


def main():
    points = []
    for run, delta, train_moved, valid_risk, test_risk, series in [
            ('relabel_d017', 0.17, 2, 1, 0, 'hard'),
            ('relabel_interval', 0.35, 689, 147, 130, 'hard'),
            ('relabel_d067', 0.67, 1231, 265, 230, 'hard'),
            ('soft_g010', 0.10, 0, 1, 0, 'soft'),
            ('soft_g017', 0.17, 2, 1, 0, 'soft'),
            ('soft_gauss', 0.42, 0, 147, 130, 'soft')]:
        h = pd.read_csv(path('problem2/results') / run / 'training_history.csv')
        best = h.loc[h.full_accuracy.idxmax(), 'full_accuracy']
        points.append({'delta': delta, 'valid': best, 'train_moved': train_moved,
                       'valid_risk': valid_risk, 'test_risk': test_risk, 'series': series})
    df = pd.DataFrame(points)
    df.to_csv(path('problem2/results') / 'delta_sweep.csv', index=False)

    plt = pyplot()
    plt.rcParams['font.sans-serif'] = ['AR PL UMing CN', 'DejaVu Sans']
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhline(0.6827, color='gray', lw=1.2, ls='--')
    ax.text(0.45, 0.6855, '基线（无区间化，68.27%）', fontsize=9, color='gray')
    hard = df[df.series == 'hard']; soft = df[df.series == 'soft']
    ax.plot(hard.delta, hard.valid, 'o-', color='#c0392b', lw=1.6, ms=6,
            label='硬区间重标注 δ')
    ax.plot(soft.delta, soft.valid, '^-', color='#2980b9', lw=1.6, ms=6,
            label='高斯软标签 γ')
    for _, r in df.iterrows():
        ax.annotate(f'{r.valid:.2%}', (r.delta, r.valid),
                    textcoords='offset points', xytext=(6, -12), fontsize=8.5)
    ax.set_xlabel('区间带宽参数（硬重标注 δ / 高斯软标签 γ；数据粒度 1/6）')
    ax.set_ylabel('valid 完整观测 Accuracy')
    ax.set_title('中性区间化超参扫描：两族均单调收敛于基线，无更优解')
    ax.legend(fontsize=9, loc='lower left')
    ax.set_ylim(0.48, 0.71)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(path('problem2/figures') / f'delta_sweep.{ext}')
    print(df.to_string(index=False))
    print('figure -> problem2/figures/delta_sweep.png')


if __name__ == '__main__':
    main()
