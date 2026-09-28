import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from shared.plotting import pyplot


def metrics(frame):
    y, pred = frame['true_intensity'].to_numpy(), frame['intensity'].to_numpy()
    c, cp = frame['true_class'].to_numpy(), frame['class'].to_numpy()
    corr = float(np.corrcoef(y, pred)[0, 1]) if len(y) > 1 and y.std() > 0 and pred.std() > 0 else None
    return {'accuracy': float(accuracy_score(c, cp)),
            'macro_f1': float(f1_score(c, cp, labels=[0, 1, 2], average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(c, cp, labels=[0, 1, 2], average='weighted', zero_division=0)),
            'mae': float(np.abs(y-pred).mean()), 'pearson': corr,
            'head_disagreement': float((cp != np.sign(pred).astype(int) + 1).mean())}


def grouped_bootstrap(frame, repeats=500, seed=2026):
    rng = np.random.default_rng(seed)
    groups = frame['id'].map(lambda x: str(x).split('$_$')[0])
    names = groups.unique()
    by_group = {g: frame[groups == g] for g in names}
    samples = [metrics(pd.concat([by_group[g] for g in rng.choice(names, len(names), replace=True)]))
               for _ in range(repeats)]
    return {k: np.quantile([r[k] for r in samples if r[k] is not None], [0.025, 0.975]).tolist()
            for k in samples[0] if any(r[k] is not None for r in samples)}




def error_plots(frame, directory):
    directory.mkdir(parents=True, exist_ok=True)
    plt = pyplot()
    fig, ax = plt.subplots()
    cm = confusion_matrix(frame.true_class, frame['class'], labels=[0, 1, 2])
    im = ax.imshow(cm, cmap='Blues')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha='center')
    ax.set(xlabel='Predicted class', ylabel='True class', title='Polarity confusion matrix',
           xticks=[0, 1, 2], yticks=[0, 1, 2])
    fig.colorbar(im, ax=ax)
    fig.savefig(directory / 'confusion.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    fig, ax = plt.subplots()
    ax.scatter(frame.true_intensity, frame.intensity - frame.true_intensity, s=8, alpha=.5)
    ax.axhline(0, color='black', linewidth=1)
    ax.set(xlabel='True intensity', ylabel='Prediction residual', title='Intensity residuals')
    fig.savefig(directory / 'residual.png', dpi=300, bbox_inches='tight'); plt.close(fig)


def robustness_plot(frame, directory):
    directory.mkdir(parents=True, exist_ok=True)
    plt = pyplot()
    fig, ax = plt.subplots()
    for name, rows in frame.groupby('modalities'):
        r = rows.groupby('rate')['macro_f1'].mean()
        ax.plot(r.index, r.values, marker='o', label=name)
    ax.set(xlabel='Requested missing fraction of valid positions', ylabel='Macro-F1',
           title='Contiguous missingness robustness')
    ax.legend(fontsize=7)
    fig.savefig(directory / 'missingness.png', dpi=300, bbox_inches='tight'); plt.close(fig)
