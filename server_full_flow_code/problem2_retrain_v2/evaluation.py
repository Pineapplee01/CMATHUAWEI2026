import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
from shared.plotting import pyplot


def metrics(frame):
    y, pred = frame['true_intensity'].to_numpy(), frame['intensity'].to_numpy()
    c, cp = frame['true_class'].to_numpy(), frame['class'].to_numpy()
    precision,recall,f1,_=precision_recall_fscore_support(c,cp,labels=[0,1,2],zero_division=0)
    corr = float(np.corrcoef(y, pred)[0, 1]) if len(y) > 1 and y.std() > 0 and pred.std() > 0 else None
    return {'accuracy': float(accuracy_score(c, cp)),
            'macro_f1': float(f1_score(c, cp, labels=[0, 1, 2], average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(c, cp, labels=[0, 1, 2], average='weighted', zero_division=0)),
            'neutral_precision': float(precision[1]), 'neutral_recall': float(recall[1]),
            'neutral_f1': float(f1[1]),
            'mae': float(np.abs(y-pred).mean()), 'pearson': corr,
            'head_disagreement': float((cp != np.sign(pred).astype(int) + 1).mean())}


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
