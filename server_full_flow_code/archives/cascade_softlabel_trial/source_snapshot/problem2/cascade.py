"""级联中性检测：离线组合已落盘的证据（完成试验，结论为负，保留复核）。

试验结论（2026-09-24，全部以 train 按视频分组5折诚实CV评估，未读test）：
- 主模型 log_neu 单特征 CV-AUC 0.7630，已是全部证据中最强；
- cardiffnlp 先验 prior_neu 0.7141，与 log_neu 组合仅 0.7695，堆叠 TF-IDF
  词元 n-gram（单独 0.635）后 0.7762，增益 +0.013 AUC 无决策价值；
- DeBERTa-v3-large MLM 底座嵌入对中性近随机（0.339 vs 0.330 基率）；
- 未对齐500槽音视频时序统计 AUC 0.551；
- 在该检测器下，硬级联（demote→argmax(负,正)）最优 CV 0.7087 低于
  plain argmax 0.7175；单调偏移规则 k(g-0.5) 同样无增益；
- 12个历史模型全组合等权logit集成在 valid 上最高 0.6854，仅 +0.27 个点，
  在728条样本的噪声内。
故未对test执行级联读数（--freeze 未使用），主模型保持不变。
"""
import argparse
import json
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from shared.common import json_write

RESULTS = 'problem2/results/cascade'


def load(split):
    npz = np.load(f'{RESULTS}/{split}_logits.npz', allow_pickle=True)
    prior = np.load(f'{RESULTS}/{split}_prior.npz', allow_pickle=True)
    assert (npz['ids'] == prior['ids']).all()
    features = np.concatenate([npz['logits'], npz['intensity'][:, None],
                               prior['probs']], 1)
    return features, npz['label'].astype(int), npz['intensity'], npz['ids']


def cascade_predict(features, intensity, detector, eta, logits):
    neutral_prob = detector.predict_proba(features)[:, 1]
    polarity = np.stack([logits[:, 0], np.full(len(logits), -np.inf), logits[:, 2]], 1)
    polarity_class = polarity.argmax(1)
    return np.where(neutral_prob >= eta, 1, polarity_class), neutral_prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--grid', type=float, nargs='+',
                        default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55])
    parser.add_argument('--C', type=float, nargs='+', default=[0.1, 0.3, 1.0, 3.0])
    parser.add_argument('--freeze', action='store_true', help='冻结后读取test一次')
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()

    X, y, intensity, ids = load('train')
    groups = np.asarray([s.split('$_$')[0] for s in ids])
    logits_train = X[:, :3]
    splitter = GroupKFold(n_splits=args.folds)
    folds = list(splitter.split(X, y, groups))

    # 阶段1：在 train 内分组CV挑组合器C；held-out中性概率全部由未见视频的头部产生。
    best = None
    for c in args.C:
        held = np.zeros((len(y), 2))
        for tr, ho in folds:
            head = LogisticRegression(max_iter=5000, C=c, class_weight='balanced') \
                .fit(X[tr], (y[tr] == 1).astype(int))
            held[ho] = head.predict_proba(X[ho])
        for eta in args.grid:
            pred = np.where(held[:, 1] >= eta, 1, logits_train.argmax(1))
            acc = (pred == y).mean()
            recall = ((held[:, 1] >= eta) & (y == 1)).sum() / (y == 1).sum()
            if best is None or acc > best[0]:
                best = (acc, c, eta, recall, held)
    cv_acc, C, eta, cv_recall, held = best
    print(f'train grouped-{args.folds}-fold honest CV: acc={cv_acc:.4f} '
          f'neutral_recall={cv_recall:.3f} (C={C}, eta={eta})')

    # 阶段2：全train拟合组合器；valid 选 eta（并复查C），不读test。
    detector = LogisticRegression(max_iter=5000, C=C, class_weight='balanced') \
        .fit(X, (y == 1).astype(int))

    Xv, yv, iv, idv = load('valid')
    logits_valid = Xv[:, :3]
    base_valid = (logits_valid.argmax(1) == yv).mean()
    pv = detector.predict_proba(Xv)[:, 1]
    rows = []
    for e in args.grid:
        pred = np.where(pv >= e, 1, np.stack(
            [logits_valid[:, 0], np.full(len(logits_valid), -np.inf), logits_valid[:, 2]], 1).argmax(1))
        rows.append({'eta': e, 'valid_accuracy': float((pred == yv).mean()),
                     'neutral_recall': float(((pv >= e) & (yv == 1)).sum() / (yv == 1).sum()),
                     'neutral_precision': float(((pv >= e) & (yv == 1)).sum() / max((pv >= e).sum(), 1))})
    eta_valid = max(rows, key=lambda r: r['valid_accuracy'])
    print(f'valid baseline {base_valid:.4f} -> cascade {eta_valid["valid_accuracy"]:.4f} '
          f'at eta={eta_valid["eta"]} (neutral recall {eta_valid["neutral_recall"]:.3f}, '
          f'precision {eta_valid["neutral_precision"]:.3f})')
    for r in rows:
        print('   ', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    chosen_eta = eta_valid['eta']
    report = {'protocol': {'detector': f'LogisticRegression(C={C}) on [main_logits,intensity,prior_probs]',
                           'detector_fit': 'train, group-KFold inner CV for C; final head on full train',
                           'eta_selection': 'valid grid, accuracy first',
                           'test_reads': 1 if args.freeze else 0},
              'train_cv': {'accuracy': float(cv_acc), 'neutral_recall': float(cv_recall),
                           'C': C, 'eta_grid_best': float(eta), 'folds': args.folds},
              'valid': {'baseline_accuracy': float(base_valid), **eta_valid, 'grid': rows}}
    json_write(f'{RESULTS}/cascade_report.json', report)

    if args.freeze:
        Xt, yt, it, idt = load('test')
        logits_test = Xt[:, :3]
        pt = detector.predict_proba(Xt)[:, 1]
        base_test = (logits_test.argmax(1) == yt).mean()
        pred = np.where(pt >= chosen_eta, 1, np.stack(
            [logits_test[:, 0], np.full(len(logits_test), -np.inf), logits_test[:, 2]], 1).argmax(1))
        acc = (pred == yt).mean()
        recall = ((pt >= chosen_eta) & (yt == 1)).sum() / (yt == 1).sum()
        precision = ((pt >= chosen_eta) & (yt == 1)).sum() / max((pt >= chosen_eta).sum(), 1)
        cm = np.zeros((3, 3), int)
        for t, q in zip(yt, pred):
            cm[t, q] += 1
        print(f'TEST (single frozen read): baseline {base_test:.4f} -> cascade {acc:.4f} '
              f'({int(round(acc * len(yt)))}/{len(yt)}), neutral recall {recall:.3f} precision {precision:.3f}')
        print(cm)
        report['test'] = {'baseline_accuracy': float(base_test), 'cascade_accuracy': float(acc),
                          'neutral_recall': float(recall), 'neutral_precision': float(precision),
                          'confusion': cm.tolist()}
        np.savez_compressed(f'{RESULTS}/test_cascade_predictions.npz',
                            pred=pred, label=yt, neutral_prob=pt, ids=idt)
        json_write(f'{RESULTS}/cascade_report.json', report)


if __name__ == '__main__':
    main()
