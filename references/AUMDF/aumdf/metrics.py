"""Paper-style regression/binary metrics plus E题 three-class metrics."""
import numpy as np


def polarity(scores, threshold=0.0):
    scores = np.asarray(scores)
    return np.where(scores < -threshold, 0, np.where(scores > threshold, 2, 1))


def f1_scores(truth, prediction, labels):
    support, f1 = [], []
    for label in labels:
        tp = np.sum((truth == label) & (prediction == label))
        fp = np.sum((truth != label) & (prediction == label))
        fn = np.sum((truth == label) & (prediction != label))
        denominator = 2*tp + fp + fn
        f1.append(2*tp/denominator if denominator else 0.0)
        support.append(np.sum(truth == label))
    return float(np.mean(f1)), float(np.average(f1, weights=support)) if sum(support) else 0.0


def select_neutral_threshold(truth, predictions):
    classes = polarity(np.asarray(truth), 0)
    grid = np.linspace(0, 0.5, 51)
    # Ties choose the smaller threshold; validation-only calibration.
    return float(max(grid, key=lambda t: f1_scores(classes, polarity(predictions, t), (0,1,2))[0]))


def metrics(truth, prediction, threshold=0.0):
    y, p = np.asarray(truth).reshape(-1), np.asarray(prediction).reshape(-1)
    if y.size == 0 or y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("nonempty, finite, equally sized predictions and labels required")
    p = np.clip(p, -3, 3)
    actual, predicted = polarity(y), polarity(p, threshold)
    macro, weighted = f1_scores(actual, predicted, (0,1,2))
    nonzero = y != 0
    result = {
        "n": int(y.size), "mae": float(np.abs(y-p).mean()),
        "pearson": float(np.corrcoef(y,p)[0,1]) if y.size > 1 and y.std() > 1e-12 and p.std() > 1e-12 else None,
        "accuracy_3": float((actual == predicted).mean()), "macro_f1_3": macro, "weighted_f1_3": weighted,
        "accuracy_7": float((np.round(y).clip(-3,3) == np.round(p)).mean()),
        "neutral_threshold": float(threshold),
        "accuracy_2_nonzero": float(((y[nonzero]>0) == (p[nonzero]>0)).mean()) if nonzero.any() else None,
        "weighted_f1_2_nonzero": f1_scores(y[nonzero]>0, p[nonzero]>0, (False,True))[1] if nonzero.any() else None,
    }
    return result
