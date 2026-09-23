"""Dependency-light competition metrics."""

from __future__ import annotations

import numpy as np


def vector(values, name="values") -> np.ndarray:
    """Reject malformed batches instead of silently flattening or dropping rows."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite one-dimensional array")
    return array


def _arrays(truth: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    actual = vector(truth, "truth")
    predicted = vector(prediction, "prediction")
    if actual.shape != predicted.shape:
        raise ValueError("truth and prediction must have the same shape")
    return actual, predicted


def accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
    actual, predicted = _arrays(truth, prediction)
    return float(np.mean(actual == predicted))


def f1_macro(truth: np.ndarray, prediction: np.ndarray, labels: tuple[int, ...] = (0, 1, 2)) -> float:
    return f1_scores(truth, prediction, labels)[0]


def f1_scores(truth, prediction, labels=(0, 1, 2)) -> tuple[float, float]:
    actual, predicted = _arrays(truth, prediction)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("labels must be nonempty and unique")
    if not np.isin(actual, labels).all() or not np.isin(predicted, labels).all():
        raise ValueError("classes must be members of labels")
    scores: list[float] = []
    support = []
    for label in labels:
        true_positive = float(np.sum((actual == label) & (predicted == label)))
        false_positive = float(np.sum((actual != label) & (predicted == label)))
        false_negative = float(np.sum((actual == label) & (predicted != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
        support.append(int(np.sum(actual == label)))
    return float(np.mean(scores)), float(np.average(scores, weights=support))


def mae(truth: np.ndarray, prediction: np.ndarray) -> float:
    actual, predicted = _arrays(truth, prediction)
    return float(np.mean(np.abs(actual.astype(float) - predicted.astype(float))))


def pearson(truth: np.ndarray, prediction: np.ndarray) -> float | None:
    actual, predicted = _arrays(truth, prediction)
    # Exact constants can acquire tiny residuals when their floating-point mean is rounded.
    if actual.size < 2 or np.all(actual == actual[0]) or np.all(predicted == predicted[0]):
        return None
    unit_vectors = []
    for values in (actual, predicted):
        # Correlation is shift/scale invariant. These local arithmetic copies avoid
        # squared-norm underflow and cancellation at nearly constant float values.
        # Power-of-two scaling preserves representable near-constant spacing;
        # dividing by an arbitrary maximum can quantize that spacing.
        _, exponent = np.frexp(np.max(np.abs(values)))
        scaled = np.ldexp(values, -int(exponent))
        shifted = scaled - scaled[0]
        centered = shifted - np.mean(shifted)
        norm = np.linalg.norm(centered)
        if norm == 0:
            return None
        unit_vectors.append(centered / norm)
    correlation = float(np.dot(*unit_vectors))
    return min(1.0, max(-1.0, correlation))  # Guard only metric roundoff, not predictions.
