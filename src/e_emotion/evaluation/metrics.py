"""Dependency-light competition metrics."""

from __future__ import annotations

import numpy as np


def _arrays(truth: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(truth).reshape(-1)
    predicted = np.asarray(prediction).reshape(-1)
    if actual.shape != predicted.shape:
        raise ValueError("truth and prediction must have the same shape")
    if actual.size == 0:
        raise ValueError("metrics require at least one sample")
    return actual, predicted


def accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
    actual, predicted = _arrays(truth, prediction)
    return float(np.mean(actual == predicted))


def f1_macro(truth: np.ndarray, prediction: np.ndarray, labels: tuple[int, ...] = (0, 1, 2)) -> float:
    actual, predicted = _arrays(truth, prediction)
    scores: list[float] = []
    for label in labels:
        true_positive = float(np.sum((actual == label) & (predicted == label)))
        false_positive = float(np.sum((actual != label) & (predicted == label)))
        false_negative = float(np.sum((actual == label) & (predicted != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def mae(truth: np.ndarray, prediction: np.ndarray) -> float:
    actual, predicted = _arrays(truth, prediction)
    return float(np.mean(np.abs(actual.astype(float) - predicted.astype(float))))


def pearson(truth: np.ndarray, prediction: np.ndarray) -> float:
    actual, predicted = _arrays(truth, prediction)
    actual = actual.astype(float)
    predicted = predicted.astype(float)
    actual_centered = actual - np.mean(actual)
    predicted_centered = predicted - np.mean(predicted)
    denominator = np.linalg.norm(actual_centered) * np.linalg.norm(predicted_centered)
    if denominator == 0:
        return 0.0
    return float(np.dot(actual_centered, predicted_centered) / denominator)
