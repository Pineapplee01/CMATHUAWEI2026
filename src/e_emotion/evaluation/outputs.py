"""Frozen model-output rules, deliberately separate from metric calculation."""
from dataclasses import dataclass

import numpy as np

from e_emotion.contracts import Polarity
from e_emotion.evaluation.metrics import f1_macro, vector

PROTOCOL_VERSION = "e-competition-v1"


def intensities(values, name="intensity") -> np.ndarray:
    result = vector(values, name)
    if np.any((result < -3) | (result > 3)):
        raise ValueError(f"{name} outside allowed range [-3, 3]; adapt before scoring")
    return result


def classes(values) -> np.ndarray:
    values = np.asarray(values, dtype=object)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("polarity must be a nonempty one-dimensional array")
    # Numeric class IDs or the canonical English names; never truncate fractional IDs.
    return np.array([Polarity.from_value(value).class_id for value in values], dtype=int)


def polarity(scores, threshold=0.0) -> np.ndarray:
    values = intensities(scores)
    if not np.isfinite(threshold) or not 0 <= threshold <= 3:
        raise ValueError("neutral threshold must be finite and in [0, 3]")
    return np.where(values < -threshold, 0, np.where(values > threshold, 2, 1))


@dataclass(frozen=True)
class AdaptedOutput:
    raw_intensity: np.ndarray
    intensity: np.ndarray
    polarity: np.ndarray
    metadata: dict


def adapt_output(raw_intensity, *, scale=1.0, offset=0.0, threshold=0.0,
                 predicted_polarity=None) -> AdaptedOutput:
    """Inverse units first, then fixed clip. Parameters must be frozen before evaluation."""
    raw = vector(raw_intensity, "raw_intensity").copy()
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(offset):
        raise ValueError("inverse scale must be positive finite; offset must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        restored = raw * scale + offset
    vector(restored, "restored_intensity")  # Do not hide overflow with clipping.
    final = np.clip(restored, -3, 3)
    derived = polarity(final, threshold)
    predicted = derived if predicted_polarity is None else classes(predicted_polarity)
    if predicted.shape != final.shape:
        raise ValueError("polarity and intensity must have the same shape")
    for array in (raw, final, predicted):
        array.setflags(write=False)
    return AdaptedOutput(raw, final, predicted, {
        "protocol_version": PROTOCOL_VERSION, "inverse_scale": float(scale),
        "inverse_offset": float(offset), "clip": [-3, 3],
        "polarity_rule": "symmetric_neutral_interval" if predicted_polarity is None else "classification_head",
        "neutral_threshold": float(threshold) if predicted_polarity is None else None,
        "clipped_count": int(np.count_nonzero(restored != final)),
    })


def select_neutral_threshold(truth, prediction, *, split: str) -> float:
    """Validation-only team convention: macro-F1, grid 0:.01:.5, smallest tie."""
    if split != "valid":
        raise ValueError("neutral threshold selection is restricted to attachment2 valid")
    actual = polarity(truth, 0)
    final = intensities(prediction)
    if actual.shape != final.shape:
        raise ValueError("truth and prediction must have the same shape")
    grid = np.linspace(0, .5, 51)
    return float(max(grid, key=lambda t: f1_macro(actual, polarity(final, t))))
