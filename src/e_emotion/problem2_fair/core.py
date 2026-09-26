"""Problem 2 fairness contracts shared by the main method and baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from e_emotion.contracts import Polarity


MODALITIES = ("text", "audio", "vision")
_CODES = {"T": "text", "A": "audio", "V": "vision"}
Q2_V2_CONDITIONS = (("complete", None, 0.0),) + tuple(
    (combination, position, fraction)
    for combination in ("T", "A", "V", "TA", "TV", "AV", "TAV")
    for position in ("beginning", "middle", "end")
    for fraction in (0.1, 0.3, 0.5)
)


def validate_model_input_view(source_view: str, model_input_view: str) -> None:
    """Ensure a recorded model layout is a valid derivative of its source view."""
    valid = {"aligned_po", "unaligned_po", "unaligned_windowed"}
    if source_view not in {"aligned_po", "unaligned_po"}:
        raise ValueError(f"unknown source view: {source_view!r}")
    if model_input_view not in valid:
        raise ValueError(f"missing or unknown model input view: {model_input_view!r}")
    allowed = {"aligned_po"} if source_view == "aligned_po" else {"unaligned_po", "unaligned_windowed"}
    if model_input_view not in allowed:
        raise ValueError(
            f"model input view {model_input_view!r} is incompatible with source view {source_view!r}"
        )


@dataclass(frozen=True)
class Problem2Split:
    """One processed_po split retaining physical support and observation masks."""

    ids: np.ndarray
    values: Mapping[str, np.ndarray]
    physical_support: Mapping[str, np.ndarray]
    observed: Mapping[str, np.ndarray]
    input_ids: np.ndarray
    regression: np.ndarray
    classification: np.ndarray
    view: str
    split: str

    def __post_init__(self) -> None:
        size = len(self.ids)
        if self.view not in {"aligned_po", "unaligned_po", "unaligned_windowed"}:
            raise ValueError(f"unknown problem2 view: {self.view!r}")
        if self.split not in {"train", "valid", "test"}:
            raise ValueError(f"unknown split: {self.split!r}")
        if self.ids.shape != (size,) or len(set(self.ids.astype(str).tolist())) != size:
            raise ValueError("ids must be one-dimensional and unique")
        if self.input_ids.shape != (size, 50):
            raise ValueError("input_ids must have shape [N, 50]")
        if self.regression.shape != (size,) or self.classification.shape != (size,):
            raise ValueError("labels must have one value per sample")
        if not np.array_equal(self.classification, np.sign(self.regression).astype(np.int64) + 1):
            raise ValueError("classification must match regression polarity")
        for name in MODALITIES:
            value = np.asarray(self.values[name])
            support = np.asarray(self.physical_support[name])
            observed = np.asarray(self.observed[name])
            if value.ndim != 3 or value.shape[0] != size:
                raise ValueError(f"{name} values must be [N, time, feature]")
            if support.shape != value.shape[:2] or observed.shape != value.shape[:2]:
                raise ValueError(f"{name} P/O masks must match values")
            if support.dtype != bool or observed.dtype != bool or np.any(observed & ~support):
                raise ValueError(f"{name} observation mask must be a subset of physical support")

    @property
    def size(self) -> int:
        return len(self.ids)


@dataclass(frozen=True)
class FinalPrediction:
    raw_intensity: np.ndarray
    intensity: np.ndarray
    polarity: tuple[Polarity, ...]


def _selected(combination: str) -> tuple[str, ...]:
    if combination == "complete":
        return ()
    try:
        return tuple(_CODES[code] for code in combination)
    except KeyError as exc:
        raise ValueError(f"unknown modality combination: {combination!r}") from exc


def _window_start(length: int, fraction: float, position: str) -> tuple[int, int]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("missing fraction must be in (0, 1]")
    width = min(length, max(1, int(round(length * fraction))))
    if position == "beginning":
        return 0, width
    if position == "end":
        return length - width, length
    if position == "middle":
        start = (length - width) // 2
        return start, start + width
    raise ValueError(f"unknown Q2 position: {position!r}")


def _raw_window(length: int, start_50: int, end_50: int) -> tuple[int, int]:
    start = int(np.floor(start_50 * length / 50))
    end = int(np.ceil(end_50 * length / 50))
    return max(0, start), min(length, max(start + 1, end))


def project_intensity(raw_intensity, polarity: tuple[Polarity, ...], *, epsilon: float = 1e-6) -> np.ndarray:
    """Project raw intensity into the feasible range of a frozen predicted polarity."""
    raw = np.asarray(raw_intensity, dtype=np.float64)
    if raw.ndim != 1 or len(raw) != len(polarity) or not np.isfinite(raw).all():
        raise ValueError("raw intensity and polarity must be finite matching one-dimensional values")
    if not 0.0 < epsilon < 3.0:
        raise ValueError("epsilon must lie in (0, 3)")
    labels = tuple(Polarity.from_value(value) for value in polarity)
    class_ids = np.asarray([label.class_id for label in labels], dtype=np.int64)
    intensity = np.where(
        class_ids == Polarity.NEGATIVE.class_id,
        np.clip(raw, -3.0, -epsilon),
        np.where(class_ids == Polarity.NEUTRAL.class_id, 0.0, np.clip(raw, epsilon, 3.0)),
    )
    return intensity


def finalize_prediction(raw_intensity, polarity: tuple[Polarity, ...], *, epsilon: float = 1e-6) -> FinalPrediction:
    """Keep raw predictions while deriving the protocol-compliant final output."""
    raw = np.asarray(raw_intensity, dtype=np.float64)
    labels = tuple(Polarity.from_value(value) for value in polarity)
    intensity = project_intensity(raw, labels, epsilon=epsilon)
    return FinalPrediction(raw_intensity=raw, intensity=intensity, polarity=labels)


__all__ = [
    "FinalPrediction",
    "MODALITIES",
    "Problem2Split",
    "Q2_V2_CONDITIONS",
    "finalize_prediction",
    "project_intensity",
    "validate_model_input_view",
]
