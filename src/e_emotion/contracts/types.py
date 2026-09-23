"""Domain data contracts used by all three task domains."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Mapping

import numpy as np


class DatasetVariant(str, Enum):
    ALIGNED = "aligned"
    UNALIGNED = "unaligned"


class SplitName(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class Modality(str, Enum):
    TEXT = "text"
    AUDIO = "audio"
    VISION = "vision"


class Polarity(str, Enum):
    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"
    POSITIVE = "Positive"

    @classmethod
    def from_value(cls, value: Any) -> "Polarity":
        if isinstance(value, str):
            normalized = value.strip().lower()
            aliases = {
                "negative": cls.NEGATIVE,
                "neutral": cls.NEUTRAL,
                "positive": cls.POSITIVE,
            }
            if normalized in aliases:
                return aliases[normalized]
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric = float(value)
            if isfinite(numeric) and numeric.is_integer() and int(numeric) in (0, 1, 2):
                return (cls.NEGATIVE, cls.NEUTRAL, cls.POSITIVE)[int(numeric)]
        raise ValueError(f"unsupported polarity value: {value!r}")

    @property
    def class_id(self) -> int:
        return (Polarity.NEGATIVE, Polarity.NEUTRAL, Polarity.POSITIVE).index(self)


@dataclass(frozen=True)
class SampleId:
    video_id: str
    clip_id: str | int

    @property
    def as_string(self) -> str:
        return f"{self.video_id}$_${self.clip_id}"

    def __str__(self) -> str:
        return self.as_string


@dataclass(frozen=True)
class ModalitySequence:
    values: np.ndarray
    valid_length: int
    mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.values, np.ndarray) or self.values.ndim != 2:
            raise ValueError("modality values must be a two-dimensional numpy array")
        if not 0 <= self.valid_length <= self.values.shape[0]:
            raise ValueError("valid_length must be within the sequence length")
        sequence_mask = self.mask
        if sequence_mask is None:
            sequence_mask = np.arange(self.values.shape[0]) < self.valid_length
        sequence_mask = np.asarray(sequence_mask, dtype=bool)
        if sequence_mask.shape != (self.values.shape[0],):
            raise ValueError("mask must have one boolean value per sequence position")
        object.__setattr__(self, "mask", sequence_mask)


@dataclass(frozen=True)
class MultimodalSample:
    sample_id: SampleId
    text: ModalitySequence
    audio: ModalitySequence
    vision: ModalitySequence
    polarity: Polarity | None = None
    intensity: float | None = None
    raw_text: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class MultimodalBatch:
    samples: tuple[MultimodalSample, ...]
    variant: DatasetVariant


@dataclass(frozen=True)
class PredictionRecord:
    sample_id: SampleId
    polarity: Polarity | None
    intensity: float | None
    confidence: float | None = None


@dataclass(frozen=True)
class EvidenceRecord:
    sample_id: SampleId
    modality: Modality
    start: int
    end: int
    contribution: float
    locator: str | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("evidence interval must satisfy 0 <= start < end")


@dataclass(frozen=True)
class FeatureManifest:
    sample_id: SampleId
    variant: DatasetVariant
    valid_lengths: Mapping[str, int]
    feature_dimensions: Mapping[str, int]
    alignment_granularity: str
    tool_versions: Mapping[str, str] = field(default_factory=dict)
    source: str | None = None
