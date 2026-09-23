"""Problem 1 extension points."""

from __future__ import annotations

from typing import Mapping, Protocol

import numpy as np

from e_emotion.contracts import FeatureManifest, MultimodalSample


class FeatureExtractor(Protocol):
    def extract(self, sample: MultimodalSample) -> Mapping[str, np.ndarray]: ...


class TemporalAligner(Protocol):
    def align(self, features: Mapping[str, np.ndarray]) -> FeatureManifest: ...
