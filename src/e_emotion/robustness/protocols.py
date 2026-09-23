"""Problem 2 extension points."""

from __future__ import annotations

from typing import Protocol

from e_emotion.contracts import MultimodalBatch, PredictionRecord


class MissingnessSimulator(Protocol):
    def apply(self, batch: MultimodalBatch, seed: int) -> MultimodalBatch: ...


class RobustPredictor(Protocol):
    def predict(self, batch: MultimodalBatch) -> tuple[PredictionRecord, ...]: ...
