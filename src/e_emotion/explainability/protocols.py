"""Problem 3 extension points."""

from __future__ import annotations

from typing import Protocol

from e_emotion.contracts import EvidenceRecord, MultimodalBatch, PredictionRecord


class ExplainablePredictor(Protocol):
    def predict_with_evidence(
        self, batch: MultimodalBatch
    ) -> tuple[tuple[PredictionRecord, ...], tuple[EvidenceRecord, ...]]: ...


class EvidenceMapper(Protocol):
    def map(self, evidence: EvidenceRecord) -> EvidenceRecord: ...
