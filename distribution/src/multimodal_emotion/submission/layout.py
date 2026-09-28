"""Canonical project layout names and legacy path mapping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_NAME = "multimodal_emotion"


@dataclass(frozen=True)
class SubmissionLayout:
    """Locations used by the standalone distribution."""

    root: Path

    @property
    def src(self) -> Path:
        return self.root / "src" / "multimodal_emotion"

    @property
    def docs(self) -> Path:
        return self.root / "docs"

    @property
    def reference(self) -> Path:
        return self.root / "reference"


LEGACY_PATHS = {
    "AAA提交版代码及结果/问题一/代码": "src/multimodal_emotion/preprocess/problem1",
    "AAA提交版代码及结果/问题二/代码": "src/multimodal_emotion/prediction",
    "AAA提交版代码及结果/问题三/代码": "src/multimodal_emotion/explanation",
    "preprocess/problem2": "src/multimodal_emotion/preprocess",
    "preprocess/problem3": "src/multimodal_emotion/preprocess",
}
