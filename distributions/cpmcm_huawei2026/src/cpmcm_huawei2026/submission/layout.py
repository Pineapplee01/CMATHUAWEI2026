"""Canonical project layout names and legacy path mapping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_NAME = "cpmcm_huawei2026"


@dataclass(frozen=True)
class SubmissionLayout:
    """Locations used by the standalone distribution."""

    root: Path

    @property
    def src(self) -> Path:
        return self.root / "src" / "cpmcm_huawei2026"

    @property
    def docs(self) -> Path:
        return self.root / "docs"

    @property
    def external_assets(self) -> Path:
        return self.root / "external_assets"


LEGACY_PATHS = {
    "AAA提交版代码及结果/问题一/代码": "src/cpmcm_huawei2026/preprocessing/problem1",
    "AAA提交版代码及结果/问题二/代码": "src/cpmcm_huawei2026/prediction",
    "AAA提交版代码及结果/问题三/代码": "src/cpmcm_huawei2026/explanation",
    "preprocess/problem2": "src/cpmcm_huawei2026/preprocessing",
    "preprocess/problem3": "src/cpmcm_huawei2026/preprocessing",
}
