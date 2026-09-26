"""Declared MulT input views and isolated method-level artifact roots."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


RAW_UNALIGNED_ROOT = Path(
    "/user_home/gaojianan/CPMCM/AAAdata/Appendix_2/原始未预处理/未对齐版本"
)
ALIGNED_50_ROOT = Path(
    "/user_home/gaojianan/CPMCM/AAAdata/Appendix_2/标准化/对齐版本/processed"
)


@dataclass(frozen=True)
class DataView:
    name: str
    data_root: Path
    canonical: bool
    feature_version: str


def resolve_view(name: str) -> DataView:
    views = {
        "raw_unaligned": DataView("raw_unaligned", RAW_UNALIGNED_ROOT, False, "raw_unaligned"),
        "aligned_50_control": DataView("aligned_50_control", ALIGNED_50_ROOT, True, "aligned_50"),
    }
    try:
        return views[name]
    except KeyError as error:
        raise ValueError(f"unknown MulT data view: {name}") from error


def run_root(name: str) -> Path:
    return Path("runs") / resolve_view(name).name
