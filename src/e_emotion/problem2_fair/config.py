"""Fixed server locations for the Problem 2 fair-comparison protocol."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from e_emotion.baselines.workspace import artifact_root


_DATA_ROOT = PurePosixPath("/user_home/gaojianan/CPMCM/AAAdata/Appendix_2/标准化")
_VIEW_ROOTS = {
    "aligned_po": _DATA_ROOT / "对齐版本_retrain_20260925" / "processed_po",
    "unaligned_po": _DATA_ROOT / "未对齐版本" / "processed_po",
}
FAIR_SEEDS = (1, 2, 3)


def default_view_root(view: str) -> PurePosixPath:
    try:
        return _VIEW_ROOTS[view]
    except KeyError as exc:
        raise ValueError(f"unknown Problem 2 data view: {view!r}") from exc


def default_manifest_path(view: str) -> Path:
    default_view_root(view)
    return artifact_root() / "_manifests" / f"{view}_q2_v2.json"


__all__ = ["FAIR_SEEDS", "default_manifest_path", "default_view_root"]
