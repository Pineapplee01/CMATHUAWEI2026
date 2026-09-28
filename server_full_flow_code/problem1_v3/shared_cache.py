"""问题一跨版本共享 COVAREP / OpenFace 缓存，避免 v2/v3/v4 重复提取。"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # CPMCM/
COVAREP_CACHE = ROOT / "AAAdata" / "processed" / "problem1_covarep"
OPENFACE_CACHE = ROOT / "AAAdata" / "processed" / "problem1_openface"


def _stems(sample_id: str) -> list[str]:
    s = sample_id.replace("/", "_")
    alts = [s, s.replace("$_$", "__"), s.replace("__", "$_$")]
    out: list[str] = []
    seen: set[str] = set()
    for a in alts:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def ensure_caches() -> None:
    COVAREP_CACHE.mkdir(parents=True, exist_ok=True)
    OPENFACE_CACHE.mkdir(parents=True, exist_ok=True)


def covarep_cached_mat(sample_id: str) -> Path | None:
    ensure_caches()
    for stem in _stems(sample_id):
        p = COVAREP_CACHE / f"{stem}.mat"
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def install_covarep_mat(sample_id: str, dest_mat: Path) -> bool:
    src = covarep_cached_mat(sample_id)
    if src is None:
        return False
    dest_mat.parent.mkdir(parents=True, exist_ok=True)
    if dest_mat.resolve() != src.resolve():
        shutil.copy2(src, dest_mat)
    return True


def publish_covarep_mat(sample_id: str, src_mat: Path) -> None:
    if not src_mat.is_file() or src_mat.stat().st_size <= 0:
        return
    ensure_caches()
    stem = _stems(sample_id)[0]
    dst = COVAREP_CACHE / f"{stem}.mat"
    if dst.resolve() != src_mat.resolve():
        shutil.copy2(src_mat, dst)


def openface_cached_csv(sample_id: str, video_stem: str) -> Path | None:
    ensure_caches()
    for stem in _stems(sample_id):
        p = OPENFACE_CACHE / stem / f"{video_stem}.csv"
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def install_openface_csv(sample_id: str, video_stem: str, dest_csv: Path) -> bool:
    src = openface_cached_csv(sample_id, video_stem)
    if src is None:
        return False
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    if dest_csv.resolve() != src.resolve():
        shutil.copy2(src, dest_csv)
    return True


def publish_openface_csv(sample_id: str, video_stem: str, src_csv: Path) -> None:
    if not src_csv.is_file() or src_csv.stat().st_size <= 0:
        return
    ensure_caches()
    stem = _stems(sample_id)[0]
    dst = OPENFACE_CACHE / stem / f"{video_stem}.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.resolve() != src_csv.resolve():
        shutil.copy2(src_csv, dst)
