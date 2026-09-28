"""COVAREP 74-d audio feature extraction via MATLAB."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

COVAREP_NAMES = [
    "F0",
    "VUV",
    "NAQ",
    "QOQ",
    "H1H2",
    "PSP",
    "MDQ",
    "peakSlope",
    "Rd",
    "Rd_conf",
    "creak",
    *[f"MCEP_{i}" for i in range(25)],
    *[f"HMPDM_{i}" for i in range(25)],
    *[f"HMPDD_{i}" for i in range(13)],
]
assert len(COVAREP_NAMES) == 74


def run_covarep_dir(cfg: dict[str, Any], wav_dir: Path) -> None:
    """Run COVAREP on all *.wav in wav_dir; writes sibling .mat files."""
    wavs = list(wav_dir.glob("*.wav"))
    if not wavs:
        return
    matlab = Path(cfg["paths"]["matlab"])
    covarep = Path(cfg["paths"]["covarep"])
    hop = float(cfg["audio"].get("covarep_hop_sec", 0.01))
    q1 = Path(__file__).resolve().parent
    cmd = [
        str(matlab),
        "-batch",
        (
            f"addpath('{q1.as_posix()}'); "
            f"covarep_batch('{covarep.as_posix()}', '{wav_dir.as_posix()}', {hop});"
        ),
    ]
    subprocess.run(cmd, check=True, cwd=str(q1))


def load_covarep_mat(mat_path: Path, hop_sec: float = 0.01) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load COVAREP .mat → (intervals[T,2] sec, values[T,74], names)."""
    if not mat_path.exists():
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 74), dtype=np.float32), list(COVAREP_NAMES)
    data = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    feat = np.asarray(data["features"], dtype=np.float64)
    if feat.ndim == 1:
        feat = feat.reshape(1, -1)
    if feat.shape[1] != 74:
        raise ValueError(f"COVAREP 维数应为 74，实际 {feat.shape}: {mat_path}")
    names = list(COVAREP_NAMES)
    if "names" in data:
        raw = data["names"]
        try:
            parsed = [str(x) for x in np.atleast_1d(raw).tolist()]
            if len(parsed) == 74:
                names = parsed
        except Exception:
            pass
    t = (np.arange(len(feat), dtype=np.float64) + 0.5) * hop_sec
    half = hop_sec * 0.5
    intervals = np.stack([t - half, t + half], axis=1)
    values = np.nan_to_num(feat.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return intervals, values, names


def aggregate_to_windows(windows: np.ndarray, intervals: np.ndarray, values: np.ndarray, dim: int = 74) -> np.ndarray:
    """Overlap-weighted mean into equal-time windows → (L, dim)."""
    L = len(windows)
    out = np.zeros((L, dim), dtype=np.float32)
    if values.size == 0:
        return out
    for k, (ws, we) in enumerate(windows):
        overlap = np.maximum(0.0, np.minimum(we, intervals[:, 1]) - np.maximum(ws, intervals[:, 0]))
        wsum = float(overlap.sum())
        if wsum <= 0:
            continue
        out[k] = (values * overlap[:, None]).sum(axis=0) / wsum
    return out
