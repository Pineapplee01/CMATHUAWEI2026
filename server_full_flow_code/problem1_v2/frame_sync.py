"""音视频统一抽帧：重采样到共同 hop 网格。

原生：COVAREP ≈10ms，OpenFace ≈视频帧周期（常 ~33ms）。
统一后：双方都落到 [0,D) 上步长 common_hop_sec 的等间隔帧，
同下标对应同一时间片（再去做 50 窗 / MFA 词聚合或 v4 直接使用）。
"""
from __future__ import annotations

from typing import Any

import numpy as np


def common_hop_sec(cfg: dict[str, Any], default: float = 0.04) -> float:
    sync = cfg.get("av_sync") or {}
    hop = float(sync.get("common_hop_sec", cfg.get("common_hop_sec", default)))
    if hop <= 0:
        raise ValueError(f"common_hop_sec 必须为正，得到 {hop}")
    return hop


def uniform_frame_grid(duration: float, hop: float) -> np.ndarray:
    d = max(float(duration), 0.0)
    h = float(hop)
    if d <= 0 or h <= 0:
        return np.zeros((0, 2), dtype=np.float64)
    n = int(np.ceil(d / h - 1e-12))
    n = max(n, 0)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    starts = np.arange(n, dtype=np.float64) * h
    ends = np.minimum(starts + h, d)
    keep = ends > starts + 1e-12
    return np.stack([starts[keep], ends[keep]], axis=1)


def resample_to_grid(
    target: np.ndarray,
    source: np.ndarray,
    values: np.ndarray,
    quality: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """按区间重叠时长把 source 帧加权平均到 target 网格。"""
    target = np.asarray(target, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    values = np.asarray(values, dtype=np.float32)
    t = len(target)
    if values.ndim == 2 and len(values):
        dim = int(values.shape[1])
    elif values.size:
        dim = int(values.shape[-1])
    else:
        dim = 0
    if t == 0:
        return (
            np.zeros((0, max(dim, 1)), dtype=np.float32),
            np.zeros(0, dtype=bool),
        )
    if dim <= 0 or len(source) == 0 or len(values) == 0:
        return (
            np.zeros((t, max(dim, 1)), dtype=np.float32),
            np.zeros(t, dtype=bool),
        )
    if quality is None:
        quality = np.ones(len(source), dtype=np.float64)
    else:
        quality = np.asarray(quality, dtype=np.float64).reshape(-1)
        if len(quality) != len(source):
            raise ValueError("quality 长度须与 source 帧数一致")
    overlap = np.maximum(
        0.0,
        np.minimum(target[:, None, 1], source[None, :, 1])
        - np.maximum(target[:, None, 0], source[None, :, 0]),
    )
    weights = overlap * quality[None, :]
    denom = weights.sum(axis=1)
    feat = weights @ values.astype(np.float64) / np.maximum(denom[:, None], 1e-12)
    usable = denom > 1e-12
    feat = np.nan_to_num(feat.astype(np.float32), nan=0.0)
    feat[~usable] = 0.0
    return feat, usable


def unify_modality(
    duration: float,
    hop: float,
    source_intervals: np.ndarray,
    values: np.ndarray,
    quality: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = uniform_frame_grid(duration, hop)
    feat, usable = resample_to_grid(grid, source_intervals, values, quality)
    return grid, feat, usable
