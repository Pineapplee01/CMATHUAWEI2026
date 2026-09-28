"""特征标准化：先定 Mask，再用有效位估参，最后无效位置重新置零。

适用于文本 BERT (N,L,768)、音频、视觉连续特征。
"""

from __future__ import annotations

import numpy as np


def standardize_features(
    features: np.ndarray,
    valid_mask: np.ndarray,
    support_mask: np.ndarray | None = None,
    eps: float = 1e-6,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """对 (N,L,D) 特征做标准化。

    规则：
    - 有效位置 = valid_mask==1，且若给了 support_mask 则还需 support==1；
    - 用有效位置估计 mean/std → 标准化 → 无效或无支持位置重新置零。
    """
    x = np.asarray(features, dtype=np.float32).copy()
    if x.ndim != 3:
        raise ValueError(f"features 应为 (N,L,D)，实际 {x.shape}")
    vm = np.asarray(valid_mask, dtype=np.int8)
    if support_mask is not None:
        sm = np.asarray(support_mask, dtype=np.int8)
        usable = (vm == 1) & (sm == 1)
    else:
        usable = vm == 1
        sm = None

    flat = x[usable]
    if flat.size == 0:
        stats = {
            "mean": np.zeros(x.shape[-1], dtype=np.float32),
            "std": np.ones(x.shape[-1], dtype=np.float32),
        }
        x[:] = 0.0
        return x, stats

    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std = np.maximum(std, eps)
    x = (x - mean) / std
    bad = ~usable
    x[bad] = 0.0
    if sm is not None:
        x[sm == 0] = 0.0
    return x.astype(np.float32), {"mean": mean, "std": std}
