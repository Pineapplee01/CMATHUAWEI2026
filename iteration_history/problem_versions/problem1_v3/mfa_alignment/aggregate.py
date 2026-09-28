"""区间交叠聚合与前 50 词视图。来源：archives/problem1_original/problem1/alignment.py。"""
from __future__ import annotations

import numpy as np


def aggregate(target, source, values, quality):
    """交叠时长×质量加权；零分母置零并返回可用掩码及源索引。"""
    overlap = np.maximum(
        0,
        np.minimum(target[:, None, 1], source[None, :, 1])
        - np.maximum(target[:, None, 0], source[None, :, 0]),
    )
    weights = overlap * quality[None, :]
    denom = weights.sum(1)
    features = (weights @ values) / np.maximum(denom[:, None], 1e-12)
    mapping = [
        {
            "indices": np.flatnonzero(w > 0).tolist(),
            "weights": (w[w > 0] / max(w.sum(), 1e-12)).tolist(),
        }
        for w in weights
    ]
    return features.astype("float32"), denom > 0, mapping


def view50(intervals, features, masks):
    """保留前 50 个词位；短序列右侧补零。槽位对应词，不是 [CLS]/[SEP]。"""
    count = min(len(intervals), 50)
    observed = np.zeros((3, 50), dtype=bool)
    observed[:, :count] = masks[:, :count]
    times = np.zeros((50, 2), dtype="float32")
    times[:count] = intervals[:count]
    view = {}
    for m, name in enumerate(("text", "audio", "vision")):
        result = np.zeros((50, features[name].shape[1]), dtype="float32")
        result[:count] = features[name][:count] * masks[m, :count, None]
        view[name] = result
    return view, observed, times, [[i] for i in range(count)]


def view50_metadata(length):
    return {
        "view50_policy": "truncate_prefix_50",
        "view50_index_map": [[i] for i in range(min(length, 50))],
        "view50_original_units": length,
        "view50_retained_units": min(length, 50),
        "view50_dropped_units": max(length - 50, 0),
        "view50_truncated": length > 50,
        "full_length_archive_retained": True,
        "special_tokens_in_alignment": False,
    }


def gaps(intervals, duration):
    result, cursor = [], 0.0
    for a, b in intervals:
        if a > cursor:
            result.append([cursor, float(a)])
        cursor = max(cursor, float(b))
    if cursor < duration:
        result.append([cursor, duration])
    return result
