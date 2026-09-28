"""MFA 词轴上的源特征：BERT 按词（不含 CLS/SEP 槽）、COVAREP、OpenFace AU35。"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from problem1_v3.covarep_audio import COVAREP_NAMES, load_covarep_mat
from problem1_v3.mfa_alignment.media import command
from problem1_v3.frame_sync import common_hop_sec, unify_modality
from problem1_v3.shared_cache import install_openface_csv, publish_openface_csv

AUS = ("01", "02", "04", "05", "06", "07", "09", "10", "12", "14", "15", "17", "20", "23", "25", "26", "45")
VISION_NAMES = [f"AU{x}_r" for x in AUS] + [f"AU{x}_c" for x in (*AUS[:-1], "28", "45")]
assert len(VISION_NAMES) == 35


def bert_words(words: list[str], tokenizer, encoder, device) -> tuple[np.ndarray, list[dict]]:
    """每个内容词一个 768 维向量；CLS/SEP 仅在编码上下文中，不进入词槽。"""
    counts = [len(tokenizer.tokenize(w)) for w in words]
    groups, start, size = [], 0, 0
    for i, count in enumerate(counts):
        if count > 510:
            raise ValueError("单词子词数超过 BERT 容量")
        if size + count > 510:
            groups.append((start, i))
            start, size = i, 0
        size += count
    groups.append((start, len(words)))
    all_features, mappings = [], []
    with torch.no_grad():
        for lo, hi in groups:
            inputs = tokenizer(
                words[lo:hi], is_split_into_words=True, return_tensors="pt", truncation=False
            )
            ids = inputs.word_ids()
            hidden = encoder(**{k: v.to(device) for k, v in inputs.items()}).last_hidden_state[0].cpu().numpy()
            for word_index in range(hi - lo):
                positions = [j for j, wid in enumerate(ids) if wid == word_index]
                if not positions:
                    raise ValueError("转写词无子词表示")
                all_features.append(hidden[positions].mean(0))
                mappings.append(
                    {
                        "word_index": lo + word_index,
                        "token_positions": positions,
                        "token_ids": inputs["input_ids"][0, positions].tolist(),
                        "context_word_range": [lo, hi],
                    }
                )
    return np.asarray(all_features, dtype=np.float32), mappings


def audio_source_covarep(
    mat_path: Path, hop_sec: float, audio_offset: float = 0.0, duration: float | None = None, cfg: dict[str, Any] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """COVAREP 帧 → 统一 hop 网格 → (intervals, values, quality, names)。"""
    intervals, values, names = load_covarep_mat(mat_path, hop_sec)
    if len(intervals) == 0:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0, 74), dtype=np.float32),
            np.zeros(0, dtype=np.float64),
            names,
        )
    intervals = intervals + float(audio_offset)
    good = np.isfinite(values).all(1).astype(np.float64)
    if duration is not None and cfg is not None:
        hop = common_hop_sec(cfg)
        grid, synced, usable = unify_modality(float(duration), hop, intervals, values, good)
        return grid, synced, usable.astype(np.float64), names
    return intervals, values, good, names


def vision_source_openface(
    video: Path, work: Path, duration: float, cfg: dict[str, Any], sample_id: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """OpenFace AU35 → (intervals, values, quality, names)。"""
    of_bin = Path(cfg["paths"]["openface"])
    out_dir = work / "openface"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{video.stem}.csv"
    sid = sample_id or work.name
    if not install_openface_csv(sid, video.stem, csv_path):
        command(
            [str(of_bin), "-f", str(video), "-out_dir", str(out_dir), "-aus"],
            work / "openface.log",
            cwd=str(of_bin.parent),
        )
        if csv_path.exists():
            publish_openface_csv(sid, video.stem, csv_path)
    if not csv_path.exists():
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0, 35), dtype=np.float32),
            np.zeros(0, dtype=np.float64),
            list(VISION_NAMES),
        )
    frame = pd.read_csv(csv_path)
    frame.columns = frame.columns.str.strip()
    absent = set(VISION_NAMES) - set(frame.columns)
    if absent:
        raise ValueError(f"OpenFace 缺少 AU 字段: {sorted(absent)}")
    conf_min = float(cfg["vision"]["confidence_min"])
    if "timestamp" in frame.columns:
        ts = frame["timestamp"].to_numpy(dtype=float)
    else:
        fps = 30.0
        ts = np.arange(len(frame), dtype=float) / fps
    # 半帧宽近似支持区间
    if len(ts) > 1:
        half = float(np.median(np.diff(ts)) * 0.5)
    else:
        half = 1.0 / 60.0
    intervals = np.stack([np.maximum(0.0, ts - half), np.minimum(duration, ts + half)], axis=1)
    values = np.nan_to_num(frame[VISION_NAMES].to_numpy(dtype=np.float32), nan=0.0)
    success = frame["success"].to_numpy() if "success" in frame.columns else np.ones(len(frame))
    conf = frame["confidence"].to_numpy() if "confidence" in frame.columns else np.ones(len(frame))
    quality = ((success == 1) & (conf >= conf_min) & np.isfinite(values).all(1)).astype(np.float64)
    hop = common_hop_sec(cfg)
    grid, synced, usable = unify_modality(float(duration), hop, intervals, values, quality)
    return grid, synced, usable.astype(np.float64), list(VISION_NAMES)


def word_audio_quality(
    signal: np.ndarray | None,
    sr: int,
    intervals: np.ndarray,
    tau_a: float,
    k_min: int,
    frame_ms: float,
    hop_ms: float,
    floor: float,
) -> np.ndarray:
    """词区间上的音频质量掩码（能量帧数 + RMS）。"""
    K = len(intervals)
    out = np.zeros(K, dtype=np.float64)
    if signal is None or len(signal) == 0:
        return out
    fl = max(1, int(round(sr * frame_ms / 1000)))
    hop = max(1, int(round(sr * hop_ms / 1000)))
    for k, (s, e) in enumerate(intervals):
        i0 = max(0, int(round(s * sr)))
        i1 = min(len(signal), int(round(e * sr)))
        seg = signal[i0:i1]
        if seg.size == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(seg.astype(np.float64))) + 1e-20))
        if len(seg) < fl:
            vn = int(rms >= floor)
        else:
            n = 1 + (len(seg) - fl) // hop
            energies = [
                np.mean(np.square(seg[i * hop : i * hop + fl].astype(np.float64))) for i in range(n)
            ]
            vn = int(sum(e >= floor for e in energies))
        out[k] = 1.0 if (vn >= k_min and rms >= tau_a) else 0.0
    return out
