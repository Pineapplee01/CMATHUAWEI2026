"""问题1预处理公共工具。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# problem1/ -> CPMCM/
ROOT = Path(__file__).resolve().parents[1]
PKG = Path(__file__).resolve().parent


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg = json.loads((path or PKG / "config_q1.json").read_text(encoding="utf-8"))
    for key, rel in list(cfg["paths"].items()):
        p = Path(rel)
        cfg["paths"][key] = str(p if p.is_absolute() else ROOT / p)
    return cfg


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_samples(cfg: dict[str, Any]) -> pd.DataFrame:
    labels = pd.read_excel(cfg["paths"]["labels"])
    labels["clip_id"] = labels["clip_id"].astype(int)
    labels["sample_id"] = labels["video_id"].astype(str) + "$_$" + labels["clip_id"].astype(str)
    labels["rel_path"] = labels["video_id"].astype(str) + "/" + labels["clip_id"].astype(str) + ".mp4"
    qc_path = Path(cfg["paths"]["qc"])
    if qc_path.exists():
        qc = pd.read_csv(qc_path)
        qc["clip_id"] = qc["clip_id"].astype(int)
        keep = [
            c
            for c in (
                "video_id",
                "clip_id",
                "container_duration_sec",
                "is_silent_audio",
                "usable_for_audio",
            )
            if c in qc.columns
        ]
        labels = labels.merge(qc[keep], on=["video_id", "clip_id"], how="left")
    return labels


def probe_duration(video: Path, fallback: float | None = None) -> float:
    if fallback is not None and np.isfinite(fallback) and float(fallback) > 0:
        return float(fallback)
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        text=True,
    ).strip()
    return float(out)


def equal_windows(duration: float, L: int = 50) -> np.ndarray:
    """在共同播放时间轴 [0, D] 上划 L 个等分窗；音视频共用此边界。"""
    T = max(float(duration), 1e-6)
    edges = np.linspace(0.0, T, L + 1)
    return np.stack([edges[:-1], edges[1:]], axis=1)


def playback_windows(duration: float, L: int = 50) -> np.ndarray:
    """共同播放时间轴上的 50 窗（equal_windows 别名）。"""
    return equal_windows(duration, L)


def modality_support(windows: np.ndarray, support_end: float) -> np.ndarray:
    """窗是否与模态实际支持区间 [0, support_end) 相交。

    片尾模态提前结束时，超出部分 support=0，禁止拉伸对齐终点。
    """
    end = max(float(support_end), 0.0)
    out = np.zeros(len(windows), dtype=np.int8)
    for k, (ws, we) in enumerate(windows):
        if min(float(we), end) > max(float(ws), 0.0):
            out[k] = 1
    return out


def probe_media_duration(video: Path) -> float:
    """视频流时长（秒）；失败则回退容器时长。"""
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video),
            ],
            text=True,
        ).strip()
        if out and out.upper() != "N/A":
            val = float(out)
            if np.isfinite(val) and val > 0:
                return val
    except (subprocess.CalledProcessError, ValueError):
        pass
    return probe_duration(video)


def extract_wav(video: Path, wav: Path, sr: int = 16000) -> bool:
    ensure_dir(wav.parent)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", str(wav)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return wav.exists() and wav.stat().st_size > 44
    except subprocess.CalledProcessError:
        return False
