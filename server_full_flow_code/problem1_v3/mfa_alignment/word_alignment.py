"""词级 MFA 强制对齐与时间边界检查。

来源：problem3/word_alignment.py（独立拷贝）。
对齐对象是规范化后的转写词，不是 BERT 词元；[CLS]/[SEP] 不参与对齐与时间聚合。
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np

from problem1_v3.mfa_alignment.media import command
from shared.common import path


def words_of(text: str) -> list[str]:
    """从转写抽取内容词；不含任何 BERT 特殊符号。"""
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*", text.lower())


def content_token_mask(input_ids, tokenizer) -> np.ndarray:
    """BERT 槽位上仅内容词元为 1；[CLS]/[SEP]/[PAD] 及特殊词元为 0。

    MFA 对齐与音视→词区间聚合只应使用内容词；特殊词元可保留编码，但不进时间映射。
    """
    ids = np.asarray(input_ids).reshape(-1)
    special = {
        int(tokenizer.cls_token_id),
        int(tokenizer.sep_token_id),
        int(tokenizer.pad_token_id),
    }
    if getattr(tokenizer, "mask_token_id", None) is not None:
        special.add(int(tokenizer.mask_token_id))
    return np.array([1 if int(i) not in special else 0 for i in ids], dtype=np.int8)


def _reuse_textgrid(work: Path, cfg: dict) -> bool:
    """若配置了 reuse_mfa_root 且该样本已有 TextGrid，拷入 work/aligned 并跳过 mfa align。"""
    root = cfg.get("reuse_mfa_root")
    if not root:
        return False
    src_aligned = Path(root) / work.name / "aligned"
    if not src_aligned.is_dir():
        return False
    candidates = list(src_aligned.rglob("*.TextGrid"))
    if len(candidates) != 1:
        return False
    aligned = work / "aligned"
    aligned.mkdir(parents=True, exist_ok=True)
    dst = aligned / "clip.TextGrid"
    if candidates[0].resolve() != dst.resolve():
        shutil.copy2(candidates[0], dst)
    return True


def word_alignment(text, audio, status, duration, work, cfg):
    """MFA 词对齐。失败时不均分降级，返回空区间并记录 failure。

    可通过 cfg['reuse_mfa_root'] 指向已有对齐目录（如 problem1_v3/results/_tmp_mfa），
    直接复用 TextGrid，避免 v2/v4 重复跑 MFA。注意：intervals.npz 是 view50 截断，
    复用请用完整 TextGrid，不要只用 (N,50,2) 的 intervals.npz。
    """
    words = words_of(text)
    empty = np.zeros((0, 2), dtype=np.float64)
    if not words:
        return [], empty, "mfa_failed", "none", "无可用转写词"
    failure = ""
    work = Path(work)
    if audio is not None:
        try:
            from praatio import textgrid

            aligned = work / "aligned"
            aligned.mkdir(exist_ok=True)
            reused = _reuse_textgrid(work, cfg)
            if not cfg.get("batch_alignment") and not reused:
                corpus = work / "corpus"
                corpus.mkdir(exist_ok=True)
                shutil.copyfile(work / "speech.wav", corpus / "clip.wav")
                (corpus / "clip.lab").write_text(" ".join(words), encoding="utf-8")
                command(
                    [
                        cfg["mfa_executable"],
                        "align",
                        corpus,
                        path(cfg["dictionary"]),
                        path(cfg["acoustic"]),
                        aligned,
                        "--clean",
                        "--single_speaker",
                        "--num_jobs",
                        "2",
                        "--temporary_directory",
                        work / "mfa_temp",
                    ],
                    work / "mfa.log",
                )
            candidates = list(aligned.rglob("*.TextGrid"))
            if len(candidates) != 1:
                raise ValueError("MFA 未输出唯一 TextGrid")
            tg = textgrid.openTextgrid(str(candidates[0]), includeEmptyIntervals=False)
            tier = next(
                tg.getTier(name) for name in tg.tierNames if name.lower().endswith("words")
            )
            entries = [
                (float(e.start), float(e.end), str(e.label).lower())
                for e in tier.entries
                if str(e.label).strip()
            ]
            if [e[2] for e in entries] != words:
                raise ValueError("MFA 词序列与规范化转写不一致，不能静默错配")
            intervals = np.array(
                [[a + status["audio_offset"], b + status["audio_offset"]] for a, b, _ in entries]
            )
            validate_intervals(intervals, duration)
            return words, intervals, "mfa", "estimated_by_forced_alignment", failure
        except (RuntimeError, ValueError, StopIteration, FileNotFoundError, ImportError) as exc:
            failure = str(exc)
    else:
        failure = str(status.get("audio_status", "no_audio"))
    # 不再 uniform_fallback：保留词表供台账，区间为空，特征侧全零
    return words, empty, "mfa_failed", "none", failure


def validate_intervals(intervals, duration):
    if (
        not np.isfinite(intervals).all()
        or np.any(intervals[:, 1] <= intervals[:, 0])
        or intervals[0, 0] < -1e-4
        or intervals[-1, 1] > duration + 1e-3
        or np.any(intervals[1:, 0] < intervals[:-1, 1] - 1e-4)
    ):
        raise ValueError("时间区间不合法/不单调/越界")
