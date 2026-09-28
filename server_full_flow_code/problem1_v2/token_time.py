"""WordPiece ↔ MFA：内容子词继承整词时间窗（同词多子词共享同一 [s,e)）。"""
from __future__ import annotations

import re
from typing import Any

import numpy as np


def content_token_mask(ids: np.ndarray, tokenizer) -> np.ndarray:
    special = {
        int(tokenizer.cls_token_id),
        int(tokenizer.sep_token_id),
        int(tokenizer.pad_token_id),
    }
    mid = getattr(tokenizer, "mask_token_id", None)
    if mid is not None:
        special.add(int(mid))
    ids = np.asarray(ids).reshape(-1)
    return np.array([1 if int(i) not in special else 0 for i in ids], dtype=np.int8)


def lexical_char_spans(text: str) -> list[re.Match[str]]:
    return list(re.finditer(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*", text))


def bridge_wordpiece_to_mfa(
    text: str,
    words: list[str],
    intervals: np.ndarray,
    offset_mapping: list[tuple[int, int]],
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    tokenizer,
) -> dict[str, Any]:
    """内容词元继承 MFA 整词区间；同词子词得到相同 [s,e)；特殊符/标点无时间。"""
    L = len(input_ids)
    intervals = np.asarray(intervals, dtype=np.float64)
    token_intervals = np.zeros((L, 2), dtype=np.float64)
    time_mask = np.zeros(L, dtype=np.int8)
    statuses: list[str] = []
    bridges: list[list[int]] = [[] for _ in range(L)]

    if len(words) == 0 or len(intervals) == 0 or len(words) != len(intervals):
        return {
            "token_intervals": token_intervals,
            "time_mask": time_mask,
            "statuses": ["mfa_unavailable"] * L,
            "bridges": bridges,
            "word_axis": [],
        }

    spans = lexical_char_spans(text)
    if [s.group().lower() for s in spans] != words or len(spans) != len(words):
        raise ValueError("MFA 词与原文词级字符跨度不一致")

    word_axis = []
    for k, (w, span, (a, b)) in enumerate(zip(words, spans, intervals)):
        word_axis.append(
            {
                "word_index": k,
                "word": w,
                "original_text": span.group(),
                "char_start": int(span.start()),
                "char_end": int(span.end()),
                "start_seconds": float(a),
                "end_seconds": float(b),
            }
        )

    content = content_token_mask(input_ids, tokenizer) & (np.asarray(attention_mask) > 0)
    for i, ((c0, c1), ok) in enumerate(zip(offset_mapping, content)):
        if not ok:
            statuses.append("structural_or_padding")
            continue
        hit = [
            w
            for w in word_axis
            if min(c1, w["char_end"]) > max(c0, w["char_start"])
        ]
        if not hit:
            statuses.append("nonlexical_token_without_word_time")
            continue
        # 同词多子词：共享整词时间窗（hit 通常为同一词；跨词则取并集）
        a = float(hit[0]["start_seconds"])
        b = float(hit[-1]["end_seconds"])
        token_intervals[i] = (a, b)
        time_mask[i] = 1 if b > a else 0
        bridges[i] = [w["word_index"] for w in hit]
        statuses.append("mapped_content_token")

    return {
        "token_intervals": token_intervals,
        "time_mask": time_mask,
        "statuses": statuses,
        "bridges": bridges,
        "word_axis": word_axis,
    }
