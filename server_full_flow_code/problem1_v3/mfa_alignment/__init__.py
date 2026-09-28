"""problem1_v3 MFA 词级对齐包：自 problem3 / 旧 problem1 对齐页拷贝，供方案A使用。

对齐单元 = 转写词区间，不含 BERT [CLS]/[SEP]。
"""

from problem1_v3.mfa_alignment.word_alignment import (
    words_of,
    word_alignment,
    validate_intervals,
    content_token_mask,
)
from problem1_v3.mfa_alignment.aggregate import aggregate, view50, view50_metadata, gaps

__all__ = [
    "words_of",
    "word_alignment",
    "validate_intervals",
    "content_token_mask",
    "aggregate",
    "view50",
    "view50_metadata",
    "gaps",
]
