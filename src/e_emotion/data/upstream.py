"""Canonical NPZ adapters for upstream model loaders.

The reference repositories use a small legacy dictionary convention.  This
module provides that view without writing a pickle copy of the competition
data.  It keeps the input seam in the canonical NPZ loader and exposes the
distinct text-token masks and native modality masks to adapted callers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from e_emotion.data.processed import MODALITIES, load_processed_dataset


def _load_token_fields(path: Path, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read token ids, attention mask and segment ids from one canonical split."""
    with np.load(path, allow_pickle=False) as payload:
        required = ("I", "Q", "G")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"{path}: BERT token fields are missing {missing}")
        input_ids = np.asarray(payload["I"])
        attention = np.asarray(payload["Q"])
        segments = np.asarray(payload["G"])
    expected = (size, 50)
    if input_ids.shape != expected or attention.shape != expected or segments.shape != expected:
        raise ValueError(f"{path}: I/Q/G must all have shape {expected}")
    if input_ids.dtype.kind not in "iu" or segments.dtype.kind not in "iu":
        raise ValueError(f"{path}: I and G must be integer arrays")
    if attention.dtype != np.dtype(bool):
        raise ValueError(f"{path}: Q must be a boolean attention mask")
    return (
        input_ids.astype(np.int64, copy=False),
        attention.astype(np.float32, copy=False),
        segments.astype(np.int64, copy=False),
    )


def load_canonical_dataset(root: str | Path) -> dict[str, dict[str, Any]]:
    """Return an upstream-compatible view of the canonical aligned NPZ data.

    ``text`` is ``XT`` for feature-only callers and ``text_bert`` is the
    ``(input_ids, attention_mask, segment_ids)`` tensor expected by the BERT
    based reference implementations.  Native validity and current observed
    masks remain separate and are included for adapted missingness-aware code.
    """
    base = Path(root).expanduser().resolve()
    dataset = load_processed_dataset(base)
    result: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "valid", "test"):
        split = dataset[split_name]
        token_ids, attention, segments = _load_token_fields(split.source_path, split.size)
        # Legacy BERT loaders expect one numeric tensor and cast the first and
        # third rows back to integer tensors inside the encoder.
        text_bert = np.stack((token_ids, attention, segments), axis=1).astype(np.float32)
        native = split.native_valid_mask
        features = {modality: np.array(split.features[modality], copy=True) for modality in MODALITIES}
        for modality in MODALITIES:
            features[modality][~native[modality]] = 0
        result[split_name] = {
            "id": [str(item) for item in split.ids.tolist()],
            "ids": split.ids,
            "raw_text": ["" for _ in range(split.size)],
            "text": features["text"],
            "text_bert": text_bert,
            "audio": features["audio"],
            "vision": features["vision"],
            "audio_lengths": native["audio"].sum(axis=1).astype(np.int32),
            "vision_lengths": native["vision"].sum(axis=1).astype(np.int32),
            "text_lengths": native["text"].sum(axis=1).astype(np.int32),
            "regression_labels": split.regression,
            "classification_labels": split.classification,
            "valid": {modality: native[modality] for modality in MODALITIES},
            "observed": {modality: split.observed_mask[modality] for modality in MODALITIES},
            "native_valid_mask": {modality: native[modality] for modality in MODALITIES},
            "observed_mask": {modality: split.observed_mask[modality] for modality in MODALITIES},
            "Q": split.Q,
            "P": split.P,
            "q": split.q,
            "rho_content": split.rho_content,
        }
    return result


__all__ = ["load_canonical_dataset"]
