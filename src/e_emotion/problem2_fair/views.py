"""Explicit Problem 2 processed_po input views."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from e_emotion.problem2_fair.core import MODALITIES, Problem2Split


@dataclass(frozen=True)
class Problem2Dataset:
    """The organizer train/valid/test splits for one declared input view."""

    splits: Mapping[str, Problem2Split]
    view: str
    root: Path

    def __getitem__(self, split: str) -> Problem2Split:
        try:
            return self.splits[split]
        except KeyError as exc:
            raise KeyError(f"unknown split {split!r}; expected train, valid or test") from exc


def _array(payload, key: str, *, dtype=None) -> np.ndarray:
    if key not in payload:
        raise ValueError(f"processed_po is missing required field {key!r}")
    value = np.asarray(payload[key])
    if value.dtype == object and key not in {"id", "ids"}:
        raise ValueError(f"{key} must not use an object dtype")
    if dtype is not None and value.dtype != np.dtype(dtype):
        raise ValueError(f"{key} must have dtype {np.dtype(dtype)}, got {value.dtype}")
    return value


def _labels(payload, size: int) -> tuple[np.ndarray, np.ndarray]:
    regression = _array(payload, "regression_labels").astype(np.float32, copy=False).reshape(-1)
    classification = _array(payload, "classification_labels").astype(np.int64, copy=False).reshape(-1)
    if regression.shape != (size,) or classification.shape != (size,):
        raise ValueError("labels must have one value per sample")
    if not np.isfinite(regression).all() or np.any(np.abs(regression) > 3):
        raise ValueError("regression labels must be finite values in [-3, 3]")
    if not np.array_equal(classification, np.sign(regression).astype(np.int64) + 1):
        raise ValueError("classification labels must match regression polarity")
    return regression, classification


def _identifiers(payload, size: int) -> tuple[np.ndarray, np.ndarray]:
    raw_ids = _array(payload, "id" if "id" in payload else "ids")
    ids = raw_ids.astype(str)
    input_ids = _array(payload, "I", dtype=np.int64)
    if ids.shape != (size,) or len(set(ids.tolist())) != size:
        raise ValueError("id must contain unique entries")
    if input_ids.shape != (size, 50):
        raise ValueError("I must have shape [N, 50]")
    return ids, input_ids


def _bool_mask(payload, key: str, shape: tuple[int, int]) -> np.ndarray:
    value = _array(payload, key, dtype=bool)
    if value.shape != shape:
        raise ValueError(f"{key} must have shape {shape}, got {value.shape}")
    return value


def _load_split(path: Path, view: str, split: str) -> Problem2Split:
    if not path.is_file():
        raise FileNotFoundError(path)
    # Organizer processed_po exports IDs as object arrays.  Permit pickle only
    # for this trusted, fixed-format file and reject object dtypes elsewhere.
    with np.load(path, allow_pickle=True) as payload:
        text = _array(payload, "XT", dtype=np.float32)
        size = text.shape[0]
        if text.shape[:2] != (size, 50):
            raise ValueError("XT must have shape [N, 50, feature]")
        ids, input_ids = _identifiers(payload, size)
        regression, classification = _labels(payload, size)
        if view == "aligned_po":
            audio = _array(payload, "XA", dtype=np.float32)
            vision = _array(payload, "XV", dtype=np.float32)
            if audio.shape[:2] != (size, 50) or vision.shape[:2] != (size, 50):
                raise ValueError("aligned_po XA/XV must have 50 positions")
            physical = _array(payload, "P", dtype=bool)
            observed = _array(payload, "O", dtype=bool)
            if physical.shape != (size, 3, 50) or observed.shape != physical.shape:
                raise ValueError("aligned_po P/O must have shape [N, 3, 50]")
            support = {name: physical[:, index] for index, name in enumerate(MODALITIES)}
            masks = {name: observed[:, index] for index, name in enumerate(MODALITIES)}
        elif view == "unaligned_po":
            audio = _array(payload, "XA", dtype=np.float32)
            vision = _array(payload, "XV", dtype=np.float32)
            if audio.shape[:2] != (size, 500) or vision.shape[:2] != (size, 500):
                raise ValueError("unaligned_po XA/XV must have 500 positions")
            support = {
                "text": _bool_mask(payload, "P_T", (size, 50)),
                "audio": _bool_mask(payload, "P_A", (size, 500)),
                "vision": _bool_mask(payload, "P_V", (size, 500)),
            }
            masks = {
                "text": _bool_mask(payload, "O_T", (size, 50)),
                "audio": _bool_mask(payload, "O_A", (size, 500)),
                "vision": _bool_mask(payload, "O_V", (size, 500)),
            }
        else:
            raise ValueError(f"unsupported on-disk view: {view!r}")
    values = {"text": np.array(text, copy=True), "audio": np.array(audio, copy=True), "vision": np.array(vision, copy=True)}
    for modality in MODALITIES:
        values[modality][~masks[modality]] = 0.0
    input_ids = np.array(input_ids, copy=True)
    input_ids[~masks["text"]] = 0
    return Problem2Split(
        ids=ids,
        values=values,
        physical_support=support,
        observed=masks,
        input_ids=input_ids,
        regression=regression,
        classification=classification,
        view=view,
        split=split,
    )


def load_problem2_dataset(root: str | Path, *, view: str) -> Problem2Dataset:
    """Load a declared processed_po root without coercing its time layout."""
    source = Path(root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    splits = {name: _load_split(source / f"{name}.npz", view, name) for name in ("train", "valid", "test")}
    all_ids = np.concatenate([splits[name].ids for name in ("train", "valid", "test")])
    if len(set(all_ids.tolist())) != len(all_ids):
        raise ValueError("sample IDs must be unique across train/valid/test")
    return Problem2Dataset(splits=splits, view=view, root=source)


__all__ = ["Problem2Dataset", "load_problem2_dataset"]
