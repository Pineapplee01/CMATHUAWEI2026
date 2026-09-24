"""Shared loader for the organizer's processed aligned NPZ files.

The NPZ masks describe native availability, not only padding.  Synthetic Q2
missingness is kept as a separate layer so a method cannot accidentally treat
organizer-side missing observations as padding or as an experiment condition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

ModalityName = Literal["text", "audio", "vision"]
MODALITIES: tuple[ModalityName, ...] = ("text", "audio", "vision")
PROTOCOL_VERSION = "e-competition-v1"
FEATURE_VERSION = "aligned_50"
_FEATURE_KEYS = {"text": "XT", "audio": "XA", "vision": "XV"}
_MASK_KEYS = {"text": "mT", "audio": "mA", "vision": "mV"}


@dataclass(frozen=True)
class ProcessedSplit:
    """One competition split with native and current observation masks."""

    features: Mapping[str, np.ndarray]
    native_valid_mask: Mapping[str, np.ndarray]
    observed_mask: Mapping[str, np.ndarray]
    ids: np.ndarray
    regression: np.ndarray
    classification: np.ndarray
    source_path: Path
    synthetic_missing_mask: Mapping[str, np.ndarray] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return int(self.ids.shape[0])

    def with_observed_mask(self, observed_mask: Mapping[str, np.ndarray]) -> "ProcessedSplit":
        """Return a view with an additional synthetic observation condition.

        The supplied mask may only hide native-valid positions.  Native
        unavailability is never turned into an observed position.
        """
        current: dict[str, np.ndarray] = {}
        updated: dict[str, np.ndarray] = {}
        for modality in MODALITIES:
            candidate = np.asarray(observed_mask[modality], dtype=bool)
            native = self.native_valid_mask[modality]
            if candidate.shape != native.shape:
                raise ValueError(f"{modality} observed mask shape mismatch")
            current[modality] = candidate & native
            values = self.features[modality].copy()
            values[~current[modality]] = 0
            updated[modality] = values
        return ProcessedSplit(
            features=updated,
            native_valid_mask=self.native_valid_mask,
            observed_mask=current,
            ids=self.ids,
            regression=self.regression,
            classification=self.classification,
            source_path=self.source_path,
            synthetic_missing_mask={
                modality: native & ~current[modality]
                for modality, native in self.native_valid_mask.items()
            },
        )


@dataclass(frozen=True)
class ProcessedDataset:
    """All three fixed Attachment 2 splits."""

    splits: Mapping[str, ProcessedSplit]
    feature_version: str = FEATURE_VERSION

    def __getitem__(self, split: str) -> ProcessedSplit:
        try:
            return self.splits[split]
        except KeyError as exc:
            raise KeyError(f"unknown split {split!r}; expected train, valid or test") from exc


def _as_bool_mask(value: np.ndarray, name: str, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(value, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {mask.shape}")
    return mask


def load_processed_split(path: str | Path) -> ProcessedSplit:
    """Load one organizer NPZ without inferring masks from feature values."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as payload:
        required = (*_FEATURE_KEYS.values(), *_MASK_KEYS.values(), "ids",
                    "y_regression", "y_classification")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"{source}: missing required fields {missing}")
        features: dict[str, np.ndarray] = {}
        native: dict[str, np.ndarray] = {}
        size: int | None = None
        for modality in MODALITIES:
            values = np.asarray(payload[_FEATURE_KEYS[modality]], dtype=np.float32)
            if values.ndim != 3:
                raise ValueError(f"{modality} features must be rank-3, got {values.shape}")
            if size is None:
                size = int(values.shape[0])
            if values.shape[0] != size or values.shape[1] != 50:
                raise ValueError(f"{modality} features must have shape (N, 50, D), got {values.shape}")
            if not np.isfinite(values).all():
                raise ValueError(f"{modality} features contain non-finite values")
            features[modality] = values
            native[modality] = _as_bool_mask(payload[_MASK_KEYS[modality]], modality, values.shape[:2])
        assert size is not None
        ids = np.asarray(payload["ids"])
        regression = np.asarray(payload["y_regression"], dtype=np.float32)
        classification = np.asarray(payload["y_classification"], dtype=np.int64)
        if ids.shape != (size,) or len({str(item) for item in ids.tolist()}) != size:
            raise ValueError("ids must be one-dimensional and unique within a split")
        if regression.shape != (size,) or not np.isfinite(regression).all() or np.any(np.abs(regression) > 3):
            raise ValueError("y_regression must be finite and within [-3, 3]")
        expected_classes = np.where(regression < 0, 0, np.where(regression > 0, 2, 1))
        if classification.shape != (size,) or not np.array_equal(classification, expected_classes):
            raise ValueError("y_classification must match y_regression polarity mapping")
    return ProcessedSplit(
        features=features,
        native_valid_mask=native,
        observed_mask={key: value.copy() for key, value in native.items()},
        ids=ids,
        regression=regression,
        classification=classification,
        source_path=source,
        synthetic_missing_mask={
            modality: np.zeros_like(mask, dtype=bool)
            for modality, mask in native.items()
        },
    )


def load_processed_dataset(root: str | Path) -> ProcessedDataset:
    """Load train/valid/test from a processed directory."""
    base = Path(root).expanduser().resolve()
    splits = {name: load_processed_split(base / f"{name}.npz") for name in ("train", "valid", "test")}
    all_ids = np.concatenate([splits[name].ids for name in ("train", "valid", "test")])
    if len({str(item) for item in all_ids.tolist()}) != len(all_ids):
        raise ValueError("sample IDs must be unique across train/valid/test")
    return ProcessedDataset(splits=splits)


def contiguous_missing_mask(
    split: ProcessedSplit,
    modality: ModalityName,
    fraction: float,
    position: Literal["beginning", "middle", "end"],
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic Q2 block mask in native-valid coordinates.

    Returns `(synthetic_missing_mask, observed_mask)`.  Padding/native-missing
    positions are never counted toward the requested fraction or selected as
    block coordinates.
    """
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must lie in [0, 1]")
    if position not in {"beginning", "middle", "end"}:
        raise ValueError("position must be beginning, middle or end")
    native = split.native_valid_mask[modality]
    coordinate_mask = native_coordinate_mask(split, modality)
    synthetic_missing = np.zeros_like(native, dtype=bool)
    for row, valid in enumerate(coordinate_mask):
        coordinates = np.flatnonzero(valid)
        count = int(round(len(coordinates) * fraction))
        if count == 0:
            continue
        if position == "beginning":
            selected = coordinates[:count]
        elif position == "end":
            selected = coordinates[-count:]
        else:
            start = (len(coordinates) - count) // 2
            selected = coordinates[start:start + count]
        synthetic_missing[row, selected] = True
    observed = native & ~synthetic_missing
    return synthetic_missing, observed


def native_coordinate_mask(split: ProcessedSplit, modality: ModalityName) -> np.ndarray:
    """Return the coordinate domain used for Q2 continuous missingness.

    Text positions are defined by ``mT``. Audio and vision positions are
    restricted to their intersection with text content; native-only positions
    therefore do not inflate the requested missing ratio.
    """
    if modality not in MODALITIES:
        raise ValueError(f"unknown modality {modality!r}")
    text_mask = split.native_valid_mask["text"]
    if modality == "text":
        return text_mask
    return text_mask & split.native_valid_mask[modality]


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a data file."""
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_processed_manifest(
    dataset: ProcessedDataset | str | Path,
    *,
    experiment_seed: int = 2026,
    mask_seed: int = 2026,
    normalization_source: str | None = None,
    checkpoint: str | None = None,
    threshold_source: str | None = None,
    mask_manifest_hash: str | None = None,
    mask_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable audit manifest without writing artifacts."""
    loaded = load_processed_dataset(dataset) if isinstance(dataset, (str, Path)) else dataset
    resolved_mask_hash = mask_sha256 if mask_sha256 is not None else mask_manifest_hash
    split_records: dict[str, Any] = {}
    for name in ("train", "valid", "test"):
        split = loaded[name]
        split_records[name] = {
            "source_path": str(split.source_path.resolve()),
            "sha256": sha256_file(split.source_path),
            "size": split.size,
            "id_count": split.size,
        }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "feature_version": FEATURE_VERSION,
        "splits": split_records,
        "split_sizes": {name: loaded[name].size for name in ("train", "valid", "test")},
        "source_paths": {name: record["source_path"] for name, record in split_records.items()},
        "data_hashes": {name: record["sha256"] for name, record in split_records.items()},
        "experiment_seed": int(experiment_seed),
        "mask_seed": int(mask_seed),
        "normalization_source": normalization_source,
        "checkpoint": checkpoint,
        "threshold_source": threshold_source,
        "mask_manifest_hash": resolved_mask_hash,
        "mask_sha256": resolved_mask_hash,
        "mask_semantics": {
            "native_valid_mask": {"text": "mT", "audio": "mA", "vision": "mV"},
            "synthetic_missing_mask": "independent artificial Q2 deletions",
            "observed_mask": "native_valid_mask & ~synthetic_missing_mask",
            "q2_coordinate_mask": {
                "text": "mT",
                "audio": "mT & mA",
                "vision": "mT & mV",
            },
        },
    }


build_manifest = build_processed_manifest


__all__ = [
    "MODALITIES",
    "PROTOCOL_VERSION",
    "FEATURE_VERSION",
    "ModalityName",
    "ProcessedDataset",
    "ProcessedSplit",
    "contiguous_missing_mask",
    "native_coordinate_mask",
    "sha256_file",
    "build_processed_manifest",
    "build_manifest",
    "load_processed_dataset",
    "load_processed_split",
]
