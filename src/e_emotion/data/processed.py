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
    token_attention_mask: np.ndarray | None = None
    token_padding_mask: np.ndarray | None = None
    sample_modality_available: np.ndarray | None = None
    content_missing_rate: np.ndarray | None = None

    @property
    def Q(self) -> np.ndarray:
        """Organizer token attention mask (not a native modality mask)."""
        if self.token_attention_mask is None:
            raise AttributeError("token attention mask Q is unavailable")
        return self.token_attention_mask

    @property
    def P(self) -> np.ndarray:
        """Organizer token padding mask, exactly ``~Q``."""
        if self.token_padding_mask is None:
            raise AttributeError("token padding mask P is unavailable")
        return self.token_padding_mask

    @property
    def q(self) -> np.ndarray:
        """Sample-level modality availability vector."""
        if self.sample_modality_available is None:
            raise AttributeError("sample modality availability q is unavailable")
        return self.sample_modality_available

    @property
    def rho_content(self) -> np.ndarray:
        """Content-relative native missing rates for text/audio/vision."""
        if self.content_missing_rate is None:
            raise AttributeError("content missing rate rho_content is unavailable")
        return self.content_missing_rate

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
            native = self.native_valid_mask[modality]
            candidate = _as_bool_mask(observed_mask[modality], f"{modality} observed mask", native.shape)
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
            token_attention_mask=self.token_attention_mask,
            token_padding_mask=self.token_padding_mask,
            sample_modality_available=self.sample_modality_available,
            content_missing_rate=self.content_missing_rate,
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


def _as_bool_mask(value: np.ndarray, name: str, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != np.dtype(bool):
        raise ValueError(f"{name} must have boolean dtype, got {raw.dtype}")
    mask = raw
    if mask.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {mask.shape}")
    return mask


def load_processed_split(path: str | Path) -> ProcessedSplit:
    """Load one organizer NPZ without inferring masks from feature values."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as payload:
        required = (*_FEATURE_KEYS.values(), *_MASK_KEYS.values(), "Q", "P", "q",
                    "rho_content", "ids", "y_regression", "y_classification")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"{source}: missing required fields {missing}")
        features: dict[str, np.ndarray] = {}
        native: dict[str, np.ndarray] = {}
        size: int | None = None
        for modality in MODALITIES:
            values = np.asarray(payload[_FEATURE_KEYS[modality]])
            if values.ndim != 3:
                raise ValueError(f"{modality} features must be rank-3, got {values.shape}")
            expected_dim = {"text": 768, "audio": 74, "vision": 35}[modality]
            if size is None:
                size = int(values.shape[0])
            if values.shape != (size, 50, expected_dim):
                raise ValueError(
                    f"aligned_50 {modality} features must have shape (N, 50, {expected_dim}), got {values.shape}"
                )
            if values.dtype != np.dtype(np.float32):
                raise ValueError(f"{modality} features must have float32 dtype, got {values.dtype}")
            if not np.isfinite(values).all():
                raise ValueError(f"{modality} features contain non-finite values")
            features[modality] = values
            native[modality] = _as_bool_mask(payload[_MASK_KEYS[modality]], modality, values.shape[:2])
        assert size is not None
        q = _as_bool_mask(payload["q"], "q", (size, 3))
        Q = _as_bool_mask(payload["Q"], "Q", (size, 50))
        P = _as_bool_mask(payload["P"], "P", (size, 50))
        if not np.array_equal(P, ~Q):
            raise ValueError("P must equal ~Q")
        if np.any(native["text"] & ~Q):
            raise ValueError("mT must be a subset of Q")
        expected_q = np.stack([native[modality].any(axis=1) for modality in MODALITIES], axis=1)
        if not np.array_equal(q, expected_q):
            raise ValueError("q must match sample-level native modality availability")
        rho_content = np.asarray(payload["rho_content"])
        if rho_content.shape != (size, 3):
            raise ValueError(f"rho_content must have shape {(size, 3)}, got {rho_content.shape}")
        if rho_content.dtype.kind not in "fc" or not np.isfinite(rho_content).all():
            raise ValueError("rho_content must contain finite floating-point values")
        text_lengths = native["text"].sum(axis=1).astype(np.float32)
        expected_rho = np.zeros((size, 3), dtype=np.float32)
        for index, modality in enumerate(("audio", "vision"), start=1):
            overlap = (native["text"] & native[modality]).sum(axis=1)
            expected_rho[:, index] = np.divide(
                text_lengths - overlap,
                text_lengths,
                out=np.zeros(size, dtype=np.float32),
                where=text_lengths > 0,
            )
        if np.any(rho_content < 0) or np.any(rho_content > 1) or not np.allclose(rho_content, expected_rho):
            raise ValueError("rho_content must be [0,1] content-relative native missing rates")
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
        token_attention_mask=Q,
        token_padding_mask=P,
        sample_modality_available=q,
        content_missing_rate=rho_content.astype(np.float32, copy=False),
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
    field_schema_by_split: dict[str, Any] = {}
    sample_coverage: dict[str, Any] = {}
    sample_id_coverage: dict[str, list[str]] = {}
    for name in ("train", "valid", "test"):
        split = loaded[name]
        split_records[name] = {
            "source_path": str(split.source_path.resolve()),
            "sha256": sha256_file(split.source_path),
            "size": split.size,
            "id_count": split.size,
        }
        arrays: dict[str, np.ndarray] = {
            "XT": split.features["text"],
            "XA": split.features["audio"],
            "XV": split.features["vision"],
            "mT": split.native_valid_mask["text"],
            "mA": split.native_valid_mask["audio"],
            "mV": split.native_valid_mask["vision"],
            "Q": split.Q,
            "P": split.P,
            "q": split.q,
            "rho_content": split.rho_content,
            "ids": split.ids,
            "y_regression": split.regression,
            "y_classification": split.classification,
        }
        field_schema_by_split[name] = {
            key: {"dtype": str(value.dtype), "shape": list(value.shape)}
            for key, value in arrays.items()
        }
        ids = [str(item) for item in split.ids.tolist()]
        sample_id_coverage[name] = ids
        sample_coverage[name] = {
            "size": split.size,
            "ids": ids,
            "first_id": ids[0] if ids else None,
            "last_id": ids[-1] if ids else None,
        }
    field_schema = {
        "XT": {"dtype": "float32", "dims": ["N", 50, 768]},
        "XA": {"dtype": "float32", "dims": ["N", 50, 74]},
        "XV": {"dtype": "float32", "dims": ["N", 50, 35]},
        "mT": {"dtype": "bool", "dims": ["N", 50]},
        "mA": {"dtype": "bool", "dims": ["N", 50]},
        "mV": {"dtype": "bool", "dims": ["N", 50]},
        "Q": {"dtype": "bool", "dims": ["N", 50]},
        "P": {"dtype": "bool", "dims": ["N", 50]},
        "q": {"dtype": "bool", "dims": ["N", 3]},
        "rho_content": {"dtype": "float32", "dims": ["N", 3]},
        "ids": {"dtype": "string", "dims": ["N"]},
        "y_regression": {"dtype": "float32", "dims": ["N"]},
        "y_classification": {"dtype": "int64", "dims": ["N"]},
    }
    # Keep split-specific observed shapes alongside the canonical contract.
    field_schema.update(field_schema_by_split)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "feature_version": FEATURE_VERSION,
        "splits": split_records,
        "split_sizes": {name: loaded[name].size for name in ("train", "valid", "test")},
        "source_paths": {name: record["source_path"] for name, record in split_records.items()},
        "data_hashes": {name: record["sha256"] for name, record in split_records.items()},
        "field_schema": field_schema,
        "field_schema_by_split": field_schema_by_split,
        "sample_id_coverage": sample_id_coverage,
        "sample_coverage": sample_coverage,
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
