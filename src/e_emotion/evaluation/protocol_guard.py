"""Strict downstream data and artifact protocol checks.

The guard is intentionally small: the shared ``e_emotion.data`` loader owns the
NPZ schema, while this module owns the comparison boundary (hashes, provenance,
and method-local artifact paths).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from e_emotion.data import (
    FEATURE_VERSION,
    PROTOCOL_VERSION,
    ProcessedDataset,
    build_processed_manifest,
    load_processed_dataset,
    sha256_file,
)

_SPLITS = ("train", "valid", "test")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_REQUIRED_FIELDS = (
    "XT",
    "XA",
    "XV",
    "mT",
    "mA",
    "mV",
    "Q",
    "P",
    "q",
    "rho_content",
    "ids",
    "y_regression",
    "y_classification",
)
_EXPECTED_SCHEMA = {
    "XT": ("float32", ["N", 50, 768]),
    "XA": ("float32", ["N", 50, 74]),
    "XV": ("float32", ["N", 50, 35]),
    "mT": ("bool", ["N", 50]),
    "mA": ("bool", ["N", 50]),
    "mV": ("bool", ["N", 50]),
    "Q": ("bool", ["N", 50]),
    "P": ("bool", ["N", 50]),
    "q": ("bool", ["N", 3]),
    "rho_content": ("float32", ["N", 3]),
    "ids": ("string", ["N"]),
    "y_regression": ("float32", ["N"]),
    "y_classification": ("int64", ["N"]),
}


def _load_npz_dataset(dataset: ProcessedDataset | str | Path) -> ProcessedDataset:
    if isinstance(dataset, ProcessedDataset):
        return dataset
    source = Path(dataset).expanduser()
    if source.suffix.lower() == ".pkl" or source.name.lower().endswith(".pkl"):
        raise ValueError("legacy .pkl inputs are rejected; use the processed NPZ directory")
    if source.is_file():
        if source.suffix.lower() != ".npz":
            raise ValueError("dataset input must be a directory containing train/valid/test .npz files")
        raise ValueError("dataset input must be the processed NPZ directory, not a single split")
    return load_processed_dataset(source)


def _resolve_path(value: str | Path, *, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


def _inside(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


def _required_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a 64-character SHA-256 hash")
    return value.lower()


def _check_dataset(
    dataset: ProcessedDataset,
    *,
    expected_split_sizes: Mapping[str, int] | None,
    expected_hashes: Mapping[str, str] | None,
) -> None:
    for split in _SPLITS:
        source = dataset[split].source_path.resolve()
        if source.suffix.lower() != ".npz":
            raise ValueError(f"{split} input must be an NPZ file, got {source}")
        if expected_split_sizes is not None:
            if split not in expected_split_sizes:
                raise ValueError(f"expected split sizes missing {split}")
            expected = int(expected_split_sizes[split])
            actual = dataset[split].size
            if actual != expected:
                raise ValueError(f"split size mismatch for {split}: expected {expected}, got {actual}")
        if expected_hashes is not None:
            if split not in expected_hashes:
                raise ValueError(f"expected hashes missing {split}")
            expected_hash = _required_sha(expected_hashes[split], f"expected {split} hash")
            actual_hash = sha256_file(source)
            if actual_hash != expected_hash:
                raise ValueError(f"hash mismatch for {split}: expected {expected_hash}, got {actual_hash}")


def _artifact_context(
    *,
    method: str,
    method_root: str | Path,
    artifact_root: str | Path,
    checkpoint: str | Path,
    threshold_source: str | Path | None,
    artifact_paths: Mapping[str, str | Path] | None,
) -> dict[str, Any]:
    if not isinstance(method, str) or not method.strip():
        raise ValueError("method must be a non-empty name")
    root = _resolve_path(method_root)
    artifacts = _resolve_path(artifact_root)
    if not _inside(artifacts, root):
        raise ValueError(f"artifact root {artifacts} must be inside method root {root}")

    def artifact_path(value: str | Path, label: str) -> Path:
        path = _resolve_path(value, base=artifacts)
        if not _inside(path, artifacts):
            raise ValueError(f"{label} must be inside artifact root {artifacts}")
        return path

    checkpoint_path = artifact_path(checkpoint, "checkpoint")
    threshold_path = artifact_path(threshold_source, "threshold source") if threshold_source is not None else None
    paths = {
        name: str(artifact_path(value, f"artifact path {name}"))
        for name, value in (artifact_paths or {}).items()
    }
    return {
        "method": method,
        "method_root": str(root),
        "artifact_root": str(artifacts),
        "checkpoint": str(checkpoint_path),
        "threshold_source": str(threshold_path) if threshold_path is not None else None,
        "artifact_paths": paths,
    }


def build_downstream_manifest(
    dataset: ProcessedDataset | str | Path,
    *,
    method: str,
    method_root: str | Path,
    artifact_root: str | Path,
    experiment_seed: int,
    mask_seed: int,
    normalization_source: str,
    checkpoint: str | Path,
    threshold_source: str | Path | None = None,
    mask_manifest_hash: str | None = None,
    expected_split_sizes: Mapping[str, int] | None = None,
    expected_hashes: Mapping[str, str] | None = None,
    artifact_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Validate a strict run boundary and return a JSON-serializable manifest."""
    if not isinstance(normalization_source, str) or not normalization_source.strip():
        raise ValueError("normalization source provenance is required")
    mask_hash = _required_sha(mask_manifest_hash, "mask manifest hash")
    loaded = _load_npz_dataset(dataset)
    _check_dataset(loaded, expected_split_sizes=expected_split_sizes, expected_hashes=expected_hashes)
    context = _artifact_context(
        method=method,
        method_root=method_root,
        artifact_root=artifact_root,
        checkpoint=checkpoint,
        threshold_source=threshold_source,
        artifact_paths=artifact_paths,
    )
    manifest = build_processed_manifest(
        loaded,
        experiment_seed=experiment_seed,
        mask_seed=mask_seed,
        normalization_source=normalization_source,
        checkpoint=context["checkpoint"],
        threshold_source=context["threshold_source"],
        mask_manifest_hash=mask_hash,
    )
    manifest.update(context)
    manifest["normalization"] = {"source": normalization_source}
    manifest["provenance"] = {
        "normalization_source": normalization_source,
        "checkpoint": context["checkpoint"],
        "threshold_source": context["threshold_source"],
        "mask_manifest_hash": mask_hash,
    }
    manifest["mask_semantics"] = {
        "native": "mT/mA/mV",
        "synthetic": "independent artificial Q2 deletions",
        "observed": "native_valid_mask & ~synthetic_missing_mask",
        "native_valid_mask": {"text": "mT", "audio": "mA", "vision": "mV"},
        "synthetic_missing_mask": "independent artificial Q2 deletions",
        "observed_mask": "native_valid_mask & ~synthetic_missing_mask",
        "q2_coordinate_mask": {"text": "mT", "audio": "mT & mA", "vision": "mT & mV"},
    }
    json.dumps(manifest, allow_nan=False)
    return manifest


def validate_downstream_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_split_sizes: Mapping[str, int] | None = None,
    expected_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate the serialized manifest before accepting a downstream result."""
    try:
        json.dumps(manifest, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest must be JSON serializable") from exc
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("manifest protocol_version does not match the shared protocol")
    if manifest.get("feature_version") != FEATURE_VERSION:
        raise ValueError("manifest feature_version does not match aligned_50")
    splits = manifest.get("splits")
    hashes = manifest.get("data_hashes")
    sizes = manifest.get("split_sizes")
    if not isinstance(splits, Mapping) or not isinstance(hashes, Mapping) or not isinstance(sizes, Mapping):
        raise ValueError("manifest must include splits, data_hashes and split_sizes")
    for split in _SPLITS:
        if split not in splits or split not in hashes or split not in sizes:
            raise ValueError(f"manifest missing {split} split hash or size")
        record = splits[split]
        if not isinstance(record, Mapping) or Path(str(record.get("source_path", ""))).suffix.lower() != ".npz":
            raise ValueError(f"{split} source_path must point to an NPZ file")
        _required_sha(hashes[split], f"{split} data hash")
        source_path = _resolve_path(str(record.get("source_path", "")))
        if source_path.is_file() and sha256_file(source_path) != str(hashes[split]).lower():
            raise ValueError(f"hash mismatch for {split}: source file does not match manifest")
        if int(record.get("size", -1)) != int(sizes[split]):
            raise ValueError(f"manifest split size mismatch for {split}")
        if expected_split_sizes is not None and int(sizes[split]) != int(expected_split_sizes[split]):
            raise ValueError(f"split size mismatch for {split}")
        if expected_hashes is not None and hashes[split].lower() != _required_sha(expected_hashes[split], f"expected {split} hash"):
            raise ValueError(f"hash mismatch for {split}")
    schema = manifest.get("field_schema")
    if not isinstance(schema, Mapping) or any(field not in schema for field in _REQUIRED_FIELDS):
        raise ValueError("manifest field schema is incomplete")
    for field, (dtype, dims) in _EXPECTED_SCHEMA.items():
        record = schema[field]
        if not isinstance(record, Mapping) or record.get("dtype") != dtype or list(record.get("dims", ())) != dims:
            raise ValueError(f"manifest field schema is invalid for {field}")
    if not manifest.get("mask_manifest_hash") or not _SHA256.fullmatch(str(manifest["mask_manifest_hash"])):
        raise ValueError("mask manifest hash provenance is required")
    semantics = manifest.get("mask_semantics")
    if not isinstance(semantics, Mapping) or not all(semantics.get(key) for key in ("native", "synthetic", "observed")):
        raise ValueError("mask provenance semantics are incomplete")
    method_root = _resolve_path(str(manifest.get("method_root", "")))
    artifact_root = _resolve_path(str(manifest.get("artifact_root", "")))
    if not _inside(artifact_root, method_root):
        raise ValueError("artifact root must be inside method root")
    for label in ("checkpoint", "threshold_source"):
        value = manifest.get(label)
        if value is not None and not _inside(_resolve_path(str(value), base=artifact_root), artifact_root):
            raise ValueError(f"{label} must be inside artifact root")
    artifact_paths = manifest.get("artifact_paths", {})
    if not isinstance(artifact_paths, Mapping):
        raise ValueError("artifact_paths must be a mapping")
    for name, value in artifact_paths.items():
        if not _inside(_resolve_path(str(value), base=artifact_root), artifact_root):
            raise ValueError(f"artifact path {name} must be inside artifact root")
    return dict(manifest)


def validate_processed_dataset(
    dataset: ProcessedDataset | str | Path,
    *,
    expected_split_sizes: Mapping[str, int] | None = None,
    expected_hashes: Mapping[str, str] | None = None,
) -> ProcessedDataset:
    """Load and validate the strict processed NPZ dataset boundary."""
    loaded = _load_npz_dataset(dataset)
    _check_dataset(loaded, expected_split_sizes=expected_split_sizes, expected_hashes=expected_hashes)
    return loaded


validate_protocol_manifest = validate_downstream_manifest
build_protocol_manifest = build_downstream_manifest
validate_npz_dataset = validate_processed_dataset
build_run_manifest = build_downstream_manifest


__all__ = [
    "build_downstream_manifest",
    "build_protocol_manifest",
    "build_run_manifest",
    "validate_processed_dataset",
    "validate_npz_dataset",
    "validate_downstream_manifest",
    "validate_protocol_manifest",
]
