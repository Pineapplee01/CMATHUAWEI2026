"""Strict downstream data and artifact protocol checks.

The guard is intentionally small: the shared ``e_emotion.data`` loader owns the
NPZ schema, while this module owns the comparison boundary (hashes, provenance,
and method-local artifact paths).
"""

from __future__ import annotations

import json
import os
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
CANONICAL_PROCESSED_ROOT = "/user_home/gaojianan/CPMCM/AAAdata/Appendix_2/标准化/对齐版本/processed"
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
_EXPECTED_MASK_SEMANTICS = {
    "native": "mT/mA/mV",
    "synthetic": "independent artificial Q2 deletions",
    "observed": "native_valid_mask & ~synthetic_missing_mask",
    "native_valid_mask": {"text": "mT", "audio": "mA", "vision": "mV"},
    "synthetic_missing_mask": "independent artificial Q2 deletions",
    "observed_mask": "native_valid_mask & ~synthetic_missing_mask",
    "q2_coordinate_mask": {"text": "mT", "audio": "mT & mA", "vision": "mT & mV"},
}


def _load_npz_dataset(
    dataset: ProcessedDataset | str | Path,
    *,
    canonical_root: str | Path = CANONICAL_PROCESSED_ROOT,
) -> ProcessedDataset:
    if isinstance(dataset, ProcessedDataset):
        raise ValueError("strict protocol rejects preloaded/fabricated ProcessedDataset; pass the canonical NPZ root")
    source = Path(dataset).expanduser().resolve()
    if source.suffix.lower() == ".pkl" or source.name.lower().endswith(".pkl"):
        raise ValueError("legacy .pkl inputs are rejected; use the processed NPZ directory")
    if source.is_file():
        if source.suffix.lower() != ".npz":
            raise ValueError("dataset input must be a directory containing train/valid/test .npz files")
        raise ValueError("dataset input must be the processed NPZ directory, not a single split")
    root = _canonical_existing_dir(str(canonical_root), "canonical processed root")
    if source != root:
        raise ValueError(f"dataset root must be the canonical processed root {root}")
    expected_files = {f"{name}.npz" for name in _SPLITS}
    actual_files = {item.name for item in root.iterdir() if item.is_file()}
    if actual_files != expected_files:
        raise ValueError("canonical processed root must contain exactly train.npz, valid.npz and test.npz")
    return load_processed_dataset(root)


def _resolve_path(value: str | Path, *, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


def _canonical_existing_dir(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty absolute path")
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise ValueError(f"{label} must be absolute")
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"{label} must exist") from exc
    if not resolved.is_dir() or os.path.normcase(str(raw)) != os.path.normcase(str(resolved)):
        raise ValueError(f"{label} must be an existing canonical directory")
    return resolved


def _canonical_existing_file(value: Any, label: str, *, base: Path | None = None) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path")
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        if base is None:
            raise ValueError(f"{label} must be absolute")
        raw = base / raw
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"{label} must exist") from exc
    if not resolved.is_file() or os.path.normcase(str(raw)) != os.path.normcase(str(resolved)):
        raise ValueError(f"{label} must be an existing canonical file")
    return resolved


def _required_seed(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _inside(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


def _strictly_inside(candidate: Path, parent: Path) -> bool:
    return candidate != parent and _inside(candidate, parent)


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
    root = _canonical_existing_dir(str(method_root), "method root")
    artifacts = _canonical_existing_dir(str(artifact_root), "artifact root")
    if not _strictly_inside(artifacts, root):
        raise ValueError(f"artifact root {artifacts} must be inside method root {root}")

    def artifact_path(value: str | Path, label: str) -> Path:
        path = _canonical_existing_file(str(value), label, base=artifacts)
        if not _inside(path, artifacts):
            raise ValueError(f"{label} must be inside artifact root {artifacts}")
        return path

    checkpoint_path = artifact_path(checkpoint, "checkpoint")
    threshold_path = artifact_path(threshold_source, "threshold source")
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
    canonical_root: str | Path = CANONICAL_PROCESSED_ROOT,
) -> dict[str, Any]:
    """Validate a strict run boundary and return a JSON-serializable manifest."""
    if not isinstance(normalization_source, str) or not normalization_source.strip():
        raise ValueError("normalization source provenance is required")
    _required_seed(experiment_seed, "experiment_seed")
    _required_seed(mask_seed, "mask_seed")
    mask_hash = _required_sha(mask_manifest_hash, "mask manifest hash")
    loaded = _load_npz_dataset(dataset, canonical_root=canonical_root)
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
    # Keep the top-level schema at the exact public contract.  The shared data
    # builder also includes split-specific shape records for diagnostics, but
    # those belong in ``field_schema_by_split`` rather than the canonical map.
    manifest["field_schema"] = {
        field: {"dtype": dtype, "dims": list(dims)}
        for field, (dtype, dims) in _EXPECTED_SCHEMA.items()
    }
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
    canonical_root: str | Path = CANONICAL_PROCESSED_ROOT,
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
    method = manifest.get("method")
    if not isinstance(method, str) or not method.strip():
        raise ValueError("method must be a non-empty name")
    method_root = _canonical_existing_dir(manifest.get("method_root"), "method root")
    artifact_root = _canonical_existing_dir(manifest.get("artifact_root"), "artifact root")
    if not _strictly_inside(artifact_root, method_root):
        raise ValueError("artifact root must be inside method root")

    experiment_seed = manifest.get("experiment_seed")
    mask_seed = manifest.get("mask_seed")
    _required_seed(experiment_seed, "experiment_seed")
    _required_seed(mask_seed, "mask_seed")

    normalization_source = manifest.get("normalization_source")
    if not isinstance(normalization_source, str) or not normalization_source.strip():
        raise ValueError("normalization source provenance is required")
    normalization = manifest.get("normalization")
    provenance = manifest.get("provenance")
    if not isinstance(normalization, Mapping) or normalization.get("source") != normalization_source:
        raise ValueError("normalization provenance does not match normalization_source")
    if not isinstance(provenance, Mapping):
        raise ValueError("provenance is required")
    if provenance.get("normalization_source") != normalization_source:
        raise ValueError("provenance normalization_source does not match normalization_source")

    def provenance_file(value: Any, label: str) -> Path:
        path = _canonical_existing_file(value, label)
        if not _inside(path, artifact_root):
            raise ValueError(f"{label} must be inside artifact root")
        return path

    checkpoint = provenance_file(manifest.get("checkpoint"), "checkpoint")
    threshold_source = provenance_file(manifest.get("threshold_source"), "threshold source")
    if provenance.get("checkpoint") != str(checkpoint):
        raise ValueError("provenance checkpoint does not match checkpoint")
    if provenance.get("threshold_source") != str(threshold_source):
        raise ValueError("provenance threshold_source does not match threshold_source")

    canonical_root_path = _canonical_existing_dir(str(canonical_root), "canonical processed root")
    expected_files = {f"{name}.npz" for name in _SPLITS}
    if {item.name for item in canonical_root_path.iterdir() if item.is_file()} != expected_files:
        raise ValueError("canonical processed root must contain exactly train.npz, valid.npz and test.npz")
    splits = manifest.get("splits")
    hashes = manifest.get("data_hashes")
    sizes = manifest.get("split_sizes")
    source_paths = manifest.get("source_paths")
    if not isinstance(splits, Mapping) or not isinstance(hashes, Mapping) or not isinstance(sizes, Mapping) or not isinstance(source_paths, Mapping):
        raise ValueError("manifest must include splits, source_paths, data_hashes and split_sizes")
    for split in _SPLITS:
        if split not in splits or split not in hashes or split not in sizes:
            raise ValueError(f"manifest missing {split} split hash or size")
        record = splits[split]
        if not isinstance(record, Mapping):
            raise ValueError(f"{split} split record is invalid")
        source_path = _canonical_existing_file(record.get("source_path"), f"{split} source_path")
        if source_path.suffix.lower() != ".npz":
            raise ValueError(f"{split} source_path must point to an NPZ file")
        expected_source = (canonical_root_path / f"{split}.npz").resolve(strict=True)
        if source_path != expected_source:
            raise ValueError(f"{split} source_path must be the canonical {split}.npz")
        if source_paths.get(split) != str(source_path):
            raise ValueError(f"{split} source_paths record does not match split source_path")
        _required_sha(hashes[split], f"{split} data hash")
        actual_sha = sha256_file(source_path)
        if actual_sha != str(hashes[split]).lower() or record.get("sha256") != actual_sha:
            raise ValueError(f"hash mismatch for {split}: source file does not match manifest")
        if int(record.get("size", -1)) != int(sizes[split]) or int(record.get("id_count", -1)) != int(sizes[split]):
            raise ValueError(f"manifest split size mismatch for {split}")
        if expected_split_sizes is not None and int(sizes[split]) != int(expected_split_sizes[split]):
            raise ValueError(f"split size mismatch for {split}")
        if expected_hashes is not None and hashes[split].lower() != _required_sha(expected_hashes[split], f"expected {split} hash"):
            raise ValueError(f"hash mismatch for {split}")
    schema = manifest.get("field_schema")
    expected_schema = {
        field: {"dtype": dtype, "dims": list(dims)}
        for field, (dtype, dims) in _EXPECTED_SCHEMA.items()
    }
    if schema != expected_schema:
        raise ValueError("manifest field schema is not the exact canonical schema")
    # Reload the canonical files through the strict loader and cross-check the
    # recorded schema, sizes, source paths, hashes, and complete sample coverage.
    loaded = load_processed_dataset(canonical_root_path)
    field_schema_by_split = manifest.get("field_schema_by_split")
    sample_coverage = manifest.get("sample_coverage")
    sample_id_coverage = manifest.get("sample_id_coverage")
    if not isinstance(field_schema_by_split, Mapping) or not isinstance(sample_coverage, Mapping) or not isinstance(sample_id_coverage, Mapping):
        raise ValueError("manifest must include split field schema and sample coverage")
    for split in _SPLITS:
        actual = loaded[split]
        if int(sizes[split]) != actual.size:
            raise ValueError(f"manifest size does not match canonical {split} file")
        if source_path := Path(str(splits[split]["source_path"])).resolve():
            if source_path != actual.source_path.resolve():
                raise ValueError(f"manifest source path does not match canonical {split} file")
        arrays = {
            "XT": actual.features["text"], "XA": actual.features["audio"], "XV": actual.features["vision"],
            "mT": actual.native_valid_mask["text"], "mA": actual.native_valid_mask["audio"], "mV": actual.native_valid_mask["vision"],
            "Q": actual.Q, "P": actual.P, "q": actual.q, "rho_content": actual.rho_content,
            "ids": actual.ids, "y_regression": actual.regression, "y_classification": actual.classification,
        }
        expected_split_schema = {key: {"dtype": str(value.dtype), "shape": list(value.shape)} for key, value in arrays.items()}
        if field_schema_by_split.get(split) != expected_split_schema:
            raise ValueError(f"field schema mismatch for canonical {split} file")
        ids = [str(item) for item in actual.ids.tolist()]
        if sample_id_coverage.get(split) != ids:
            raise ValueError(f"sample ID coverage mismatch for {split}")
        record_coverage = sample_coverage.get(split)
        if not isinstance(record_coverage, Mapping) or record_coverage.get("size") != actual.size or record_coverage.get("ids") != ids:
            raise ValueError(f"sample coverage mismatch for {split}")
    top_mask_hash = manifest.get("mask_manifest_hash")
    if not isinstance(top_mask_hash, str) or not _SHA256.fullmatch(top_mask_hash):
        raise ValueError("mask manifest hash provenance is required")
    if manifest.get("mask_sha256") != top_mask_hash or provenance.get("mask_manifest_hash") != top_mask_hash:
        raise ValueError("mask manifest hash provenance is inconsistent")
    semantics = manifest.get("mask_semantics")
    if semantics != _EXPECTED_MASK_SEMANTICS:
        raise ValueError("mask provenance semantics are not canonical")
    artifact_paths = manifest.get("artifact_paths", {})
    if not isinstance(artifact_paths, Mapping):
        raise ValueError("artifact_paths must be a mapping")
    for name, value in artifact_paths.items():
        path = _canonical_existing_file(value, f"artifact path {name}")
        if not _inside(path, artifact_root):
            raise ValueError(f"artifact path {name} must be inside artifact root")
    return dict(manifest)


def validate_processed_dataset(
    dataset: ProcessedDataset | str | Path,
    *,
    expected_split_sizes: Mapping[str, int] | None = None,
    expected_hashes: Mapping[str, str] | None = None,
    canonical_root: str | Path = CANONICAL_PROCESSED_ROOT,
) -> ProcessedDataset:
    """Load and validate the strict processed NPZ dataset boundary."""
    loaded = _load_npz_dataset(dataset, canonical_root=canonical_root)
    _check_dataset(loaded, expected_split_sizes=expected_split_sizes, expected_hashes=expected_hashes)
    return loaded


validate_protocol_manifest = validate_downstream_manifest
build_protocol_manifest = build_downstream_manifest
validate_npz_dataset = validate_processed_dataset
build_run_manifest = build_downstream_manifest


__all__ = [
    "CANONICAL_PROCESSED_ROOT",
    "build_downstream_manifest",
    "build_protocol_manifest",
    "build_run_manifest",
    "validate_processed_dataset",
    "validate_npz_dataset",
    "validate_downstream_manifest",
    "validate_protocol_manifest",
]
