"""Deterministic, method-agnostic Q2 continuous-local-missingness masks.

The manifest describes masks in native coordinates and deliberately does not
contain model-specific arrays or output paths.  A caller may save the returned
JSON object wherever its method artifacts belong.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from e_emotion.data.processed import (
    MODALITIES,
    ProcessedDataset,
    contiguous_missing_mask,
    load_processed_dataset,
    native_coordinate_mask,
)

Q2_PROTOCOL_VERSION = "q2-continuous-local-v1"
NATIVE_MASK_VERSION = "aligned_50-native-v1"
Q2_COMBINATIONS: tuple[str, ...] = ("complete", "T", "A", "V", "TA", "TV", "AV")
Q2_POSITIONS: tuple[str, ...] = ("beginning", "middle", "end")
Q2_FRACTIONS: tuple[float, ...] = (0.1, 0.3, 0.5)
_COMBINATION_MODALITIES: dict[str, tuple[str, ...]] = {
    "complete": (),
    "T": ("text",),
    "A": ("audio",),
    "V": ("vision",),
    "TA": ("text", "audio"),
    "TV": ("text", "vision"),
    "AV": ("audio", "vision"),
}
_POSITION_VALUES = set(Q2_POSITIONS)
_SHA256_LENGTH = 64


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize a JSON object in one stable representation for hashing."""
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _manifest_hash(manifest_without_hash: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(manifest_without_hash)).hexdigest()


def _dataset(value: ProcessedDataset | str | Path) -> ProcessedDataset:
    return load_processed_dataset(value) if isinstance(value, (str, Path)) else value


def _strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _strict_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positions_for_mask(mask: np.ndarray) -> tuple[int | None, int | None, int]:
    positions = np.flatnonzero(mask)
    if positions.size == 0:
        return None, None, 0
    # End positions use the standard half-open interval convention [start, end).
    return int(positions[0]), int(positions[-1]) + 1, int(positions.size)


def _entry(
    split_name: str,
    split: Any,
    sample_index: int,
    combination: str,
    position: str,
    requested_fraction: float,
    experiment_seed: int,
    mask_seed: int,
    native_mask_version: str,
) -> dict[str, Any]:
    selected = _COMBINATION_MODALITIES[combination]
    native_lengths = {
        modality: int(split.native_valid_mask[modality][sample_index].sum())
        for modality in MODALITIES
    }
    coordinate_lengths = {
        modality: int(native_coordinate_mask(split, modality)[sample_index].sum())
        for modality in MODALITIES
    }
    starts: dict[str, int | None] = {modality: None for modality in MODALITIES}
    ends: dict[str, int | None] = {modality: None for modality in MODALITIES}
    counts: dict[str, int] = {modality: 0 for modality in MODALITIES}
    effective_by_modality: dict[str, float] = {modality: 0.0 for modality in MODALITIES}
    if combination != "complete":
        for modality in selected:
            missing, _ = contiguous_missing_mask(split, modality, requested_fraction, position)
            start, end, count = _positions_for_mask(missing[sample_index])
            starts[modality], ends[modality], counts[modality] = start, end, count
            domain_length = coordinate_lengths[modality]
            effective_by_modality[modality] = count / domain_length if domain_length else 0.0
    total_domain = sum(coordinate_lengths[modality] for modality in selected)
    total_count = sum(counts[modality] for modality in selected)
    effective = total_count / total_domain if total_domain else 0.0
    return {
        "split": split_name,
        "sample_id": str(split.ids[sample_index]),
        "combination": combination,
        "position": position,
        "requested_fraction": float(requested_fraction),
        "effective_fraction": float(effective),
        "effective_fractions": effective_by_modality,
        "native_valid_lengths": native_lengths,
        "coordinate_lengths": coordinate_lengths,
        "synthetic_start_positions": starts,
        "synthetic_end_positions": ends,
        "synthetic_missing_counts": counts,
        "experiment_seed": int(experiment_seed),
        "mask_seed": int(mask_seed),
        "native_mask_version": native_mask_version,
    }


def build_q2_mask_manifest(
    dataset: ProcessedDataset | str | Path,
    *,
    experiment_seed: int = 2026,
    mask_seed: int = 2026,
    native_mask_version: str = NATIVE_MASK_VERSION,
) -> dict[str, Any]:
    """Build the complete Q2 mask matrix without writing any files."""
    _strict_int(experiment_seed, "experiment_seed")
    _strict_int(mask_seed, "mask_seed")
    if native_mask_version != NATIVE_MASK_VERSION:
        raise ValueError("native_mask_version must be the canonical native mask version")
    loaded = _dataset(dataset)
    if loaded.feature_version != "aligned_50":
        raise ValueError("Q2 requires feature_version=aligned_50")
    entries: list[dict[str, Any]] = []
    for split_name in ("train", "valid", "test"):
        split = loaded[split_name]
        for sample_index in range(split.size):
            for combination in Q2_COMBINATIONS:
                for position in Q2_POSITIONS:
                    for fraction in Q2_FRACTIONS:
                        entries.append(
                            _entry(
                                split_name,
                                split,
                                sample_index,
                                combination,
                                position,
                                fraction,
                                experiment_seed,
                                mask_seed,
                                native_mask_version,
                            )
                        )
    manifest: dict[str, Any] = {
        "protocol_version": Q2_PROTOCOL_VERSION,
        "feature_version": loaded.feature_version,
        "experiment_seed": int(experiment_seed),
        "mask_seed": int(mask_seed),
        "native_mask_version": native_mask_version,
        "coordinate_domains": {"text": "mT", "audio": "mT & mA", "vision": "mT & mV"},
        "combinations": list(Q2_COMBINATIONS),
        "positions": list(Q2_POSITIONS),
        "fractions": list(Q2_FRACTIONS),
        "entries": entries,
    }
    manifest["mask_sha256"] = _manifest_hash(manifest)
    return manifest


def save_q2_mask_manifest(manifest: Mapping[str, Any], path: str | Path) -> str:
    """Write a validated manifest using stable, human-readable JSON.

    The output path is always supplied by the caller.  The returned value is
    the manifest's SHA-256, which is independent of indentation/newline style.
    """
    validate_q2_mask_manifest(manifest)
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return str(manifest["mask_sha256"])


def load_q2_mask_manifest(path: str | Path, dataset: ProcessedDataset | str | Path | None = None) -> dict[str, Any]:
    """Load and validate a JSON mask manifest."""
    source = Path(path).expanduser()
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Q2 mask manifest must be a JSON object")
    validate_q2_mask_manifest(value, dataset=dataset)
    return value


def validate_q2_mask_manifest(
    manifest: Mapping[str, Any],
    *,
    dataset: ProcessedDataset | str | Path | None = None,
) -> bool:
    """Validate hash, matrix coverage, and native-coordinate semantics."""
    required = {
        "protocol_version", "feature_version", "experiment_seed", "mask_seed",
        "native_mask_version", "coordinate_domains", "combinations", "positions",
        "fractions", "entries", "mask_sha256",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"Q2 mask manifest missing fields: {missing}")
    if manifest["protocol_version"] != Q2_PROTOCOL_VERSION:
        raise ValueError("unsupported Q2 mask protocol version")
    _strict_int(manifest["experiment_seed"], "manifest experiment_seed")
    _strict_int(manifest["mask_seed"], "manifest mask_seed")
    if manifest["feature_version"] != "aligned_50":
        raise ValueError("Q2 requires feature_version=aligned_50")
    if manifest["native_mask_version"] != NATIVE_MASK_VERSION:
        raise ValueError("Q2 native_mask_version is not canonical")
    if list(manifest["combinations"]) != list(Q2_COMBINATIONS):
        raise ValueError("Q2 combination matrix is not canonical")
    if list(manifest["positions"]) != list(Q2_POSITIONS):
        raise ValueError("Q2 position matrix is not canonical")
    if (
        not isinstance(manifest["fractions"], list)
        or any(type(value) is not float for value in manifest["fractions"])
        or manifest["fractions"] != list(Q2_FRACTIONS)
    ):
        raise ValueError("Q2 fraction matrix is not canonical")
    if manifest["coordinate_domains"] != {"text": "mT", "audio": "mT & mA", "vision": "mT & mV"}:
        raise ValueError("Q2 coordinate domains are not canonical")
    supplied_hash = manifest["mask_sha256"]
    if not isinstance(supplied_hash, str) or len(supplied_hash) != _SHA256_LENGTH:
        raise ValueError("mask_sha256 must be a SHA-256 hex digest")
    try:
        int(supplied_hash, 16)
    except ValueError as exc:
        raise ValueError("mask_sha256 must be a SHA-256 hex digest") from exc
    without_hash = dict(manifest)
    without_hash.pop("mask_sha256", None)
    if _manifest_hash(without_hash) != supplied_hash:
        raise ValueError("Q2 mask manifest hash mismatch")
    entries = manifest["entries"]
    if not isinstance(entries, list):
        raise ValueError("entries must be a list")
    if not entries:
        raise ValueError("entries must not be empty")
    expected_per_sample = len(Q2_COMBINATIONS) * len(Q2_POSITIONS) * len(Q2_FRACTIONS)
    if dataset is not None:
        loaded = _dataset(dataset)
        expected_total = sum(loaded[name].size for name in ("train", "valid", "test")) * expected_per_sample
        if len(entries) != expected_total:
            raise ValueError(f"expected {expected_total} Q2 entries, got {len(entries)}")
    seen: set[tuple[str, str, str, str, float]] = set()
    sample_counts: dict[tuple[str, str], int] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("each Q2 entry must be an object")
        for field in (
            "split", "sample_id", "combination", "position", "requested_fraction", "effective_fraction",
            "native_valid_lengths", "coordinate_lengths", "synthetic_start_positions",
            "synthetic_end_positions", "synthetic_missing_counts", "experiment_seed", "mask_seed",
            "native_mask_version",
        ):
            if field not in entry:
                raise ValueError(f"Q2 entry missing field {field!r}")
        combination = entry["combination"]
        position = entry["position"]
        if type(entry["sample_id"]) is not str or type(entry["split"]) is not str:
            raise ValueError("Q2 split and sample_id must be strings")
        if type(entry["requested_fraction"]) is not float:
            raise ValueError("Q2 requested_fraction must be a canonical float")
        fraction = entry["requested_fraction"]
        key = (str(entry["split"]), str(entry["sample_id"]), str(combination), str(position), fraction)
        if key in seen:
            raise ValueError(f"duplicate Q2 entry {key}")
        seen.add(key)
        sample_key = (str(entry["split"]), str(entry["sample_id"]))
        sample_counts[sample_key] = sample_counts.get(sample_key, 0) + 1
        if combination not in _COMBINATION_MODALITIES or position not in _POSITION_VALUES or fraction not in Q2_FRACTIONS:
            raise ValueError("Q2 entry is outside the canonical matrix")
        if entry["experiment_seed"] != manifest["experiment_seed"] or entry["mask_seed"] != manifest["mask_seed"]:
            raise ValueError("Q2 entry seed differs from manifest seed")
        if entry["native_mask_version"] != manifest["native_mask_version"]:
            raise ValueError("Q2 entry native mask version differs from manifest")
        starts = entry["synthetic_start_positions"]
        ends = entry["synthetic_end_positions"]
        counts = entry["synthetic_missing_counts"]
        lengths = entry["coordinate_lengths"]
        native_lengths = entry["native_valid_lengths"]
        effective_by_modality = entry.get("effective_fractions")
        if not isinstance(starts, Mapping) or not isinstance(ends, Mapping) or not isinstance(counts, Mapping) or not isinstance(lengths, Mapping) or not isinstance(native_lengths, Mapping) or not isinstance(effective_by_modality, Mapping):
            raise ValueError("Q2 entry position/count maps must be mappings")
        if not all(modality in starts and modality in ends and modality in counts and modality in lengths and modality in native_lengths and modality in effective_by_modality for modality in MODALITIES):
            raise ValueError("Q2 entry position/count maps must cover all modalities")
        selected = _COMBINATION_MODALITIES[combination]
        for modality in MODALITIES:
            count = _strict_int(counts[modality], f"{modality} synthetic count")
            start, end = starts[modality], ends[modality]
            domain_length = _strict_int(lengths[modality], f"{modality} coordinate length")
            native_length = _strict_int(native_lengths[modality], f"{modality} native length")
            if domain_length < 0 or native_length < 0 or native_length > 50 or domain_length > native_length:
                raise ValueError("coordinate domain length must be non-negative")
            if modality not in selected and (count != 0 or start is not None or end is not None):
                raise ValueError("synthetic deletion found outside the requested combination")
            if count == 0 and (start is not None or end is not None):
                raise ValueError("empty synthetic deletion must have null positions")
            if count < 0:
                raise ValueError("synthetic deletion count must be non-negative")
            if count > 0:
                if isinstance(start, bool) or not isinstance(start, int) or isinstance(end, bool) or not isinstance(end, int):
                    raise ValueError("synthetic deletion positions must be integers")
                if start < 0 or end <= start or end > 50:
                    raise ValueError("invalid synthetic deletion positions")
                if end - start < count:
                    raise ValueError("synthetic deletion interval is shorter than its count")
            if count > domain_length:
                raise ValueError("synthetic deletion exceeds coordinate domain")
            effective = _strict_float(effective_by_modality[modality], f"{modality} effective fraction")
            expected_effective = count / domain_length if domain_length else 0.0
            if effective != expected_effective:
                raise ValueError("effective fraction does not equal count/domain")
        total_count = sum(_strict_int(counts[m], f"{m} synthetic count") for m in selected)
        total_domain = sum(_strict_int(lengths[m], f"{m} coordinate length") for m in selected)
        expected_total_effective = total_count / total_domain if total_domain else 0.0
        if _strict_float(entry["effective_fraction"], "effective_fraction") != expected_total_effective:
            raise ValueError("effective_fraction does not equal count/domain")
        if combination == "complete" and (float(entry["effective_fraction"]) != 0.0 or any(int(v) for v in counts.values())):
            raise ValueError("complete condition must preserve native masks")
    for sample_key, count in sample_counts.items():
        if count != expected_per_sample:
            raise ValueError(f"Q2 matrix coverage mismatch for {sample_key}: expected {expected_per_sample}, got {count}")
    if dataset is not None:
        loaded = _dataset(dataset)
        for split_name in ("train", "valid", "test"):
            split = loaded[split_name]
            split_entries = [e for e in entries if e["split"] == split_name]
            if len(split_entries) != split.size * expected_per_sample:
                raise ValueError(f"entry coverage mismatch for {split_name}")
            ids = {str(item) for item in split.ids.tolist()}
            if {str(e["sample_id"]) for e in split_entries} != ids:
                raise ValueError(f"sample coverage mismatch for {split_name}")
            for entry in split_entries:
                row = next(i for i, value in enumerate(split.ids.tolist()) if str(value) == str(entry["sample_id"]))
                selected = _COMBINATION_MODALITIES[entry["combination"]]
                expected_lengths = {m: int(split.native_valid_mask[m][row].sum()) for m in MODALITIES}
                expected_domains = {m: int(native_coordinate_mask(split, m)[row].sum()) for m in MODALITIES}
                if entry["native_valid_lengths"] != expected_lengths or entry["coordinate_lengths"] != expected_domains:
                    raise ValueError("native length or coordinate domain mismatch")
                for modality in selected:
                    missing, _ = contiguous_missing_mask(split, modality, float(entry["requested_fraction"]), entry["position"])
                    start, end, count = _positions_for_mask(missing[row])
                    if (entry["synthetic_start_positions"][modality], entry["synthetic_end_positions"][modality], entry["synthetic_missing_counts"][modality]) != (start, end, count):
                        raise ValueError("synthetic positions do not match native coordinate semantics")
    return True


# Concise aliases for callers that use the protocol's shorter name.
generate_q2_mask_manifest = build_q2_mask_manifest
write_q2_mask_manifest = save_q2_mask_manifest

__all__ = [
    "Q2_PROTOCOL_VERSION", "NATIVE_MASK_VERSION", "Q2_COMBINATIONS", "Q2_POSITIONS", "Q2_FRACTIONS",
    "build_q2_mask_manifest", "generate_q2_mask_manifest", "save_q2_mask_manifest", "write_q2_mask_manifest",
    "load_q2_mask_manifest", "validate_q2_mask_manifest",
]
