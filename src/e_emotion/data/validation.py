"""Data-contract validation and lightweight inspection."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from e_emotion.contracts import DatasetVariant
from e_emotion.data.repositories import (
    Attachment2Repository,
    ExplainabilityRepository,
    MissingModalityRepository,
)
from e_emotion.data.processed import build_processed_manifest, load_processed_dataset


@dataclass(frozen=True)
class ValidationReport:
    variant: str
    split_sizes: Mapping[str, int]
    errors: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.errors


def validate_attachment2_payload(
    payload: Mapping[str, Any], variant: DatasetVariant | str
) -> ValidationReport:
    selected = variant.value if isinstance(variant, DatasetVariant) else str(variant)
    expected_steps = 50 if selected == DatasetVariant.ALIGNED.value else 500
    errors: list[str] = []
    split_sizes: dict[str, int] = {}
    all_ids: list[str] = []
    for split in ("train", "valid", "test"):
        section = payload.get(split)
        if not isinstance(section, Mapping):
            errors.append(f"missing split: {split}")
            continue
        ids = section.get("id")
        if ids is None:
            errors.append(f"{split}.id is missing")
            continue
        n = len(ids)
        split_sizes[split] = n
        all_ids.extend(str(item) for item in ids)
        expected_shapes = {
            "text": (n, 50, 768),
            "audio": (n, expected_steps, 74),
            "vision": (n, expected_steps, 35),
        }
        for field, shape in expected_shapes.items():
            value = section.get(field)
            if not hasattr(value, "shape") or tuple(value.shape) != shape:
                errors.append(f"{split}.{field} shape must be {shape}")
        for field in ("classification_labels", "regression_labels"):
            values = np.asarray(section.get(field, []))
            if values.shape != (n,):
                errors.append(f"{split}.{field} shape must be {(n,)}")
        classes = np.asarray(section.get("classification_labels", []), dtype=float)
        if classes.size and np.any(~np.isin(classes, [0.0, 1.0, 2.0])):
            errors.append(f"{split}.classification_labels contains values outside 0,1,2")
        intensities = np.asarray(section.get("regression_labels", []), dtype=float)
        if intensities.size and (np.any(intensities < -3.0) or np.any(intensities > 3.0)):
            errors.append(f"{split}.regression_labels contains values outside [-3,3]")
    if len(all_ids) != len(set(all_ids)):
        errors.append("sample IDs are not unique across splits")
    return ValidationReport(selected, split_sizes, tuple(errors))


def inspect_data_root(data_root: str | Path) -> dict[str, Any]:
    root = Path(data_root).resolve()
    attachment2 = root / "附件2-数据集特征文件"
    attachment3 = root / "附件3-模态缺失特征样本"
    attachment4 = root / "附件4-可解释专项视频样本与特征文件"
    return {
        "data_root": str(root),
        "attachment2_files": sorted(path.name for path in attachment2.glob("*.pkl")),
        "attachment3_aligned_files": len(list((attachment3 / "对齐版本").glob("*.pkl"))),
        "attachment3_unaligned_files": len(list((attachment3 / "未对齐版本").glob("*.pkl"))),
        "attachment4_aligned_files": len(
            list((attachment4 / "附件4-可解释专项视频样本与特征文件" / "对齐版本").glob("*.pkl"))
        ),
        "attachment4_unaligned_files": len(
            list((attachment4 / "附件4-可解释专项视频样本与特征文件" / "未对齐版本").glob("*.pkl"))
        ),
    }


def validate_data_root(data_root: str | Path) -> dict[str, Any]:
    reports: dict[str, Any] = {"inspection": inspect_data_root(data_root), "reports": {}}
    repository = Attachment2Repository(data_root)
    for variant in (DatasetVariant.ALIGNED, DatasetVariant.UNALIGNED):
        path = repository.path_for(variant)
        if not path.is_file():
            reports["reports"][variant.value] = {"errors": [f"missing file: {path}"]}
            continue
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        report = validate_attachment2_payload(payload, variant)
        reports["reports"][variant.value] = {
            "split_sizes": dict(report.split_sizes),
            "errors": list(report.errors),
        }
    missing = MissingModalityRepository(data_root)
    explain = ExplainabilityRepository(data_root)
    reports["special_tests"] = {
        "attachment3_aligned": sum(missing.path_for(i, "aligned").is_file() for i in range(1, 31)),
        "attachment3_unaligned": sum(missing.path_for(i, "unaligned").is_file() for i in range(1, 31)),
        "attachment4_aligned": sum(explain.path_for(i, "aligned").is_file() for i in range(1, 21)),
        "attachment4_unaligned": sum(explain.path_for(i, "unaligned").is_file() for i in range(1, 21)),
    }
    return reports


def inspect_processed_root(processed_root: str | Path) -> dict[str, Any]:
    """Inspect the canonical processed NPZ root without opening legacy PKL files."""

    root = Path(processed_root).expanduser().resolve()
    split_records: dict[str, Any] = {}
    for split in ("train", "valid", "test"):
        path = root / f"{split}.npz"
        split_records[split] = {
            "path": str(path),
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
        }
    return {
        "processed_root": str(root),
        "source_format": "npz",
        "feature_version": "aligned_50",
        "splits": split_records,
        "mask_semantics": {
            "text_token_padding": "P = ~Q",
            "text_content_observed": "mT (content positions; mT is a subset of Q)",
            "audio_native_valid": "mA (native availability; not a pure padding mask)",
            "vision_native_valid": "mV (native availability; not a pure padding mask)",
            "synthetic_missing": "generated separately for Q2",
            "observed": "native_valid_mask & ~synthetic_missing_mask",
        },
    }


def validate_processed_root(
    processed_root: str | Path,
    *,
    expected_split_sizes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Validate train/valid/test through the shared NPZ contract."""

    report = inspect_processed_root(processed_root)
    errors: list[str] = []
    try:
        dataset = load_processed_dataset(processed_root)
    except (OSError, ValueError) as exc:
        report["errors"] = [str(exc)]
        report["valid"] = False
        return report

    manifest = build_processed_manifest(dataset)
    split_sizes = manifest["split_sizes"]
    if expected_split_sizes is not None:
        for split, expected in expected_split_sizes.items():
            actual = split_sizes.get(split)
            if actual != int(expected):
                errors.append(f"{split} size must be {expected}, got {actual}")

    native_false_counts: dict[str, dict[str, int]] = {}
    observed_false_counts: dict[str, dict[str, int]] = {}
    for split_name in ("train", "valid", "test"):
        split = dataset[split_name]
        native_false_counts[split_name] = {
            modality: int((~split.native_valid_mask[modality]).sum())
            for modality in ("text", "audio", "vision")
        }
        observed_false_counts[split_name] = {
            modality: int((~split.observed_mask[modality]).sum())
            for modality in ("text", "audio", "vision")
        }

    report.update(
        {
            "valid": not errors,
            "errors": errors,
            "split_sizes": split_sizes,
            "field_schema": manifest["field_schema"],
            "data_hashes": manifest["data_hashes"],
            "native_invalid_counts": native_false_counts,
            "observed_invalid_counts": observed_false_counts,
            "sample_id_coverage": manifest["sample_id_coverage"],
        }
    )
    return report
