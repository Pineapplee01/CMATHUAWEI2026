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
