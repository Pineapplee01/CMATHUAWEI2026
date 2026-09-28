"""Configuration loading and project path policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import yaml

from e_emotion.contracts import DatasetVariant


class ConfigError(ValueError):
    """Raised when a project configuration is invalid."""


@dataclass(frozen=True)
class AppConfig:
    """Resolved configuration shared by all task domains."""

    config_path: Path
    project_root: Path
    data_root: Path
    processed_root: Path
    split_files: Mapping[str, str]
    feature_version: str
    source_format: str
    output_root: Path
    variant: DatasetVariant | None
    seed: int

    def require_variant(self) -> DatasetVariant:
        if self.variant is None:
            raise ConfigError("dataset.variant must be explicitly set for this command")
        return self.variant

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "project_root": str(self.project_root),
            "data_root": str(self.data_root),
            "processed_root": str(self.processed_root),
            "split_files": dict(self.split_files),
            "feature_version": self.feature_version,
            "source_format": self.source_format,
            "output_root": str(self.output_root),
            "variant": self.variant.value if self.variant else None,
            "seed": self.seed,
        }


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def load_config(path: str | Path) -> AppConfig:
    """Load YAML configuration and enforce the data-root boundary."""

    config_path = Path(path).resolve()
    if not config_path.exists():
        raise ConfigError(f"configuration file does not exist: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML: {config_path}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a mapping")

    project_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    dataset = raw.get("dataset", {})
    if not isinstance(dataset, dict):
        raise ConfigError("dataset must be a mapping")

    data_root = _resolve_path(project_root, dataset.get("data_root", "data/raw"))
    allowed_data_root = (project_root / "data").resolve()
    if not _is_within(data_root, allowed_data_root):
        raise ConfigError(f"dataset.data_root must be inside {allowed_data_root}")

    processed_root = _resolve_path(project_root, dataset.get("processed_root", "data/processed"))
    source_format = str(dataset.get("source_format", "npz")).lower()
    if source_format != "npz":
        raise ConfigError("dataset.source_format must be npz for the processed-data protocol")
    feature_version = str(dataset.get("feature_version", "aligned_50"))
    if feature_version not in {"aligned_50", "unaligned_50"}:
        raise ConfigError("dataset.feature_version must be aligned_50 or unaligned_50")
    raw_split_files = dataset.get(
        "split_files", {"train": "train.npz", "valid": "valid.npz", "test": "test.npz"}
    )
    if not isinstance(raw_split_files, Mapping) or set(raw_split_files) != {"train", "valid", "test"}:
        raise ConfigError("dataset.split_files must define train, valid and test")
    split_files: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        value = raw_split_files[split]
        if not isinstance(value, str) or Path(value).name != value or not value.endswith(".npz"):
            raise ConfigError(f"dataset.split_files.{split} must be a simple .npz filename")
        split_files[split] = value

    raw_variant = dataset.get("variant")
    if raw_variant is None or raw_variant == "":
        variant = None
    else:
        try:
            variant = DatasetVariant(str(raw_variant).lower())
        except ValueError as exc:
            raise ConfigError("dataset.variant must be aligned or unaligned") from exc

    output_root = _resolve_path(project_root, raw.get("output_root", "artifacts"))
    if not _is_within(output_root, (project_root / "artifacts").resolve()):
        raise ConfigError("output_root must be inside artifacts/")

    try:
        seed = int(raw.get("seed", 2026))
    except (TypeError, ValueError) as exc:
        raise ConfigError("seed must be an integer") from exc

    return AppConfig(
        config_path=config_path,
        project_root=project_root,
        data_root=data_root,
        processed_root=processed_root,
        split_files=split_files,
        feature_version=feature_version,
        source_format=source_format,
        output_root=output_root,
        variant=variant,
        seed=seed,
    )
