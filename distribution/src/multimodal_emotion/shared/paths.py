"""Path helpers for the CPMCM Huawei Cup 2026 distribution."""

from __future__ import annotations

import os
from pathlib import Path

ENV_REFERENCE_ROOT = "CPMCM_HUAWEI2026_REFERENCE_ROOT"

__all__ = [
    "ENV_REFERENCE_ROOT",
    "data_root",
    "distribution_root",
    "reference_root",
    "results_root",
    "relative_to_distribution_root",
    "resolve_distribution_path",
]


def distribution_root() -> Path:
    """Return the root of this standalone distribution checkout."""
    return Path(__file__).resolve().parents[3]


def data_root() -> Path:
    """Return the bundled, immutable contest-data directory."""
    return distribution_root() / "data"


def results_root() -> Path:
    """Return the directory reserved for data and results generated at runtime."""
    return distribution_root() / "results"


def reference_root() -> Path:
    """Return the local or user-configured root for excluded external assets."""
    configured = os.environ.get(ENV_REFERENCE_ROOT)
    return (
        Path(configured).expanduser().resolve()
        if configured
        else distribution_root() / "reference"
    )


def resolve_distribution_path(value: str | Path) -> Path:
    """Resolve a distribution-relative path without changing the current directory."""
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "reference":
        return reference_root().joinpath(*path.parts[1:])
    return distribution_root() / path


def relative_to_distribution_root(value: str | Path) -> str:
    """Display a path relative to the distribution root when possible."""
    path = Path(value)
    try:
        return str(path.relative_to(distribution_root()))
    except ValueError:
        return str(path)
