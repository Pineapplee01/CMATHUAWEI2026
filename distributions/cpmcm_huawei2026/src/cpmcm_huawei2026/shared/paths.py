"""Path helpers for the CPMCM Huawei Cup 2026 distribution."""

from __future__ import annotations

import os
from pathlib import Path

ENV_ASSET_ROOT = "CPMCM_HUAWEI2026_ASSET_ROOT"


def distribution_root() -> Path:
    """Return the root of this standalone distribution checkout."""
    return Path(__file__).resolve().parents[3]


def asset_root() -> Path:
    """Return the external asset root used for AAAdata/AAAmodel/AAAcheckpoints."""
    configured = os.environ.get(ENV_ASSET_ROOT)
    return Path(configured).expanduser().resolve() if configured else distribution_root()


def resolve_asset_path(value: str | Path) -> Path:
    """Resolve a project-relative data/model path against the external asset root."""
    path = Path(value)
    return path if path.is_absolute() else asset_root() / path


def relative_to_asset_root(value: str | Path) -> str:
    """Display a path relative to the external asset root when possible."""
    path = Path(value)
    try:
        return str(path.relative_to(asset_root()))
    except ValueError:
        return str(path)
