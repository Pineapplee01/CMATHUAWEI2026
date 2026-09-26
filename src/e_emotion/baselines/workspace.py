"""Filesystem seams for the Baseline Workspace."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def reference_root() -> Path:
    """Return the memory and vendor root for local or server execution."""
    configured = os.environ.get("E_EMOTION_BASELINE_REFERENCE_ROOT")
    return Path(configured).expanduser().resolve() if configured else PROJECT_ROOT / "references"


def vendor_root() -> Path:
    return reference_root() / "vendor"


def memory_root() -> Path:
    return reference_root() / "memory"


def artifact_root() -> Path:
    configured = os.environ.get("E_EMOTION_BASELINE_ARTIFACT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return reference_root().parent / "artifacts" / "problem2"


def bert_root() -> Path:
    configured = os.environ.get("E_EMOTION_BERT_ROOT")
    return Path(configured).expanduser().resolve() if configured else Path(
        "/user_home/gaojianan/CPMCM/AAAmodel/bert-base-uncased"
    )


__all__ = ["PROJECT_ROOT", "artifact_root", "bert_root", "memory_root", "reference_root", "vendor_root"]
