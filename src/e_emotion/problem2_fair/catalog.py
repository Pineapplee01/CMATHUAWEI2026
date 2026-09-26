"""Compatibility facade over the Baseline Workspace method registry."""

from __future__ import annotations

from pathlib import Path

from e_emotion.problem2_fair.registry import MethodRecord, load_registry


MethodSpec = MethodRecord


def method_spec(method_id: str) -> MethodSpec:
    return load_registry().get(method_id)


def active_method_specs() -> tuple[MethodSpec, ...]:
    return load_registry().runnable()


def all_method_specs() -> tuple[MethodSpec, ...]:
    return load_registry().all()


def is_frozen_output_path(path: str | Path) -> bool:
    """The workspace has no per-method frozen output directories."""
    del path
    return False


__all__ = ["MethodSpec", "active_method_specs", "all_method_specs", "is_frozen_output_path", "method_spec"]
