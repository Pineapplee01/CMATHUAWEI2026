"""Cross-method reports within one Problem 2 data view and input layout."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from e_emotion.problem2_fair.artifacts import ArtifactStore
from e_emotion.problem2_fair.catalog import active_method_specs, method_spec
from e_emotion.problem2_fair.config import FAIR_SEEDS
from e_emotion.problem2_fair.report import summarize_runs
from e_emotion.problem2_fair.verification import verify_run


def summarize_view(
    view: str,
    *,
    artifact_root: Path | None = None,
    methods: Iterable[str] | None = None,
    verify: bool = False,
) -> dict:
    """Build a fair comparison without mixing native and windowed time layouts."""
    if view not in {"aligned_po", "unaligned_po"}:
        raise ValueError(f"unknown fair data view: {view!r}")
    selected = tuple(methods) if methods is not None else tuple(spec.method_id for spec in active_method_specs())
    if not selected:
        raise ValueError("at least one method is required")
    store = ArtifactStore(artifact_root)
    groups: dict[str, list[dict]] = {view: []}
    if view == "unaligned_po":
        groups["unaligned_windowed"] = []
    mask_hash: str | None = None
    for method in selected:
        spec = method_spec(method)
        if spec.deferred or spec.status == "paper_only":
            raise ValueError(f"{method} is not part of the fair report")
        roots = [store.run_dir(method, view, seed) for seed in FAIR_SEEDS]
        for root in roots:
            if not root.is_dir():
                raise FileNotFoundError(root)
            if verify:
                try:
                    verify_run(root, require_source=True)
                except (OSError, ValueError) as exc:
                    raise ValueError(f"{root}: verify failed: {exc}") from exc
        report = summarize_runs(roots)
        if mask_hash is None:
            mask_hash = report["mask_sha256"]
        elif report["mask_sha256"] != mask_hash:
            raise ValueError(f"{method} Q2-v2 mask hash differs across methods in {view}")
        group = report["model_input_view"]
        if group not in groups:
            raise ValueError(f"{method} reports unsupported model input view {group!r} for source view {view!r}")
        groups[group].append(report)
    return {
        "protocol_version": "problem2-fair-v1",
        "view": view,
        "mask_sha256": mask_hash,
        "method_count": len(selected),
        "groups": groups,
    }


__all__ = ["summarize_view"]
