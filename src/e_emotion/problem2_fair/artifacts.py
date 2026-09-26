"""Canonical artifact paths for Baseline runs."""

from __future__ import annotations

from pathlib import Path

from e_emotion.baselines.workspace import artifact_root
from e_emotion.problem2_fair.registry import MethodRegistry, load_registry


class ArtifactStore:
    """Keep method, view and seed path construction behind one interface."""

    def __init__(self, root: str | Path | None = None, *, registry: MethodRegistry | None = None):
        self.root = Path(root).expanduser().resolve() if root is not None else artifact_root()
        self.registry = registry or load_registry()

    def run_dir(self, method_id: str, view: str, seed: int) -> Path:
        record = self.registry.get(method_id)
        if view not in record.views:
            raise ValueError(f"{method_id} does not support view {view}")
        if seed not in (1, 2, 3):
            raise ValueError("Baseline fair runs use seeds 1, 2 and 3")
        return self.root / record.directory / view / f"seed-{seed}"

    def report_path(self, method_id: str, view: str) -> Path:
        return self.root / self.registry.method_directory(method_id) / view / "three-seed-report.json"

    def log_path(self, method_id: str, view: str, seed: int) -> Path:
        run = self.run_dir(method_id, view, seed)
        return run.parent / f"{run.name}.log"

    def history_dir(self, label: str) -> Path:
        return self.root / "history" / label


__all__ = ["ArtifactStore"]
