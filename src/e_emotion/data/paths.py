"""Safe resolution of project data paths."""

from __future__ import annotations

from pathlib import Path


class DataPathPolicy:
    """Resolve only paths contained by the configured data root."""

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root).resolve()

    def resolve(self, relative_path: str | Path) -> Path:
        candidate = Path(relative_path)
        if not candidate.is_absolute():
            candidate = self.data_root / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.data_root)
        except ValueError as exc:
            raise ValueError(f"path escapes data root: {relative_path}") from exc
        return resolved

    def require_file(self, relative_path: str | Path) -> Path:
        resolved = self.resolve(relative_path)
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return resolved
