"""Method registry for the Baseline Workspace."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from e_emotion.baselines.workspace import memory_root, reference_root, vendor_root


@dataclass(frozen=True)
class MethodRecord:
    method_id: str
    directory: str
    display_name: str
    kind: str
    adapter: str | None
    input_layout: str
    views: tuple[str, ...]
    q2: str
    q3: str
    state: str
    vendor: str | None
    deferred: bool = False
    blocker: str | None = None

    @property
    def is_runnable(self) -> bool:
        return self.state == "ready" and self.adapter is not None

    @property
    def status(self) -> str:
        """Compatibility name used by the public runner."""
        return self.state


class MethodRegistry:
    """Load method identity and source provenance from one YAML interface."""

    def __init__(self, records: Iterable[MethodRecord], *, root: Path):
        self.root = root
        self._records = {record.method_id: record for record in records}

    @classmethod
    def load(cls, path: str | Path | None = None) -> "MethodRegistry":
        source = Path(path) if path is not None else memory_root() / "catalog" / "methods.yaml"
        with source.open(encoding="utf-8") as stream:
            payload: dict[str, Any] = yaml.safe_load(stream) or {}
        records = []
        for item in payload.get("methods", []):
            records.append(MethodRecord(
                method_id=str(item["id"]),
                directory=str(item["directory"]),
                display_name=str(item["display_name"]),
                kind=str(item["kind"]),
                adapter=item.get("adapter"),
                input_layout=str(item["input_layout"]),
                views=tuple(item.get("views", ())),
                q2=str(item.get("q2", "unavailable")),
                q3=str(item.get("q3", "unavailable")),
                state=str(item.get("state", "unknown")),
                vendor=item.get("vendor"),
                deferred=bool(item.get("deferred", False)),
                blocker=item.get("blocker"),
            ))
        if len({record.method_id for record in records}) != len(records):
            raise ValueError("method registry contains duplicate method IDs")
        return cls(records, root=source.parent.parent)

    def get(self, method_id: str) -> MethodRecord:
        try:
            return self._records[method_id]
        except KeyError as exc:
            raise KeyError(f"unknown Baseline method: {method_id}") from exc

    def all(self) -> tuple[MethodRecord, ...]:
        return tuple(self._records.values())

    def runnable(self) -> tuple[MethodRecord, ...]:
        return tuple(record for record in self.all() if record.is_runnable)

    def vendor_path(self, method_id: str) -> Path | None:
        vendor = self.get(method_id).vendor
        return None if vendor is None else vendor_root() / vendor

    def method_directory(self, method_id: str) -> str:
        return self.get(method_id).directory


def load_registry(path: str | Path | None = None) -> MethodRegistry:
    return MethodRegistry.load(path)


__all__ = ["MethodRecord", "MethodRegistry", "load_registry", "memory_root", "reference_root", "vendor_root"]
