"""Helpers for reading migration manifests."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    size_bytes: int
    sha256: str


def read_manifest(path: str | Path) -> list[ManifestEntry]:
    """Read a tab-separated migration manifest."""
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="	")
        return [
            ManifestEntry(
                path=row["path"],
                size_bytes=int(row["size_bytes"]),
                sha256=row["sha256"],
            )
            for row in reader
        ]
