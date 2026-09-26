"""Tracked compact summaries for verified Problem 2 runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from e_emotion.baselines.workspace import memory_root
from e_emotion.problem2_fair.verification import verify_run


_FIELDS = (
    "method_id", "method_directory", "view", "seed", "status", "run_path",
    "checkpoint_sha256", "mask_sha256", "accuracy", "macro_f1", "weighted_f1",
    "mae", "pearson", "summary_path",
)


def result_index_path() -> Path:
    return memory_root() / "results" / "problem2" / "index.csv"


def _row(run_dir: str | Path) -> dict[str, str]:
    verified = verify_run(run_dir, require_source=True)
    root = Path(verified["run"])
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    clean = metrics["clean"]
    return {
        "method_id": str(metrics["method"]),
        "method_directory": root.parents[1].name,
        "view": str(metrics["view"]),
        "seed": str(metrics["seed"]),
        "status": "verified",
        "run_path": str(root),
        "checkpoint_sha256": str(metrics["checkpoint_sha256"]),
        "mask_sha256": str(metrics["q2"]["mask_sha256"]),
        "accuracy": str(clean["accuracy"]),
        "macro_f1": str(clean["macro_f1"]),
        "weighted_f1": str(clean["weighted_f1"]),
        "mae": str(clean["mae"]),
        "pearson": "" if clean["pearson"] is None else str(clean["pearson"]),
        "summary_path": str(root / "metrics.json"),
    }


def update_result_index(run_dirs: Iterable[str | Path], *, path: str | Path | None = None) -> Path:
    target = Path(path) if path is not None else result_index_path()
    rows: dict[str, dict[str, str]] = {}
    if target.is_file():
        with target.open(encoding="utf-8", newline="") as stream:
            rows = {row["run_path"]: row for row in csv.DictReader(stream)}
    for run_dir in run_dirs:
        row = _row(run_dir)
        rows[row["run_path"]] = row
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=lambda row: (row["method_directory"], row["view"], int(row["seed"]))))
    return target


__all__ = ["result_index_path", "update_result_index"]
