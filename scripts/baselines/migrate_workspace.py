"""One-time migration of a legacy Baseline/reference tree."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
from typing import Iterable


VENDOR_DIRECTORIES = (
    "CACR", "CaReFlow", "CMAD", "CMU-MultimodalSDK", "DiscRead-MSA", "EBMC",
    "EMOE", "HyperDiff", "MFN", "MMRest", "MMSA", "MRUF",
    "Multimodal-Transformer", "P-RMF", "QA-MoE",
    "quality-aware-fusion-diagnostic", "SentiLLM", "TLRA", "VG-TPT",
)
PROJECT_DIRECTORIES = (
    "AUMDF", "Concat-MLP", "Early-Fusion-GRU", "EF-LSTM", "MulT", "TFN",
    "problem2_fair", "protocol_src", "protocol_tests",
)
ARTIFACT_CHILDREN = ("runs", "results", "artifacts", "log", "logs", "ckpt", "pt")
SUPPORT_DIRECTORIES = ("_data", "data_views")


def _directory_summary(path: Path) -> dict:
    files = [item for item in path.rglob("*") if item.is_file()]
    return {"files": len(files), "bytes": sum(item.stat().st_size for item in files)}


def _git_state(path: Path) -> dict | None:
    if not (path / ".git").exists():
        return None
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
    return {"commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"), "origin": git("remote", "get-url", "origin")}


def _move(source: Path, target: Path, *, dry_run: bool) -> None:
    if not source.exists():
        return
    if target.exists():
        raise FileExistsError(f"migration target already exists: {target}")
    print(f"{source} -> {target}")
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))


def _extract_artifacts(source: Path, history: Path, *, dry_run: bool) -> None:
    for name in ARTIFACT_CHILDREN:
        child = source / name
        if child.exists():
            _move(child, history / source.name / name, dry_run=dry_run)


def _snapshot(reference: Path) -> dict:
    vendor = {}
    for name in VENDOR_DIRECTORIES:
        source = reference / name
        if source.is_dir():
            vendor[name] = {"summary": _directory_summary(source), "git": _git_state(source)}
    artifacts = {}
    for name in (*VENDOR_DIRECTORIES, *PROJECT_DIRECTORIES, *SUPPORT_DIRECTORIES):
        source = reference / name
        if source.is_dir():
            artifacts[name] = {
                child: _directory_summary(source / child)
                for child in ARTIFACT_CHILDREN
                if (source / child).is_dir()
            }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_root": str(reference),
        "vendor": vendor,
        "artifacts": artifacts,
    }


def migrate(reference: Path, *, dry_run: bool) -> None:
    baseline = reference.parent
    memory = reference / "memory"
    vendor = reference / "vendor"
    runtime = reference / "runtime"
    history = baseline / "artifacts" / "problem2" / "history" / "legacy-202609"
    snapshot = _snapshot(reference)
    if dry_run:
        print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    else:
        memory.mkdir(parents=True, exist_ok=True)
        (memory / "history").mkdir(parents=True, exist_ok=True)
        (memory / "history" / "migration-preflight.json").write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    environment = reference / "baseline"
    if environment.exists():
        _move(environment, baseline / ".venv", dry_run=dry_run)
    conda = reference / ".conda"
    if conda.exists():
        _move(conda, baseline / ".conda", dry_run=dry_run)

    for name in VENDOR_DIRECTORIES:
        source = reference / name
        if source.exists():
            _extract_artifacts(source, history, dry_run=dry_run)
            _move(source, vendor / name, dry_run=dry_run)

    for name in PROJECT_DIRECTORIES:
        source = reference / name
        if source.exists():
            _extract_artifacts(source, history, dry_run=dry_run)
            _move(source, runtime / "legacy" / name, dry_run=dry_run)

    for name in SUPPORT_DIRECTORIES:
        source = reference / name
        if source.exists():
            _move(source, history / "support" / name, dry_run=dry_run)

    keep = {"README.md", "memory", "runtime", "vendor"}
    for source in list(reference.iterdir()):
        if source.name in keep:
            continue
        if source.name in {".pytest_cache", "__pycache__"}:
            _move(source, history / "cache" / source.name, dry_run=dry_run)
        elif source.is_dir():
            _move(source, memory / "history" / "root-directories" / source.name, dry_run=dry_run)
        elif source.suffix in {".py", ".sh"}:
            _move(source, runtime / "legacy" / "root-scripts" / source.name, dry_run=dry_run)
        elif source.suffix in {".log", ".pt", ".pth", ".ckpt", ".npz", ".pkl"}:
            _move(source, history / "root-files" / source.name, dry_run=dry_run)
        else:
            _move(source, memory / "history" / "root-files" / source.name, dry_run=dry_run)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, default=Path("/user_home/gaojianan/CPMCM/Baseline/reference"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    reference = args.reference_root.resolve()
    if not reference.is_dir():
        raise FileNotFoundError(reference)
    migrate(reference, dry_run=not args.execute)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
