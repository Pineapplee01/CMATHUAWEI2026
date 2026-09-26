"""Shared sequential queue for Baseline Workspace Problem 2 methods."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

from e_emotion.problem2_fair.artifacts import ArtifactStore
from e_emotion.problem2_fair.config import FAIR_SEEDS
from e_emotion.problem2_fair.registry import load_registry
from e_emotion.problem2_fair.result_index import update_result_index


VIEWS = ("aligned_po", "unaligned_po")


def runtime_root() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root


def workspace_reference_root() -> Path:
    root = runtime_root()
    return root.parent if root.name == "runtime" else root / "references"


def jobs(method_ids: Iterable[str]) -> tuple[tuple[str, str, str, int], ...]:
    registry = load_registry(workspace_reference_root() / "memory" / "catalog" / "methods.yaml")
    return tuple(
        (method_id, view, registry.get(method_id).directory, seed)
        for method_id in method_ids
        for view in VIEWS
        for seed in FAIR_SEEDS
    )


def run_queue(method_ids: Iterable[str], *, gpu: int, dry_run: bool, artifact_root: Path | None = None) -> int:
    if gpu != 3:
        raise ValueError("Baseline queues are restricted to physical GPU 3")
    reference = workspace_reference_root()
    registry = load_registry(reference / "memory" / "catalog" / "methods.yaml")
    store = ArtifactStore(artifact_root, registry=registry)
    runtime = runtime_root()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "3"
    env["PYTHONPATH"] = str(runtime / "src")
    env["E_EMOTION_BASELINE_REFERENCE_ROOT"] = str(reference)
    env["E_EMOTION_BASELINE_ARTIFACT_ROOT"] = str(store.root)
    for method_id in method_ids:
        record = registry.get(method_id)
        if not record.is_runnable:
            raise ValueError(f"{method_id} is not runnable ({record.state})")
        for view in VIEWS:
            roots = []
            for seed in FAIR_SEEDS:
                run_dir = store.run_dir(method_id, view, seed)
                log_path = store.log_path(method_id, view, seed)
                command = [
                    sys.executable, "-m", "e_emotion", "baseline", "run",
                    "--method", method_id, "--view", view, "--seed", str(seed),
                    "--run-dir", str(run_dir), "--artifact-root", str(store.root),
                ]
                if dry_run:
                    print(" ".join(command), flush=True)
                    continue
                if run_dir.exists() or log_path.exists():
                    raise FileExistsError(f"refusing to overwrite existing run: {run_dir}")
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("x", encoding="utf-8") as stream:
                    result = subprocess.run(command, cwd=runtime, env=env, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"{method_id} {view} seed {seed} failed; inspect {log_path}")
                verified = subprocess.run(
                    [sys.executable, "-m", "e_emotion", "baseline", "verify", "--run-dir", str(run_dir)],
                    cwd=runtime, env=env, capture_output=True, text=True,
                )
                if verified.returncode:
                    raise RuntimeError(f"{method_id} {view} seed {seed} failed verification: {verified.stdout}")
                roots.append(run_dir)
                print(f"completed {method_id} {view} seed {seed}", flush=True)
            if dry_run:
                continue
            report = subprocess.run(
                [sys.executable, "-m", "e_emotion", "baseline", "report",
                 "--artifact-root", str(store.root),
                 *[item for root in roots for item in ("--run-dir", str(root))]],
                cwd=runtime, env=env, capture_output=True, text=True,
            )
            if report.returncode:
                raise RuntimeError(f"{method_id} {view} report failed: {report.stdout}")
            output = store.report_path(method_id, view)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(json.loads(report.stdout), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            update_result_index(roots, path=reference / "memory" / "results" / "problem2" / "index.csv")
            print(f"reported {method_id} {view}", flush=True)
    return 0


__all__ = ["VIEWS", "jobs", "run_queue", "runtime_root", "workspace_reference_root"]
