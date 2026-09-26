#!/usr/bin/env python3
"""Run CMAD's teacher/student stages without publishing a partial run.

This launcher is deployed to the CMAD method directory on the GPU host.  It
keeps ``launch_status.json`` in ``runs/<run_id>`` as ``running`` or ``failed``
until both stages have finished successfully.  The Q2 evaluator therefore
cannot accidentally consume a teacher-only or interrupted run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def checked_run_dir(run_id: str) -> Path:
    if Path(run_id).name != run_id:
        raise ValueError("--run-id must be one directory name")
    runs = (ROOT / "runs").resolve()
    run_dir = (runs / run_id).resolve()
    if not run_dir.is_relative_to(runs):
        raise ValueError("--run-id must remain within CMAD/runs")
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {run_dir}")
    return run_dir


def write_status(run_dir: Path, **changes: object) -> None:
    status_path = run_dir / "launch_status.json"
    payload: dict[str, object] = {}
    if status_path.exists():
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    payload.update(changes)
    payload["updated"] = time.time()
    temporary = status_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(status_path)


def run_stage(
    *,
    stage: str,
    run_dir: Path,
    args: argparse.Namespace,
    environment: dict[str, str],
    teacher_checkpoint: Path | None = None,
) -> int:
    stage_dir = run_dir / stage
    command = [
        sys.executable,
        "-u",
        "competition_launcher.py",
        "--stage",
        stage,
        "--device",
        "cuda:0",
        "--epochs",
        str(args.epochs),
        "--seed",
        str(args.seed),
        "--run-dir",
        str(stage_dir),
    ]
    if teacher_checkpoint is not None:
        command.extend(["--teacher-checkpoint", str(teacher_checkpoint)])

    write_status(
        run_dir,
        state="running",
        stage=stage,
        command=command,
        stage_started=time.time(),
    )
    with (run_dir / f"{stage}.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    write_status(run_dir, stage_exit_code=result.returncode, stage_finished=time.time())
    return result.returncode


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    run_dir = checked_run_dir(args.run_id)
    run_dir.mkdir(parents=True)
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        PYTHONUNBUFFERED="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    write_status(
        run_dir,
        method="CMAD",
        run=args.run_id,
        seed=args.seed,
        gpu=args.gpu,
        state="running",
        stage="teacher",
        started=time.time(),
    )

    try:
        if run_stage(stage="teacher", run_dir=run_dir, args=args, environment=environment):
            write_status(run_dir, state="failed", finished=time.time())
            raise SystemExit(1)
        teacher_checkpoint = run_dir / "teacher" / "teacher_best.pt"
        if not teacher_checkpoint.is_file():
            write_status(run_dir, state="failed", failure="missing teacher_best.pt", finished=time.time())
            raise SystemExit(1)
        if run_stage(
            stage="student",
            run_dir=run_dir,
            args=args,
            environment=environment,
            teacher_checkpoint=teacher_checkpoint,
        ):
            write_status(run_dir, state="failed", finished=time.time())
            raise SystemExit(1)
    except BaseException as error:
        if not isinstance(error, SystemExit):
            write_status(run_dir, state="failed", failure=repr(error), finished=time.time())
        raise

    write_status(run_dir, state="completed", stage="student", exit_code=0, finished=time.time())


if __name__ == "__main__":
    main()
