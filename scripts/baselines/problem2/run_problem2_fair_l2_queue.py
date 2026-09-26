"""Run L2 methods through the Baseline Workspace queue."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.baselines.problem2.queue_runner import jobs as queue_jobs
from scripts.baselines.problem2.queue_runner import run_queue

METHODS = ("tfn", "mfn", "graph_mfn_dfg")


def jobs() -> tuple[tuple[str, str, str, int], ...]:
    return queue_jobs(METHODS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=3, type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--artifact-root", type=Path)
    args = parser.parse_args(argv)
    return run_queue(METHODS, gpu=args.gpu, dry_run=args.dry_run, artifact_root=args.artifact_root)


if __name__ == "__main__":
    raise SystemExit(main())
