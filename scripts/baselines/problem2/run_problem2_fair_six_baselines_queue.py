"""Run the six direct and robustness baselines through the shared queue."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.baselines.problem2.queue_runner import jobs as queue_jobs
from scripts.baselines.problem2.queue_runner import run_queue


SECOND_WAVE_METHODS = ("concat_mlp", "early_fusion_gru", "ef_lstm", "mult", "p_rmf", "cmad")


def jobs() -> tuple[tuple[str, str, str, str, int], ...]:
    return tuple(
        (method_id, view, directory, seed)
        for method_id, view, directory, seed in queue_jobs(SECOND_WAVE_METHODS)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.gpu != 3:
        parser.error("this queue is restricted to physical GPU 3")
    return run_queue(SECOND_WAVE_METHODS, gpu=args.gpu, dry_run=args.dry_run, artifact_root=args.artifact_root)


if __name__ == "__main__":
    raise SystemExit(main())
