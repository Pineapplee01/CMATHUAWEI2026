"""Evaluate six verified baseline checkpoints on the aligned test-252 protocol."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from e_emotion.problem2_fair.artifacts import ArtifactStore
from e_emotion.problem2_fair.config import FAIR_SEEDS
from e_emotion.problem2_fair.registry import load_registry
from scripts.baselines.problem2.queue_runner import runtime_root, workspace_reference_root


METHODS = ("concat_mlp", "early_fusion_gru", "ef_lstm", "mult", "p_rmf", "cmad")


def jobs() -> tuple[tuple[str, int, Path], ...]:
    registry = load_registry(workspace_reference_root() / "memory" / "catalog" / "methods.yaml")
    store = ArtifactStore(registry=registry)
    return tuple((method_id, seed, store.run_dir(method_id, "aligned_po", seed)) for method_id in METHODS for seed in FAIR_SEEDS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=3, type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--artifact-root", type=Path)
    args = parser.parse_args(argv)
    if args.gpu != 3:
        parser.error("Baseline queues are restricted to physical GPU 3")
    runtime = runtime_root()
    reference = workspace_reference_root()
    registry = load_registry(reference / "memory" / "catalog" / "methods.yaml")
    store = ArtifactStore(args.artifact_root, registry=registry)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "3"
    env["PYTHONPATH"] = str(runtime / "src")
    env["E_EMOTION_BASELINE_REFERENCE_ROOT"] = str(reference)
    env["E_EMOTION_BASELINE_ARTIFACT_ROOT"] = str(store.root)
    for method_id in METHODS:
        for seed in FAIR_SEEDS:
            run_dir = store.run_dir(method_id, "aligned_po", seed)
            command = [
                sys.executable, "-m", "e_emotion", "baseline", "test252",
                "--method", method_id, "--seed", str(seed),
                "--artifact-root", str(store.root),
            ]
            if args.dry_run:
                print(" ".join(command), flush=True)
                continue
            verified = subprocess.run(
                [sys.executable, "-m", "e_emotion", "baseline", "verify", "--run-dir", str(run_dir)],
                cwd=runtime, env=env, capture_output=True, text=True,
            )
            if verified.returncode:
                raise RuntimeError(f"{method_id} seed {seed} is not a verified fair run: {verified.stdout}")
            result = subprocess.run(command, cwd=runtime, env=env)
            if result.returncode:
                raise RuntimeError(f"{method_id} seed {seed} test252 evaluation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
