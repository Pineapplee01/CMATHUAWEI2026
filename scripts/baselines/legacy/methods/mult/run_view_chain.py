"""Run aligned-control and raw-unaligned MulT Q2 stages serially on one GPU."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
PYTHON = Path("/user_home/gaojianan/CPMCM/Baseline/reference/baseline/bin/python")
ALIGNED_RUN = "strict_npz_maskaware_seed2026_v1"
RAW_RUN = "strict_raw_maskaware_seed2026_v1"


def save_status(path: Path, **changes: object) -> None:
    payload: dict[str, object] = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(changes, updated=time.time())
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def wait_for_run(run: Path, status_path: Path) -> None:
    while True:
        if not run.is_file():
            time.sleep(20)
            continue
        state = json.loads(run.read_text(encoding="utf-8")).get("status")
        save_status(status_path, stage="waiting_aligned_training", aligned_status=state)
        if state == "completed":
            return
        if state == "failed":
            raise RuntimeError("aligned_50_control training failed")
        time.sleep(20)


def q2_complete(path: Path) -> bool:
    metrics = path / "q2_strict_test_maskaware_v1" / "metrics.json"
    if not metrics.is_file():
        return False
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    return payload.get("n_conditions") == 63 and payload.get("n_predictions") == 45801


def run(command: list[str], environment: dict[str, str], status_path: Path, stage: str) -> None:
    save_status(status_path, stage=stage, command=command, state="running")
    result = subprocess.run(command, cwd=ROOT, env=environment)
    if result.returncode:
        save_status(status_path, stage=stage, state="failed", exit_code=result.returncode)
        raise RuntimeError(f"{stage} failed with exit code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    status_path = ROOT / "runs" / "view_chain_status.json"
    environment = dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    aligned_root = ROOT / "runs" / "aligned_50_control" / ALIGNED_RUN
    raw_root = ROOT / "runs" / "raw_unaligned" / RAW_RUN
    save_status(status_path, state="waiting", gpu=args.gpu, aligned_run=ALIGNED_RUN, raw_run=RAW_RUN, started=time.time())
    wait_for_run(aligned_root / "status.json", status_path)
    if not q2_complete(aligned_root):
        run([str(PYTHON), "-u", "q2_evaluate_strict.py", "--data-view", "aligned_50_control", "--run", ALIGNED_RUN, "--device", "cuda:0"], environment, status_path, "aligned_q2")
    if not (raw_root / "status.json").exists():
        run([str(PYTHON), "-u", "cpmcm_run.py", "--data-view", "raw_unaligned", "--run", RAW_RUN, "--seed", str(args.seed), "--epochs", str(args.epochs)], environment, status_path, "raw_training")
    raw_status = json.loads((raw_root / "status.json").read_text(encoding="utf-8")).get("status")
    if raw_status != "completed":
        raise RuntimeError(f"raw_unaligned training did not complete: {raw_status}")
    if not q2_complete(raw_root):
        run([str(PYTHON), "-u", "q2_evaluate_strict.py", "--data-view", "raw_unaligned", "--run", RAW_RUN, "--device", "cuda:0"], environment, status_path, "raw_q2")
    save_status(status_path, state="completed", stage="completed", finished=time.time())


if __name__ == "__main__":
    main()
