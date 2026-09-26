#!/usr/bin/env python3
"""Run the two position-mask repair reruns serially on physical GPU 3.

GPU 3 is intentionally shared with the existing service requested by the
operator. P-RMF training and Q2 complete before CMAD teacher/student training
and Q2 begin. Each child receives physical GPU 3 through
``CUDA_VISIBLE_DEVICES`` and uses ``cuda:0`` inside its namespace.
"""
from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
import shutil
import subprocess


ROOT = Path("/user_home/gaojianan/CPMCM/Baseline/reference")
PYTHON = ROOT / "baseline" / "bin" / "python"
PRMF_RUN = "strict_npz_maskaware_stable_seed2026_v1"
CMAD_RUN = "strict_npz_maskaware_stable_seed2026_v1"
SEED = 2026
TARGET_GPU = 3
MASK_SHA256 = "bddfc02985e9528bcab58a252ebb9beaa8c6a9812b609b6b88c7a547fd623916"


def select_gpu(label: str) -> int:
    """Return the operator-selected GPU without an exclusivity check."""
    print(f"{label}: using shared physical GPU {TARGET_GPU}", flush=True)
    return TARGET_GPU


def child_environment(gpu: int) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHONUNBUFFERED="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    return environment


def run(command: list[str], *, cwd: Path, gpu: int) -> None:
    print("running:", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=cwd, env=child_environment(gpu))
    if result.returncode:
        raise RuntimeError(f"child failed with exit code {result.returncode}: {command[2]}")


def ensure_q2_manifest(run_dir: Path, source_dir: Path) -> None:
    def digest(path: Path) -> str:
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(block)
        return value.hexdigest()

    filenames = ("q2_test_manifest_index.json", "q2_mask_manifest.json", "q2_mask_sha256.txt")
    source_index = json.loads((source_dir / filenames[0]).read_text(encoding="utf-8"))
    source_hash = (source_dir / filenames[2]).read_text(encoding="utf-8").strip()
    if source_index.get("mask_sha256") != MASK_SHA256 or source_hash != MASK_SHA256:
        raise RuntimeError("shared Q2 manifest hash does not match the fixed protocol")
    for filename in filenames:
        target = run_dir / filename
        if target.exists():
            if digest(target) != digest(source_dir / filename):
                raise RuntimeError(f"existing Q2 manifest differs: {target}")
            continue
        shutil.copy2(source_dir / filename, target)


def completed_training(run_dir: Path) -> bool:
    status_path = run_dir / "status.json"
    if not status_path.exists():
        return False
    status = json.loads(status_path.read_text(encoding="utf-8"))
    return status.get("status") == "completed" and all(
        (run_dir / filename).is_file()
        for filename in ("pt/best.pth", "neutral_interval.json", "metrics.json")
    )


def completed_q2(run_dir: Path) -> bool:
    metric_path = run_dir / "q2_strict_test_maskaware_v1" / "metrics.json"
    if not metric_path.is_file():
        return False
    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    return metrics.get("mask_sha256") == MASK_SHA256 and len(metrics.get("conditions", [])) == 63


def completed_cmad_training(run_dir: Path) -> bool:
    status_path = run_dir / "launch_status.json"
    if not status_path.is_file():
        return False
    status = json.loads(status_path.read_text(encoding="utf-8"))
    return (
        status.get("state") == "completed"
        and status.get("stage") == "student"
        and status.get("exit_code") == 0
        and all((run_dir / filename).is_file() for filename in (
            "teacher/teacher_best.pt", "student/student_best.pt",
            "student/metrics.json", "student/neutral_interval.json",
        ))
    )


def ensure_cmad_q2_binding(run_dir: Path, source_dir: Path) -> None:
    source = source_dir / "q2_mask_manifest.json"
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if manifest.get("mask_sha256") != MASK_SHA256:
        raise RuntimeError("CMAD source Q2 manifest hash does not match the fixed protocol")
    target = run_dir / source.name
    if target.exists():
        if target.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"existing CMAD Q2 manifest differs: {target}")
    else:
        shutil.copy2(source, target)


def main() -> None:
    prmf = ROOT / "P-RMF"
    cmad = ROOT / "CMAD"

    gpu = select_gpu("P-RMF training")
    prmf_run = prmf / "runs" / PRMF_RUN
    if completed_training(prmf_run):
        print(f"P-RMF training already completed: {prmf_run}", flush=True)
    else:
        if prmf_run.exists():
            raise RuntimeError(f"P-RMF run exists but is incomplete: {prmf_run}")
        run(
            [str(PYTHON), "-u", "cpmcm_run.py", "--run", PRMF_RUN, "--seed", str(SEED), "--max-epochs", "60"],
            cwd=prmf,
            gpu=gpu,
        )
    ensure_q2_manifest(prmf_run, prmf / "runs" / "strict_npz_repair_seed2026_v1")
    gpu = select_gpu("P-RMF Q2")
    if completed_q2(prmf_run):
        print(f"P-RMF Q2 already completed: {prmf_run}", flush=True)
    else:
        run(
            [str(PYTHON), "-u", "q2_evaluate_strict.py", "--run", PRMF_RUN, "--device", "cuda:0", "--output-name", "q2_strict_test_maskaware_v1"],
            cwd=prmf,
            gpu=gpu,
        )
    if not (prmf_run / "comparison_vs_pre_maskaware.json").is_file():
        run(
        [
            str(PYTHON),
            "-u",
            "compare_maskaware_results.py",
            "--method",
            "P-RMF",
            "--legacy-train",
            "runs/strict_npz_repair_seed2026_v1/metrics.json",
            "--mask-aware-train",
            f"runs/{PRMF_RUN}/metrics.json",
            "--legacy-q2",
            "runs/strict_npz_repair_seed2026_v1/q2_strict_test_v1/metrics.json",
            "--mask-aware-q2",
            f"runs/{PRMF_RUN}/q2_strict_test_maskaware_v1/metrics.json",
            "--output",
            f"runs/{PRMF_RUN}/comparison_vs_pre_maskaware.json",
        ],
        cwd=prmf,
        gpu=gpu,
        )

    gpu = select_gpu("CMAD training")
    cmad_run = cmad / "runs" / CMAD_RUN
    if completed_cmad_training(cmad_run):
        print(f"CMAD training already completed: {cmad_run}", flush=True)
    else:
        if cmad_run.exists():
            raise RuntimeError(f"CMAD run exists but is incomplete: {cmad_run}")
        run(
            [str(PYTHON), "-u", "launch_cmad_maskaware.py", "--run-id", CMAD_RUN, "--gpu", str(gpu), "--epochs", "100", "--seed", str(SEED)],
            cwd=cmad,
            gpu=gpu,
        )
    ensure_cmad_q2_binding(cmad_run, cmad / "runs" / "strict_npz_seed2026_v1")
    gpu = select_gpu("CMAD Q2")
    if completed_q2(cmad_run):
        print(f"CMAD Q2 already completed: {cmad_run}", flush=True)
    else:
        run(
            [str(PYTHON), "-u", "q2_evaluate_strict.py", "--source-run", CMAD_RUN, "--device", "cuda:0", "--output-name", "q2_strict_test_maskaware_v1"],
            cwd=cmad,
            gpu=gpu,
        )
    if not (cmad_run / "comparison_vs_pre_maskaware.json").is_file():
        run(
        [
            str(PYTHON),
            "-u",
            "compare_maskaware_results.py",
            "--method",
            "CMAD",
            "--legacy-train",
            "runs/strict_npz_seed2026_v1/student/metrics.json",
            "--mask-aware-train",
            f"runs/{CMAD_RUN}/student/metrics.json",
            "--legacy-q2",
            "runs/strict_npz_seed2026_v1/q2_strict_test_v1/metrics.json",
            "--mask-aware-q2",
            f"runs/{CMAD_RUN}/q2_strict_test_maskaware_v1/metrics.json",
            "--output",
            f"runs/{CMAD_RUN}/comparison_vs_pre_maskaware.json",
        ],
        cwd=cmad,
        gpu=gpu,
        )
    print("Position-mask repair reproduction completed.", flush=True)


if __name__ == "__main__":
    main()
