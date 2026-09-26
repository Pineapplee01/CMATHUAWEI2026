from __future__ import annotations

import json

import pytest

from e_emotion.problem2_fair import FAIR_SEEDS, Q2_V2_CONDITIONS
from e_emotion.problem2_fair.view_report import summarize_view
from e_emotion.cli.main import main


def _write_run(root, seed: int, *, view: str, model_input_view: str, mask_hash: str):
    root.mkdir(parents=True)
    (root / "metrics.json").write_text(json.dumps({
        "protocol_version": "problem2-fair-v1", "method": "aumdf", "view": view,
        "model_input_view": model_input_view,
        "seed": seed,
        "clean": {"accuracy": 0.5, "macro_f1": 0.5, "weighted_f1": 0.5, "mae": 0.5, "pearson": 0.5},
    }), encoding="utf-8")
    q2 = root / "q2_v2"
    q2.mkdir()
    (q2 / "metrics.json").write_text(json.dumps({
        "method": "aumdf", "view": view, "model_input_view": model_input_view, "seed": seed,
        "n_conditions": 64, "mask_sha256": mask_hash,
        "conditions": [{"combination": combination, "position": position, "requested_fraction": fraction,
                        "accuracy": 0.5, "macro_f1": 0.5, "weighted_f1": 0.5, "mae": 0.5, "pearson": 0.5}
                       for combination, position, fraction in Q2_V2_CONDITIONS],
    }), encoding="utf-8")


def test_view_report_discovers_three_aumdf_runs_and_labels_windowed_view(tmp_path):
    artifacts = tmp_path / "artifacts"
    for seed in FAIR_SEEDS:
        _write_run(artifacts / "AUMDF" / "unaligned_po" / f"seed-{seed}", seed,
                   view="unaligned_po", model_input_view="unaligned_windowed", mask_hash="a" * 64)

    report = summarize_view("unaligned_po", artifact_root=artifacts, methods=("aumdf",))

    assert report["view"] == "unaligned_po"
    assert report["mask_sha256"] == "a" * 64
    assert report["groups"]["unaligned_windowed"][0]["method"] == "aumdf"
    assert report["groups"]["unaligned_windowed"][0]["seed_count"] == 3
    assert report["groups"]["unaligned_windowed"][0]["source_view"] == "unaligned_po"
    assert report["groups"]["unaligned_windowed"][0]["model_input_view"] == "unaligned_windowed"


def test_view_report_requires_all_three_runs(tmp_path):
    artifacts = tmp_path / "artifacts"
    for seed in (1, 2):
        _write_run(artifacts / "AUMDF" / "aligned_po" / f"seed-{seed}", seed,
                   view="aligned_po", model_input_view="aligned_po", mask_hash="a" * 64)

    with pytest.raises(FileNotFoundError):
        summarize_view("aligned_po", artifact_root=artifacts, methods=("aumdf",))


def test_cli_view_report_rejects_metrics_only_runs(tmp_path, capsys):
    artifacts = tmp_path / "artifacts"
    for seed in FAIR_SEEDS:
        _write_run(artifacts / "AUMDF" / "unaligned_po" / f"seed-{seed}", seed,
                   view="unaligned_po", model_input_view="unaligned_windowed", mask_hash="a" * 64)

    assert main(["baseline", "report", "--view", "unaligned_po", "--method", "aumdf",
                 "--artifact-root", str(artifacts)]) == 2
    assert "verify" in capsys.readouterr().out
