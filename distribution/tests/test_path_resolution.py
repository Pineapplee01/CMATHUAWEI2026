from __future__ import annotations

from pathlib import Path

from multimodal_emotion.shared.paths import (
    data_root,
    distribution_root,
    reference_root,
    results_root,
    resolve_distribution_path,
)


def test_default_roots_are_inside_the_distribution(monkeypatch) -> None:
    monkeypatch.delenv("CPMCM_HUAWEI2026_REFERENCE_ROOT", raising=False)
    root = distribution_root()

    assert data_root() == root / "data"
    assert results_root() == root / "results"
    assert reference_root() == root / "reference"


def test_external_asset_root_can_be_overridden(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CPMCM_HUAWEI2026_REFERENCE_ROOT", str(tmp_path))

    assert reference_root() == tmp_path
    assert resolve_distribution_path("reference/models/bert-base-uncased") == (
        tmp_path / "models" / "bert-base-uncased"
    )
