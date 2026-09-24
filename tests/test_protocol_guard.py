import json

import pytest

from e_emotion.evaluation.protocol_guard import (
    build_downstream_manifest,
    validate_downstream_manifest,
)
from e_emotion.data import load_processed_dataset, sha256_file

from tests.test_processed_data import _write_split


def _dataset(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    for split_name, ids in (("train", ["tr1", "tr2"]), ("valid", ["va"]), ("test", ["te"])):
        _write_split(tmp_path / f"{split_name}.npz", ids)
    return load_processed_dataset(tmp_path)


def _valid_kwargs(tmp_path):
    method_root = tmp_path / "artifacts" / "demo"
    artifact_root = method_root / "run-1"
    artifact_root.mkdir(parents=True)
    return {
        "method": "demo",
        "method_root": method_root,
        "artifact_root": artifact_root,
        "experiment_seed": 2026,
        "mask_seed": 2026,
        "normalization_source": "organizer_processed_train_fitted",
        "checkpoint": artifact_root / "checkpoint.pt",
        "threshold_source": artifact_root / "threshold.json",
        "mask_manifest_hash": "a" * 64,
    }


def test_valid_downstream_manifest_is_json_serializable_and_complete(tmp_path):
    dataset = _dataset(tmp_path / "processed")
    manifest = build_downstream_manifest(dataset, **_valid_kwargs(tmp_path))

    assert manifest["protocol_version"] == "e-competition-v1"
    assert manifest["feature_version"] == "aligned_50"
    assert manifest["split_sizes"] == {"train": 2, "valid": 1, "test": 1}
    assert manifest["mask_manifest_hash"] == "a" * 64
    assert manifest["method"] == "demo"
    assert manifest["artifact_root"].endswith("artifacts\\demo\\run-1") or manifest["artifact_root"].endswith("artifacts/demo/run-1")
    assert set(manifest["data_hashes"]) == {"train", "valid", "test"}
    assert manifest["mask_semantics"]["native"] == "mT/mA/mV"
    assert manifest["mask_semantics"]["synthetic"]
    assert manifest["mask_semantics"]["observed"]
    json.dumps(manifest)


def test_rejects_legacy_pkl_input(tmp_path):
    legacy = tmp_path / "aligned_50.pkl"
    legacy.write_bytes(b"legacy")
    with pytest.raises(ValueError, match="NPZ|legacy|pkl"):
        build_downstream_manifest(legacy, **_valid_kwargs(tmp_path))


def test_rejects_missing_or_mismatched_hashes(tmp_path):
    processed = tmp_path / "processed"
    dataset = _dataset(processed)
    kwargs = _valid_kwargs(tmp_path)
    expected = {name: sha256_file(processed / f"{name}.npz") for name in ("train", "valid", "test")}
    expected["valid"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        build_downstream_manifest(dataset, expected_hashes=expected, **kwargs)


def test_rejects_wrong_split_sizes(tmp_path):
    dataset = _dataset(tmp_path / "processed")
    with pytest.raises(ValueError, match="split size"):
        build_downstream_manifest(
            dataset,
            expected_split_sizes={"train": 99, "valid": 1, "test": 1},
            **_valid_kwargs(tmp_path),
        )


def test_rejects_missing_mask_provenance(tmp_path):
    dataset = _dataset(tmp_path / "processed")
    kwargs = _valid_kwargs(tmp_path)
    kwargs.pop("mask_manifest_hash")
    with pytest.raises(ValueError, match="mask"):
        build_downstream_manifest(dataset, **kwargs)


def test_rejects_artifact_root_outside_method_root(tmp_path):
    dataset = _dataset(tmp_path / "processed")
    kwargs = _valid_kwargs(tmp_path)
    kwargs["artifact_root"] = tmp_path / "artifacts" / "other-method"
    with pytest.raises(ValueError, match="artifact root|method"):
        build_downstream_manifest(dataset, **kwargs)


def test_manifest_validator_rejects_removed_hash(tmp_path):
    dataset = _dataset(tmp_path / "processed")
    manifest = build_downstream_manifest(dataset, **_valid_kwargs(tmp_path))
    manifest["data_hashes"].pop("test")
    with pytest.raises(ValueError, match="hash"):
        validate_downstream_manifest(manifest)
