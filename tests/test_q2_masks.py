import json
import copy

import numpy as np
import pytest

from e_emotion.data.processed import ProcessedDataset, ProcessedSplit
from e_emotion.robustness.q2_masks import (
    Q2_COMBINATIONS,
    Q2_FRACTIONS,
    Q2_POSITIONS,
    build_q2_mask_manifest,
    load_q2_mask_manifest,
    save_q2_mask_manifest,
    validate_q2_mask_manifest,
)


def _dataset() -> ProcessedDataset:
    n, length = 2, 8
    native = {
        "text": np.array([[1, 1, 1, 1, 1, 1, 0, 0]] * n, dtype=bool),
        "audio": np.array([[1, 1, 0, 1, 1, 1, 0, 0]] * n, dtype=bool),
        "vision": np.array([[1, 1, 1, 0, 1, 1, 0, 0]] * n, dtype=bool),
    }
    split = ProcessedSplit(
        features={m: np.zeros((n, length, 1), dtype=np.float32) for m in native},
        native_valid_mask=native,
        observed_mask={m: value.copy() for m, value in native.items()},
        ids=np.asarray(["a", "b"]),
        regression=np.zeros(n, dtype=np.float32),
        classification=np.ones(n, dtype=np.int64),
        source_path="synthetic.npz",
    )
    return ProcessedDataset({"train": split, "valid": split, "test": split})


def test_q2_manifest_matrix_hash_and_json(tmp_path):
    first = build_q2_mask_manifest(_dataset())
    second = build_q2_mask_manifest(_dataset())
    assert first["mask_sha256"] == second["mask_sha256"]
    assert len(first["entries"]) == 3 * 2 * len(Q2_COMBINATIONS) * len(Q2_POSITIONS) * len(Q2_FRACTIONS)
    assert validate_q2_mask_manifest(first, dataset=_dataset())
    path = tmp_path / "q2.json"
    assert save_q2_mask_manifest(first, path) == first["mask_sha256"]
    assert load_q2_mask_manifest(path, dataset=_dataset())["mask_sha256"] == first["mask_sha256"]
    json.dumps(first)


def test_q2_manifest_coordinates_and_native_separation():
    manifest = build_q2_mask_manifest(_dataset())
    one_position_seen = False
    for entry in manifest["entries"]:
        for modality in ("text", "audio", "vision"):
            assert entry["synthetic_missing_counts"][modality] <= entry["coordinate_lengths"][modality]
            if entry["synthetic_missing_counts"][modality] == 1:
                one_position_seen = True
                assert entry["synthetic_end_positions"][modality] == entry["synthetic_start_positions"][modality] + 1
            if entry["combination"] == "complete":
                assert entry["synthetic_missing_counts"][modality] == 0
                assert entry["synthetic_start_positions"][modality] is None
                assert entry["synthetic_end_positions"][modality] is None
    assert one_position_seen
    assert validate_q2_mask_manifest(manifest, dataset=_dataset())


def test_q2_manifest_rejects_incomplete_matrix_without_dataset():
    manifest = build_q2_mask_manifest(_dataset())
    tampered = copy.deepcopy(manifest)
    tampered["entries"].pop()
    # Recompute the hash to isolate the coverage check from hash validation.
    payload = dict(tampered)
    payload.pop("mask_sha256")
    import hashlib

    tampered["mask_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    try:
        validate_q2_mask_manifest(tampered)
    except ValueError as exc:
        assert "coverage" in str(exc)
    else:
        raise AssertionError("incomplete Q2 matrix was accepted")


def test_q2_manifest_rejects_rehashed_tampered_interval_without_dataset():
    manifest = build_q2_mask_manifest(_dataset())
    tampered = copy.deepcopy(manifest)
    entry = next(item for item in tampered["entries"] if item["combination"] == "T" and item["synthetic_missing_counts"]["text"] > 0)
    entry["effective_fractions"]["text"] += 0.1
    payload = dict(tampered)
    payload.pop("mask_sha256")
    import hashlib

    tampered["mask_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="effective"):
        validate_q2_mask_manifest(tampered)


def test_q2_manifest_rejects_tampered_data_source_hash():
    manifest = build_q2_mask_manifest(_dataset())
    tampered = copy.deepcopy(manifest)
    tampered["data_hashes"]["train"] = "0" * 64
    payload = dict(tampered)
    payload.pop("mask_sha256")
    import hashlib

    tampered["mask_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="data hash"):
        validate_q2_mask_manifest(tampered, dataset=_dataset())
