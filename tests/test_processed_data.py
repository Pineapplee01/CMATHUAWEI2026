import numpy as np
import pytest
import json

from e_emotion.data import (
    build_processed_manifest,
    contiguous_missing_mask,
    load_processed_dataset,
    load_processed_split,
    native_coordinate_mask,
)


def _write_split(path, ids):
    n = len(ids)
    rng = np.random.default_rng(3)
    m_t = np.ones((n, 50), dtype=bool)
    m_a = np.ones((n, 50), dtype=bool)
    m_v = np.ones((n, 50), dtype=bool)
    m_t[:, -2:] = False
    m_a[:, 10] = False
    m_v[:, 20:22] = False
    y = np.linspace(-1, 1, n, dtype=np.float32)
    np.savez(
        path,
        XT=rng.normal(size=(n, 50, 768)).astype(np.float32),
        XA=rng.normal(size=(n, 50, 74)).astype(np.float32),
        XV=rng.normal(size=(n, 50, 35)).astype(np.float32),
        mT=m_t,
        mA=m_a,
        mV=m_v,
        q=np.stack([m_t.any(1), m_a.any(1), m_v.any(1)], axis=1),
        P=~m_t,
        ids=np.asarray(ids),
        y_regression=y,
        y_classification=np.where(y < 0, 0, np.where(y > 0, 2, 1)),
    )


def test_loader_preserves_native_missing_separately(tmp_path):
    path = tmp_path / "train.npz"
    _write_split(path, ["a", "b"])
    split = load_processed_split(path)
    assert not split.native_valid_mask["audio"][0, 10]
    assert not split.observed_mask["audio"][0, 10]
    missing, observed = contiguous_missing_mask(split, "audio", 0.5, "beginning")
    assert missing.shape == (2, 50)
    assert not observed[0, 10]
    assert observed[0].sum() == split.native_valid_mask["audio"][0].sum() - 24


def test_loader_rejects_classification_mismatch(tmp_path):
    path = tmp_path / "train.npz"
    _write_split(path, ["a"])
    with np.load(path, allow_pickle=False) as payload:
        values = {key: payload[key] for key in payload.files}
    values["y_classification"] = np.asarray([2], dtype=np.int64)
    np.savez(path, **values)
    with pytest.raises(ValueError, match="y_classification"):
        load_processed_split(path)


def test_dataset_checks_cross_split_ids(tmp_path):
    for split in ("train", "valid", "test"):
        _write_split(tmp_path / f"{split}.npz", ["same"])
    with pytest.raises(ValueError, match="unique across"):
        load_processed_dataset(tmp_path)


def test_q2_audio_coordinates_intersect_text_native_mask(tmp_path):
    path = tmp_path / "train.npz"
    _write_split(path, ["a"])
    with np.load(path, allow_pickle=False) as payload:
        values = {key: payload[key] for key in payload.files}
    # Audio is natively present at a text-invalid position; it must not count
    # toward the Q2 continuous coordinate domain.
    values["mT"][:, 48:] = False
    values["mA"][:, 48:] = True
    np.savez(path, **values)
    split = load_processed_split(path)
    assert not native_coordinate_mask(split, "audio")[0, 48]
    synthetic, observed = contiguous_missing_mask(split, "audio", 1.0, "beginning")
    assert synthetic[0].sum() == 48 - 1  # mA position 10 is native-missing
    assert observed[0, 48]
    assert not split.synthetic_missing_mask["audio"].any()


def test_manifest_contains_hashes_protocol_and_is_json_serializable(tmp_path):
    for split_name, ids in (("train", ["tr"]), ("valid", ["va"]), ("test", ["te"])):
        _write_split(tmp_path / f"{split_name}.npz", ids)
    dataset = load_processed_dataset(tmp_path)
    manifest = build_processed_manifest(dataset)
    assert manifest["protocol_version"] == "e-competition-v1"
    assert manifest["feature_version"] == "aligned_50"
    assert manifest["split_sizes"] == {"train": 1, "valid": 1, "test": 1}
    for record in manifest["splits"].values():
        assert len(record["sha256"]) == 64
        assert record["source_path"].endswith(".npz")
    assert manifest["mask_semantics"]["observed_mask"] == "native_valid_mask & ~synthetic_missing_mask"
    json.dumps(manifest)


def test_loader_rejects_nonfinite_feature_and_regression(tmp_path):
    path = tmp_path / "train.npz"
    _write_split(path, ["a"])
    with np.load(path, allow_pickle=False) as payload:
        values = {key: payload[key] for key in payload.files}
    values["XA"][0, 0, 0] = np.nan
    np.savez(path, **values)
    with pytest.raises(ValueError, match="non-finite"):
        load_processed_split(path)
    values["XA"][0, 0, 0] = 0
    values["y_regression"][0] = np.inf
    np.savez(path, **values)
    with pytest.raises(ValueError, match="y_regression"):
        load_processed_split(path)
