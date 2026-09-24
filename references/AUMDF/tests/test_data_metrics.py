import pickle

import numpy as np
import pytest
import torch

from aumdf.data import load_attachment2, resolve_data_path, resolve_source_path
from aumdf.missingness import corrupt
from aumdf.metrics import metrics, select_neutral_threshold


def test_corruption_is_reproducible_and_does_not_mutate_input():
    x = {m: torch.ones(2, 10, 3) for m in ("text", "audio", "vision")}
    valid = {m: torch.arange(10)[None, :].expand(2, -1) < 8 for m in x}
    a, mask_a = corrupt(x, valid, rate=0.5, mode="block", seed=1, modalities=("audio",))
    b, mask_b = corrupt(x, valid, rate=0.5, mode="block", seed=1, modalities=("audio",))
    torch.testing.assert_close(a["audio"], b["audio"])
    assert (x["audio"] == 1).all()
    for row in mask_a["audio"]:
        dropped = torch.where(~row[:8])[0]
        assert len(dropped) == 4
        assert (dropped[1:] - dropped[:-1] == 1).all()
    assert mask_a["text"].sum() == valid["text"].sum()


def test_polarity_keeps_zero_as_neutral_and_pearson_undefined():
    result = metrics(np.array([-1., 0., 1.]), np.array([-0.8, 0.01, 0.9]), threshold=0.05)
    assert result["accuracy_3"] == 1
    assert result["macro_f1_3"] == 1
    assert metrics(np.zeros(3), np.zeros(3))["pearson"] is None
    assert select_neutral_threshold(np.array([-1., 0., 1.]), np.array([-.8, .1, .9]), split="valid") >= .1


def test_path_escape_is_rejected(tmp_path):
    (tmp_path / "data").mkdir()
    with pytest.raises(ValueError, match="data"):
        resolve_data_path(tmp_path, "../outside.pkl")


def _write_processed_split(path, sample_id):
    n = 1
    native = np.ones((n, 50), dtype=bool)
    q = native.copy()
    y = np.array([0.5], dtype=np.float32)
    np.savez(
        path,
        XT=np.zeros((n, 50, 768), dtype=np.float32),
        XA=np.zeros((n, 50, 74), dtype=np.float32),
        XV=np.zeros((n, 50, 35), dtype=np.float32),
        mT=native,
        mA=native,
        mV=native,
        Q=q,
        P=~q,
        q=np.ones((n, 3), dtype=bool),
        rho_content=np.zeros((n, 3), dtype=np.float32),
        ids=np.asarray([sample_id]),
        y_regression=y,
        y_classification=np.asarray([2], dtype=np.int64),
    )


def test_strict_source_rejects_noncanonical_processed_root(tmp_path):
    canonical = tmp_path / "canonical"
    arbitrary = tmp_path / "other-processed"
    arbitrary.mkdir()
    with pytest.raises(ValueError, match="canonical processed root"):
        resolve_source_path(tmp_path, arbitrary, strict=True, canonical_root=canonical)


def test_strict_source_rejects_arbitrary_processed_root_with_splits(tmp_path):
    canonical = tmp_path / "canonical"
    arbitrary = tmp_path / "other-processed"
    arbitrary.mkdir()
    for split in ("train", "valid", "test"):
        _write_processed_split(arbitrary / f"{split}.npz", f"{split}$_$0")
    with pytest.raises(ValueError, match="canonical processed root"):
        resolve_source_path(tmp_path, arbitrary, strict=True, canonical_root=canonical)


def test_strict_source_rejects_legacy_pickle(tmp_path):
    path = tmp_path / "legacy.pkl"
    path.write_bytes(b"not used")
    with pytest.raises(ValueError, match="legacy .pkl"):
        resolve_source_path(tmp_path, path, strict=True, canonical_root=tmp_path)


def test_strict_source_rejects_single_split(tmp_path):
    path = tmp_path / "train.npz"
    path.write_bytes(b"not used")
    with pytest.raises(ValueError, match="single split"):
        resolve_source_path(tmp_path, path, strict=True, canonical_root=tmp_path)


def test_loader_defaults_to_strict_for_legacy_pickle(tmp_path):
    path = tmp_path / "data" / "legacy.pkl"
    path.parent.mkdir()
    path.write_bytes(b"not used")
    with pytest.raises(ValueError, match="legacy .pkl"):
        load_attachment2(path, project_root=tmp_path)


def test_strict_source_resolves_and_loads_canonical_npz_root(tmp_path):
    root = tmp_path / "processed"
    root.mkdir()
    for split in ("train", "valid", "test"):
        _write_processed_split(root / f"{split}.npz", f"{split}$_$0")
    np.savez(root / "scaler_params.npz", mean=np.zeros(1, dtype=np.float32))
    for name in ("preprocess_report.json", "model_input_contract.json", "bert_encode_report.json"):
        (root / name).write_text("{}", encoding="utf-8")
    resolved = resolve_source_path(tmp_path, root, strict=True, canonical_root=root)
    assert resolved == root.resolve()
    datasets, _, audit = load_attachment2(root, expected_dims=(768, 74, 35), strict=True, canonical_root=root)
    assert len(datasets["train"]) == 1
    assert audit["data_root"] == str(root.resolve())


def test_train_only_scaling_and_no_sample_drop(tmp_path):
    path = tmp_path / "data" / "aligned.pkl"
    path.parent.mkdir()
    payload = {}
    for split, offset in (("train", 0), ("valid", 100), ("test", 200)):
        payload[split] = {
            "id": [f"{split}$_$0", f"{split}$_$1"],
            "text": np.full((2, 5, 8), offset + 2., dtype=np.float32),
            "audio": np.full((2, 5, 4), offset + 3., dtype=np.float32),
            "vision": np.full((2, 5, 2), offset + 4., dtype=np.float32),
            "text_bert": np.tile(np.array([[1]*5, [1,1,1,0,0], [0]*5]), (2,1,1)),
            "regression_labels": np.array([-1., 1.]),
            "classification_labels": np.array([0., 2.]),
        }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    datasets, scaler, audit = load_attachment2(path, project_root=tmp_path, expected_dims=(8,4,2), strict=False)
    assert scaler["text"]["mean"] == [2.] * 8
    assert len(datasets["valid"]) == 2
    assert audit["split_sizes"] == {"train": 2, "valid": 2, "test": 2}
    assert datasets["train"][0]["valid"]["text"].sum() == 3
    assert torch.count_nonzero(datasets["train"][0]["features"]["text"][3:]) == 0


def test_aumdf_scoring_does_not_silently_clip():
    with pytest.raises(ValueError, match="range"):
        metrics(np.array([-3., 0., 3.]), np.array([-4., 0., 4.]))


def test_aumdf_metrics_delegate_to_shared_protocol():
    result = metrics(np.array([-1., 0., 1.]), np.array([-1., .1, 1.]), threshold=.1)
    assert result["protocol_version"] == "e-competition-v1"
    assert result["competition"]["accuracy"] == result["accuracy_3"] == 1
    assert result["competition"]["mae"] == result["mae"]
