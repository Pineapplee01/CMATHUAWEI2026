import pickle

import numpy as np
import pytest
import torch

from aumdf.data import load_attachment2, resolve_data_path
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
    assert select_neutral_threshold(np.array([-1., 0., 1.]), np.array([-.8, .1, .9])) >= .1


def test_path_escape_is_rejected(tmp_path):
    (tmp_path / "data").mkdir()
    with pytest.raises(ValueError, match="data"):
        resolve_data_path(tmp_path, "../outside.pkl")


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
    datasets, scaler, audit = load_attachment2(path, project_root=tmp_path, expected_dims=(8,4,2))
    assert scaler["text"]["mean"] == [2.] * 8
    assert len(datasets["valid"]) == 2
    assert audit["split_sizes"] == {"train": 2, "valid": 2, "test": 2}
    assert datasets["train"][0]["valid"]["text"].sum() == 3
    assert torch.count_nonzero(datasets["train"][0]["features"]["text"][3:]) == 0
