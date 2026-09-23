import pickle

import numpy as np
import pytest

from aumdf.engine import train_run, evaluate_checkpoint


@pytest.mark.parametrize("smoke", [False, True])
def test_two_stage_training_checkpoint_and_evaluation(tmp_path, smoke):
    rng = np.random.default_rng(12)
    payload = {}
    for split, n in (("train", 8), ("valid", 4), ("test", 4)):
        labels = np.linspace(-2, 2, n).astype("float32")
        payload[split] = {
            "id": [split + "$_$" + str(i) for i in range(n)],
            "text": rng.normal(size=(n, 5, 8)).astype("float32"),
            "audio": rng.normal(size=(n, 5, 4)).astype("float32"),
            "vision": rng.normal(size=(n, 5, 2)).astype("float32"),
            "text_bert": np.tile(np.array([[1]*5, [1,1,1,0,0], [0]*5]), (n,1,1)),
            "regression_labels": labels,
            "classification_labels": np.sign(labels)+1,
        }
    data = tmp_path / "data" / "features.pkl"
    data.parent.mkdir()
    with data.open("wb") as f:
        pickle.dump(payload, f)
    config = {
        "data_file": "data/features.pkl",
        "model": {"input_dims": [8,4,2], "hidden_dim": 12, "heads": 3, "dropout": 0},
        "training": {"teacher_epochs": 1, "student_epochs": 1, "batch_size": 4,
                     "learning_rate": .001, "seed": 2026, "patience": 20},
        "missingness": {"mode": "random", "rates": [.1,.3], "validation_rate": .3},
    }
    output = tmp_path / "artifacts" / "run"
    result = train_run(config, output, device="cpu", project_root=tmp_path,
                       train_limit=6 if smoke else None, valid_limit=3 if smoke else None)
    assert result["teacher"]["epochs_completed"] == 1
    assert result["student"]["epochs_completed"] == 1
    assert (output / "teacher.pt").is_file()
    assert (output / "student.pt").is_file()
    assert result["scope"] == ("smoke_only" if smoke else "competition_subset_reproduction")
    evaluation = evaluate_checkpoint(output / "student.pt", split="test", device="cpu",
                                     project_root=tmp_path, rates=(0,.5), modes=("random","block"))
    assert evaluation["split"] == "test"
    assert evaluation["scope"] == ("smoke_evaluation" if smoke else "competition_subset_evaluation")
    assert evaluation["training_provenance"]["used_train"] == (6 if smoke else 8)
    assert evaluation["training_provenance"]["used_valid"] == (3 if smoke else 4)
    assert all(c["metrics"]["n"] == 4 for c in evaluation["conditions"])
    assert len(evaluation["conditions"]) == 3
    with pytest.raises(FileExistsError):
        train_run(config, output, device="cpu", project_root=tmp_path)
