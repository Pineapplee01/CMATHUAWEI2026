import numpy as np
import os
import pytest
import torch
from torch.utils.data import DataLoader

from e_emotion.baselines.aumdf.fair_adapter import (
    AUMDFFairAdapter,
    AUMDFFairDataset,
    FrozenBertTextReencoder,
    adapt_problem2_split,
    create_adapter,
)
from e_emotion.problem2_fair import Problem2Dataset, Problem2Split, materialize_q2_condition


def _split(view):
    length = 50 if view == "aligned_po" else 500
    values = {
        "text": np.ones((1, 50, 2), dtype=np.float32),
        "audio": np.ones((1, length, 1), dtype=np.float32),
        "vision": np.ones((1, length, 1), dtype=np.float32),
    }
    values["audio"][:, :, 0] = np.arange(length)
    support = {name: np.ones(value.shape[:2], dtype=bool) for name, value in values.items()}
    observed = {name: mask.copy() for name, mask in support.items()}
    observed["text"][0, 2] = False
    observed["audio"][0, 0] = False
    return Problem2Split(
        ids=np.array(["sample-1"]),
        values=values,
        physical_support=support,
        observed=observed,
        input_ids=np.zeros((1, 50), dtype=np.int64),
        regression=np.array([1.0], dtype=np.float32),
        classification=np.array([2], dtype=np.int64),
        view=view,
        split="test",
    )


def test_aligned_adapter_preserves_physical_support_and_zeros_unobserved_values():
    split = _split("aligned_po")
    batch = adapt_problem2_split(split)

    assert batch["id"] == ("sample-1",)
    torch.testing.assert_close(batch["target"], torch.tensor([1.0]))
    assert all(batch["features"][name].shape[1] == 50 for name in split.values)
    assert batch["valid"]["text"][0, 2]
    assert not batch["observed"]["text"][0, 2]
    assert batch["features"]["text"][0, 2].count_nonzero() == 0
    assert split.values["text"][0, 2, 0] == 1.0


def test_unaligned_adapter_pools_after_q2_deletion_with_observed_only_mean():
    split = _split("unaligned_po")
    damaged = materialize_q2_condition(split, "A", "beginning", 0.1)
    batch = adapt_problem2_split(damaged)

    assert batch["features"]["audio"].shape == (1, 50, 1)
    assert batch["features"]["vision"].shape == (1, 50, 1)
    assert batch["valid"]["audio"][0, 0]
    assert not batch["observed"]["audio"][0, 0]
    assert batch["features"]["audio"][0, 0, 0] == 0
    assert batch["features"]["audio"][0, 5, 0] == 54.5
    assert batch["observed"]["vision"][0, 0]
    assert split.observed["audio"][0, 1]


def test_adapter_rejects_unexpected_time_axis():
    split = _split("unaligned_po")
    altered = Problem2Split(
        **{**split.__dict__, "view": "unaligned_windowed", "values": split.values}
    )
    try:
        adapt_problem2_split(altered)
    except ValueError as exc:
        assert "50" in str(exc)
    else:
        raise AssertionError("expected a 50-slot input error")


def test_fair_dataset_collates_model_inputs_without_losing_masks():
    split = _split("unaligned_po")
    batch = next(iter(DataLoader(AUMDFFairDataset(split), batch_size=1)))
    assert batch["id"] == ["sample-1"]
    assert batch["features"]["audio"].shape == (1, 50, 1)
    assert batch["valid"]["text"][0, 2]
    assert not batch["observed"]["text"][0, 2]


def test_text_reencoder_sees_only_observed_token_ids():
    split = _split("aligned_po")
    split = Problem2Split(**{**split.__dict__, "input_ids": np.arange(1, 51, dtype=np.int64)[None]})
    damaged = materialize_q2_condition(split, "T", "beginning", 0.1)
    received = []

    def encode(masked):
        received.append(masked.input_ids.copy())
        result = np.zeros_like(masked.values["text"])
        result[:, :, 0] = masked.input_ids.sum(axis=1)[:, None]
        return result

    adapter = AUMDFFairAdapter(config={}, device="cpu", text_reencoder=encode)
    refreshed = adapter.reencode_text(damaged)
    assert received[0][0, :5].tolist() == [0] * 5
    assert received[0][0, 5] == 6
    assert refreshed.values["text"][0, 0, 0] == 0
    assert refreshed.values["text"][0, 5, 0] == received[0].sum()
    assert split.input_ids[0, 0] == 1


def test_fair_bridge_trains_from_injected_dataset_and_predicts(tmp_path):
    splits = {
        name: Problem2Split(**{**_split("aligned_po").__dict__, "split": name})
        for name in ("train", "valid", "test")
    }
    dataset = Problem2Dataset(splits=splits, view="aligned_po", root=tmp_path)
    config = {
        "data_file": "must-not-be-read.pkl",
        "model": {"input_dims": [2, 1, 1], "hidden_dim": 6, "heads": 3, "dropout": 0},
        "training": {"teacher_epochs": 1, "student_epochs": 1, "batch_size": 1,
                     "learning_rate": .001, "seed": 2026, "patience": 1},
        "missingness": {"mode": "random", "rates": [.1], "validation_rate": .1},
    }
    adapter = AUMDFFairAdapter(config=config, device="cpu", text_reencoder=lambda s: s.values["text"])
    run_dir = tmp_path / "runs" / "trial"
    run_dir.mkdir(parents=True)
    checkpoint = adapter.train(dataset, seed=2026, run_dir=run_dir)
    assert checkpoint.is_file()
    assert (run_dir / "aumdf" / "teacher.pt").is_file()
    prediction = adapter.predict(splits["test"], checkpoint)
    assert prediction.raw_intensity.shape == (1,)
    assert np.isfinite(prediction.raw_intensity).all()
    assert prediction.decision_source.startswith("valid neutral interval")


def test_fair_bridge_requires_text_reencoder_before_training(tmp_path):
    adapter = AUMDFFairAdapter(config={}, device="cpu")
    with pytest.raises(ValueError, match="text_reencoder"):
        adapter.reencode_text(_split("aligned_po"))


def test_cli_factory_returns_aumdf_adapter():
    assert isinstance(create_adapter(), AUMDFFairAdapter)


def test_frozen_bert_reencodes_surviving_context_and_uses_view_scaler(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from transformers import AutoModel

    ids = np.zeros((1, 50), dtype=np.int64)
    ids[0, 1:6] = [11, 12, 13, 14, 15]
    split = _split("aligned_po")
    support = {**split.physical_support, "text": np.zeros((1, 50), dtype=bool)}
    support["text"][0, 1:6] = True
    observed = {**split.observed, "text": support["text"].copy()}
    split = Problem2Split(**{**split.__dict__, "input_ids": ids,
                            "physical_support": support, "observed": observed})
    damaged = materialize_q2_condition(split, "T", "beginning", 0.1)
    received = []

    class ContextModel(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            received.append(input_ids.cpu().clone())
            context = (input_ids * attention_mask).sum(-1, keepdim=True).float()
            return SimpleNamespace(last_hidden_state=context[:, None, :].expand(-1, 50, 2))

    monkeypatch.setattr(AutoModel, "from_pretrained", lambda *args, **kwargs: ContextModel())
    scaler = tmp_path / "scaler_params.npz"
    np.savez(scaler, mu_T=np.array([10., 10.]), sigma_T=np.array([2., 2.]))
    encoder = FrozenBertTextReencoder(tmp_path, scaler, device="cpu", batch_size=1)
    adapter = AUMDFFairAdapter(config={}, device="cpu", text_reencoder=encoder)
    refreshed = adapter.reencode_text(damaged)

    assert received[0][0, 0] == 101
    assert received[0][0, 1:5].tolist() == [0, 0, 0, 0]
    assert received[0][0, 6] == 102
    assert refreshed.values["text"][0, 0].tolist() == [0., 0.]
    assert refreshed.values["text"][0, 5, 0] == (101 + 15 + 102 - 10) / 2


@pytest.mark.skipif("AUMDF_FAIR_DATA_ROOT" not in os.environ, reason="real processed_po root not provided")
def test_frozen_bert_matches_clean_processed_po_text():
    root = os.environ["AUMDF_FAIR_DATA_ROOT"]
    view = os.environ["AUMDF_FAIR_VIEW"]
    with np.load(os.path.join(root, "test.npz"), allow_pickle=True) as source:
        original = np.array(source["XT"][:1], copy=True)
        input_ids = np.array(source["I"][:1], copy=True)
        if view == "aligned_po":
            text_support = np.array(source["P"][:1, 0], copy=True)
            text_observed = np.array(source["O"][:1, 0], copy=True)
            length = 50
        else:
            text_support = np.array(source["P_T"][:1], copy=True)
            text_observed = np.array(source["O_T"][:1], copy=True)
            length = 500
        target = np.asarray(source["regression_labels"][:1], dtype=np.float32)
    input_ids[~text_observed] = 0
    values = {"text": original, "audio": np.zeros((1, length, 1), np.float32),
              "vision": np.zeros((1, length, 1), np.float32)}
    support = {"text": text_support, "audio": np.zeros((1, length), bool),
               "vision": np.zeros((1, length), bool)}
    observed = {"text": text_observed, "audio": support["audio"], "vision": support["vision"]}
    split = Problem2Split(
        ids=np.array(["real-sample"]), values=values, physical_support=support, observed=observed,
        input_ids=input_ids, regression=target,
        classification=np.sign(target).astype(np.int64) + 1, view=view, split="test",
    )
    encoder = FrozenBertTextReencoder(
        os.environ.get("AUMDF_FAIR_BERT_ROOT", "/user_home/gaojianan/CPMCM/AAAmodel/bert-base-uncased"),
        os.path.join(root, "scaler_params.npz"), device="cpu", batch_size=1,
    )
    refreshed = AUMDFFairAdapter(text_reencoder=encoder).reencode_text(split)
    np.testing.assert_allclose(refreshed.values["text"][text_observed], original[text_observed], atol=2e-4)
