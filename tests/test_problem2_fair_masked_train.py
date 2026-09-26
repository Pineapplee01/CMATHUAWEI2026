from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e_emotion.baselines.direct_fusion import EarlyFusionGRU
from e_emotion.problem2_fair.core import Problem2Split
from e_emotion.problem2_fair.views import Problem2Dataset


def _split(view: str = "aligned_po", *, split: str = "train", size: int = 2) -> Problem2Split:
    length = 500 if view == "unaligned_po" else 50
    text = np.ones((size, 50, 768), dtype=np.float32)
    audio = np.broadcast_to(np.arange(length, dtype=np.float32)[None, :, None], (size, length, 74)).copy()
    vision = np.ones((size, length, 35), dtype=np.float32)
    support = {
        "text": np.ones((size, 50), dtype=bool),
        "audio": np.ones((size, length), dtype=bool),
        "vision": np.ones((size, length), dtype=bool),
    }
    observed = {key: mask.copy() for key, mask in support.items()}
    observed["audio"][:, -1] = False
    audio[:, -1] = 0
    tokens = np.broadcast_to(np.arange(100, 150, dtype=np.int64)[None], (size, 50)).copy()
    labels = np.asarray([-1.0, 1.0] * ((size + 1) // 2), dtype=np.float32)[:size]
    return Problem2Split(
        ids=np.asarray([f"{split}-{index}" for index in range(size)]),
        values={"text": text, "audio": audio, "vision": vision},
        physical_support=support,
        observed=observed,
        input_ids=tokens,
        regression=labels,
        classification=np.sign(labels).astype(np.int64) + 1,
        view=view,
        split=split,
    )


def _encoder(split: Problem2Split) -> np.ndarray:
    return np.full(split.values["text"].shape, 3.0, dtype=np.float32)


def test_aligned_augmentation_preserves_physical_support_and_reencodes_deleted_tokens():
    from e_emotion.problem2_fair.methods.masked_train import MaskedTrainAdapter

    source = _split()
    received = []

    def encoder(split):
        received.append(split.input_ids.copy())
        return _encoder(split)

    adapter = MaskedTrainAdapter(text_reencoder=encoder)
    augmented = adapter.augment_training_split(source, rng=np.random.default_rng(7), condition=("TAV", "beginning", 0.1))

    for name in ("text", "audio", "vision"):
        np.testing.assert_array_equal(augmented.physical_support[name], source.physical_support[name])
        assert np.all(augmented.observed[name] <= source.observed[name])
        assert not augmented.observed[name][:, :5].any()
        assert not augmented.values[name][:, :5].any()
    assert not received[0][:, :5].any()
    assert not augmented.values["text"][:, :5].any()
    assert np.all(augmented.values["text"][:, 5:] == 3.0)
    assert source.observed["text"].all()
    assert source.input_ids[0, 0] == 100


def test_unaligned_augmentation_deletes_native_window_before_pooling():
    from e_emotion.problem2_fair.methods.masked_train import MaskedTrainAdapter

    source = _split("unaligned_po")
    adapter = MaskedTrainAdapter(text_reencoder=_encoder)
    augmented = adapter.augment_training_split(source, rng=np.random.default_rng(7), condition=("A", "beginning", 0.1))

    assert augmented.view == "unaligned_windowed"
    assert augmented.values["audio"].shape == (source.size, 50, 74)
    assert not augmented.observed["audio"][:, :5].any()
    assert augmented.physical_support["audio"][:, :5].all()
    assert np.all(augmented.values["audio"][:, 5, :] == pytest.approx(54.5))
    assert source.observed["audio"][:, :50].all()
    assert np.all(augmented.values["text"] == 3.0)


def test_training_condition_sampling_is_seed_reproducible():
    from e_emotion.problem2_fair.methods.masked_train import MaskedTrainAdapter

    source = _split()
    adapter = MaskedTrainAdapter(text_reencoder=_encoder)
    first = np.random.default_rng(2026)
    second = np.random.default_rng(2026)
    a = [adapter.augment_training_split(source, rng=first) for _ in range(5)]
    b = [adapter.augment_training_split(source, rng=second) for _ in range(5)]

    for left, right in zip(a, b):
        for name in ("text", "audio", "vision"):
            np.testing.assert_array_equal(left.observed[name], right.observed[name])
            np.testing.assert_array_equal(left.values[name], right.values[name])


@pytest.mark.parametrize("view", ["aligned_po", "unaligned_po"])
def test_train_selects_valid_checkpoint_and_predicts_with_unchanged_gru(tmp_path, view):
    from e_emotion.problem2_fair.methods.masked_train import MaskedTrainAdapter

    direct_fusion = Path(__import__("e_emotion.baselines.direct_fusion", fromlist=["x"]).__file__)
    original_hash = hashlib.sha256(direct_fusion.read_bytes()).hexdigest()
    dataset = Problem2Dataset(
        splits={name: _split(view, split=name, size=2) for name in ("train", "valid", "test")},
        view=view,
        root=tmp_path,
    )
    adapter = MaskedTrainAdapter(text_reencoder=_encoder, device="cpu", epochs=2, batch_size=2, hidden_dim=8)
    checkpoint = adapter.train(dataset, seed=2026, run_dir=tmp_path)
    metadata = json.loads((tmp_path / "masked_train.json").read_text(encoding="utf-8"))
    prepared = adapter._prepare_clean(dataset["test"])
    prediction = adapter.predict(prepared, checkpoint)
    model = adapter._loaded_model
    adapter.predict(prepared, checkpoint)

    assert checkpoint.is_file()
    assert metadata["selection"]["split"] == "valid"
    assert metadata["augmentation"]["condition_count"] == 64
    assert metadata["augmentation"]["rng_seed"] == 2026
    assert metadata["loss_weights"] == {"cross_entropy": 1.0, "mse": 1.0}
    assert prediction.raw_intensity.shape == (dataset["test"].size,)
    assert prediction.decision_source == "native_logits_argmax"
    assert adapter._loaded_model is model
    assert hashlib.sha256(direct_fusion.read_bytes()).hexdigest() == original_hash
    assert isinstance(adapter._build_model(), EarlyFusionGRU)
