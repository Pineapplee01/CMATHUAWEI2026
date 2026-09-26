from __future__ import annotations

import numpy as np
import yaml

from e_emotion.problem2_fair.core import Problem2Split
from e_emotion.problem2_fair.methods.qa_moe import (
    QAMoEFairAdapter,
    _augment_sample,
    _mae_improved,
    _validation_modes,
)


def _split(*, view: str = "aligned_po", length: int = 50) -> Problem2Split:
    size = 2
    values = {
        "text": np.full((size, 50, 4), 17, dtype=np.float32),
        "audio": np.arange(length, dtype=np.float32)[None, :, None].repeat(size, axis=0),
        "vision": np.full((size, length, 2), 3, dtype=np.float32),
    }
    support = {name: np.ones(value.shape[:2], dtype=bool) for name, value in values.items()}
    observed = {name: mask.copy() for name, mask in support.items()}
    observed["text"][0, 10:15] = False
    observed["text"][1] = False
    observed["audio"][0, : length // 10] = False
    observed["vision"][1] = False
    tokens = np.tile(np.arange(100, 150, dtype=np.int64), (size, 1))
    return Problem2Split(
        ids=np.array(["a", "b"]), values=values,
        physical_support=support, observed=observed, input_ids=tokens,
        regression=np.array([-1.0, 1.0], dtype=np.float32),
        classification=np.array([0, 2], dtype=np.int64), view=view, split="test",
    )


def test_qa_moe_reencode_keeps_only_observed_token_semantics():
    split = _split()
    adapted = QAMoEFairAdapter().reencode_text(split)
    features = QAMoEFairAdapter().prepare_arrays(adapted)

    assert np.array_equal(adapted.input_ids[0, 10:15], np.zeros(5, dtype=np.int64))
    assert np.array_equal(features["text"][0, 0], adapted.input_ids[0])
    assert np.array_equal(features["text"][0, 1], (adapted.input_ids[0] != 0).astype(np.int64))
    assert not features["text"][:, 2].any()
    assert not features["text"][1].any()
    assert features["missing_text"].tolist() == [False, True]
    assert np.all(adapted.values["text"] == 0)
    assert split.input_ids[0, 10] != 0


def test_qa_moe_unaligned_pools_after_observation_deletion():
    split = _split(view="unaligned_po", length=500)
    arrays = QAMoEFairAdapter().prepare_arrays(split)

    assert arrays["audio"].shape == (2, 50, 1)
    assert not arrays["audio"][0, :5].any()
    assert arrays["audio"][0, 5, 0] == 54.5
    assert arrays["missing_audio"].tolist() == [False, False]
    assert arrays["missing_vision"].tolist() == [False, True]
    assert not arrays["vision"][1].any()


def test_qa_moe_masked_inputs_do_not_depend_on_stale_xt_or_unobserved_av():
    split = _split()
    first = QAMoEFairAdapter().prepare_arrays(split)
    values = {name: value.copy() for name, value in split.values.items()}
    values["text"][:] = 9999
    values["audio"][~split.observed["audio"]] = 9999
    values["vision"][~split.observed["vision"]] = 9999
    changed = Problem2Split(**{**split.__dict__, "values": values})
    second = QAMoEFairAdapter().prepare_arrays(changed)

    for key in ("text", "audio", "vision", "missing_text", "missing_audio", "missing_vision"):
        np.testing.assert_array_equal(first[key], second[key])


def test_qa_moe_restores_only_structural_tokens_outside_physical_text_support():
    split = _split()
    support = {name: mask.copy() for name, mask in split.physical_support.items()}
    observed = {name: mask.copy() for name, mask in split.observed.items()}
    support["text"][:, 0] = False
    support["text"][:, 49] = False
    observed["text"][:, 0] = False
    observed["text"][:, 49] = False
    tokens = split.input_ids.copy()
    tokens[:, [0, 49]] = 0
    changed = Problem2Split(**{
        **split.__dict__, "physical_support": support,
        "observed": observed, "input_ids": tokens,
    })

    arrays = QAMoEFairAdapter().prepare_arrays(changed)

    assert arrays["text"][0, 0, 0] == 101
    assert arrays["text"][0, 0, 49] == 102
    assert arrays["text"][0, 1, 0] == 1
    assert arrays["text"][0, 1, 49] == 1
    assert not arrays["text"][0, 0, 10:15].any()
    assert not arrays["text"][1].any()


def test_qa_moe_config_points_to_processed_po_instead_of_legacy_pickle(tmp_path):
    source = tmp_path / "QA-MoE"
    source.mkdir()
    (source / "cpmcm_mosei.yaml").write_text(
        yaml.safe_dump({"dataset": "MOSI", "data_path": "/old/aligned_50.pkl", "batch_size": 16}),
        encoding="utf-8",
    )
    data_root = tmp_path / "processed_po"
    adapter = QAMoEFairAdapter(source_root=source, device="cpu")

    config = adapter._config(_split(), data_root=data_root)

    assert config.data_path == str(data_root)
    assert config.dataset == "MOSEI"
    assert config.audio_input_dim == 1
    assert config.vision_input_dim == 2


class _FixedNoise:
    def normal(self, loc, scale, size):
        return np.full(size, scale, dtype=np.float32)

    def rand(self, *shape):
        keep = np.ones(shape, dtype=np.float32)
        keep[..., 2] = 0.0
        return keep


def test_qa_moe_native_noise_precedes_whole_modality_drop_and_respects_o():
    split = _split()
    arrays = QAMoEFairAdapter().prepare_arrays(split)
    sample = {name: values[0].copy() for name, values in arrays.items()}
    result = _augment_sample(
        sample, {name: split.observed[name][0] for name in ("text", "audio", "vision")},
        noise_level=0.1, missing_mode=1, rng=_FixedNoise(),
    )

    assert not result["text"].any()
    assert bool(result["missing_text"])
    assert not bool(result["missing_audio"])
    assert not result["audio"][:5].any()
    np.testing.assert_allclose(result["audio"][5:, 0], arrays["audio"][0, 5:, 0] + 0.1)
    assert sample["text"][0, 2] != 0


def test_qa_moe_native_word_dropout_changes_ids_only():
    split = _split()
    arrays = QAMoEFairAdapter().prepare_arrays(split)
    sample = {name: values[0].copy() for name, values in arrays.items()}
    result = _augment_sample(
        sample, {name: split.observed[name][0] for name in ("text", "audio", "vision")},
        noise_level=0.1, missing_mode=6, rng=_FixedNoise(),
    )

    assert result["text"][0, 2] == 0
    assert result["text"][1, 2] == 1
    assert result["text"][0, 3] == sample["text"][0, 3]
    assert not bool(result["missing_text"])


def test_qa_moe_valid_missing_modes_are_fixed_from_native_seed():
    first = _validation_modes(100, drop_rate=0.1, seed=1111)
    second = _validation_modes(100, drop_rate=0.1, seed=1111)

    np.testing.assert_array_equal(first, second)
    assert set(first.tolist()) <= set(range(7))
    assert (first != 6).any()


def test_qa_moe_early_stop_requires_native_delta():
    assert _mae_improved(0.8, float("inf"))
    assert not _mae_improved(0.99995, 1.0)
    assert not _mae_improved(0.9999, 1.0)
    assert _mae_improved(0.9998, 1.0)
