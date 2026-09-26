from __future__ import annotations

import numpy as np
import pytest

from e_emotion.problem2_fair.core import Problem2Split


def _split(*, view: str = "aligned_po", length: int = 50, label: float = 1.0) -> Problem2Split:
    values = {
        "text": np.ones((1, 50, 2), dtype=np.float32),
        "audio": np.arange(length, dtype=np.float32).reshape(1, length, 1),
        "vision": np.ones((1, length, 1), dtype=np.float32),
    }
    support = {name: np.ones(array.shape[:2], dtype=bool) for name, array in values.items()}
    observed = {name: array.copy() for name, array in support.items()}
    support["audio"][:, -5:] = False
    observed["audio"] &= support["audio"]
    observed["audio"][:, : length // 10] = False
    observed["text"][:, 10:15] = False
    return Problem2Split(
        ids=np.array(["item"]), values=values, physical_support=support,
        observed=observed, input_ids=np.tile(np.arange(100, 150, dtype=np.int64), (1, 1)),
        regression=np.array([label], dtype=np.float32),
        classification=np.array([int(np.sign(label)) + 1]), view=view, split="test",
    )


def test_ebmc_input_uses_observation_for_values_and_physical_support_for_attention():
    from e_emotion.problem2_fair.methods.ebmc import EBMCAdapter

    batch = EBMCAdapter(text_reencoder=lambda split: split.values["text"]).prepare_arrays(_split())

    assert batch["audio"].shape == (1, 50, 1)
    assert not batch["audio"][0, :5].any()
    assert not batch["audio"][0, -5:].any()
    assert batch["audio_mask"][0, 0] == 1
    assert batch["audio_mask"][0, -1] == 0
    assert batch["text_mask"][0, 10] == 1
    assert batch["label"][0, 0] == 1


def test_ebmc_unaligned_pools_after_native_observation_deletion():
    from e_emotion.problem2_fair.methods.ebmc import EBMCAdapter

    batch = EBMCAdapter(text_reencoder=lambda split: split.values["text"]).prepare_arrays(
        _split(view="unaligned_po", length=500)
    )

    assert batch["audio"].shape == (1, 50, 1)
    assert batch["audio"][0, 0, 0] == 0
    assert batch["audio"][0, 5, 0] == pytest.approx(54.5)


def test_ebmc_text_reencoder_gets_deleted_tokens_and_cannot_restore_missing_xt():
    from e_emotion.problem2_fair.methods.ebmc import EBMCAdapter

    received = []
    adapter = EBMCAdapter(text_reencoder=lambda split: (received.append(split.input_ids.copy()) or np.full_like(split.values["text"], 7)))
    refreshed = adapter.reencode_text(_split())

    assert not received[0][0, 10:15].any()
    assert not refreshed.values["text"][0, 10:15].any()
    assert np.all(refreshed.values["text"][0, :10] == 7)


def test_ebmc_teacher_student_state_is_complete_and_not_interchanged():
    from e_emotion.problem2_fair.methods.ebmc import split_stage2_state

    student, teacher = split_stage2_state({"head.weight": 3, "teacher_model.head.weight": 7})

    assert student == {"head.weight": 3}
    assert teacher == {"head.weight": 7}
    with pytest.raises(ValueError, match="teacher"):
        split_stage2_state({"head.weight": 3})


def test_ebmc_train_missing_matches_native_rate_and_draw_order():
    from e_emotion.problem2_fair.methods.ebmc import EBMCAdapter, native_missing_arrays

    clean = EBMCAdapter(text_reencoder=lambda split: split.values["text"]).prepare_arrays(_split())
    result = native_missing_arrays(clean, mode="train", seed=2026, rng=np.random.RandomState(17))
    reference = np.random.RandomState(17)
    rates = tuple(0.0 if reference.rand() < 0.5 else float(reference.uniform(0, 1)) for _ in range(3))
    for name, rate in zip(("audio", "text", "vision"), rates):
        positions = np.flatnonzero(clean[f"{name}_mask"][0])
        keep = reference.uniform(size=len(positions)) > rate
        if name == "text":
            keep[0] = keep[-1] = True
        expected = clean[name].copy()
        expected[0, positions[~keep]] = 0
        np.testing.assert_array_equal(result[name], expected)
        np.testing.assert_array_equal(result[f"{name}_mask"], clean[f"{name}_mask"])
    assert not result["text"][0, 10:15].any()
    assert not result["audio"][0, -5:].any()


def test_ebmc_valid_missing_is_fixed_point_five_per_sample_seed():
    from e_emotion.problem2_fair.methods.ebmc import EBMCAdapter, native_missing_arrays

    clean = EBMCAdapter(text_reencoder=lambda split: split.values["text"]).prepare_arrays(_split())
    first = native_missing_arrays(clean, mode="valid", seed=2026)
    second = native_missing_arrays(clean, mode="valid", seed=2026)
    for name in ("audio", "text", "vision"):
        positions = np.flatnonzero(clean[f"{name}_mask"][0])
        reference = np.random.RandomState(2026 * 1_000_003)
        for prior in ("audio", "text", "vision"):
            prior_positions = np.flatnonzero(clean[f"{prior}_mask"][0])
            keep = reference.uniform(size=len(prior_positions)) > 0.5
            if prior == "text":
                keep[0] = keep[-1] = True
            if prior == name:
                break
        expected = clean[name].copy()
        expected[0, positions[~keep]] = 0
        np.testing.assert_array_equal(first[name], expected)
        np.testing.assert_array_equal(first[name], second[name])
        np.testing.assert_array_equal(first[f"{name}_mask"], clean[f"{name}_mask"])
    with pytest.raises(ValueError, match="mode"):
        native_missing_arrays(clean, mode="test", seed=2026)


def test_ebmc_prediction_does_not_read_test_labels(tmp_path):
    torch = pytest.importorskip("torch")
    from e_emotion.problem2_fair.methods.ebmc import EBMCAdapter

    class Probe(torch.nn.Module):
        def forward(self, features, masks, umask, first_stage, label, batch_idx):
            assert not first_stage
            assert bool(torch.all(label == 0))
            output = features[..., :1].mean(dim=0)
            return None, output, None, None, None, {}, None

    adapter = EBMCAdapter(text_reencoder=lambda split: split.values["text"], device="cpu")
    first = adapter.raw_predict(Probe(), _split(label=-2), batch_size=1)
    second = adapter.raw_predict(Probe(), _split(label=2), batch_size=1)
    np.testing.assert_array_equal(first, second)
