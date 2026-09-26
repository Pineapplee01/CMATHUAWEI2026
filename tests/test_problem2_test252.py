from __future__ import annotations

import numpy as np

from e_emotion.problem2_fair.test252 import materialize_scenario, scenarios
from tests.test_problem2_fair_masked_train import _split


def test_test252_grid_matches_retrain_v2_cardinality_and_random_repeats():
    grid = scenarios()

    assert len(grid) == 252
    assert sum(position == "random" for _, _, position, _ in grid) == 126
    assert {seed for _, _, position, seed in grid if position == "random"} == {2026, 2027, 2028}


def test_test252_removes_contiguous_aligned_slots_without_emptying_observation():
    source = _split("aligned_po", split="test", size=2)
    partial, records = materialize_scenario(source, "TA", 0.5, "middle", 2026)

    assert len(records) == source.size * 2
    for record in records:
        assert record["observed_after"] >= 1
        assert record["removed_observed_slots"] >= 0
    for modality in ("text", "audio"):
        assert np.all(partial.observed[modality] <= source.observed[modality])
        assert np.all(partial.values[modality][~partial.observed[modality]] == 0)
    np.testing.assert_array_equal(partial.observed["vision"], source.observed["vision"])
