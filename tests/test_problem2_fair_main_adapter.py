from __future__ import annotations

import numpy as np

from e_emotion.problem2_fair import Problem2Split
from e_emotion.problem2_fair.methods.main import HISTORICAL_MAIN_SEEDS, FrozenMainMethodAdapter, checkpoint_for


def test_existing_main_checkpoint_map_is_explicitly_historical():
    for view in ("aligned_po", "unaligned_po"):
        for seed in HISTORICAL_MAIN_SEEDS:
            path = checkpoint_for(view, seed)
            assert path.name == f"retrain_processed_po_seed{seed}.pt"
    assert "retrain_f1_clssep_avP0" in str(checkpoint_for("aligned_po", 2026))
    assert "retrain_v2_unaligned_modalgate" in str(checkpoint_for("unaligned_po", 2030))


def test_main_adapter_restores_structural_tokens_for_online_bert():
    values = {
        "text": np.ones((1, 50, 2), np.float32),
        "audio": np.ones((1, 50, 1), np.float32),
        "vision": np.ones((1, 50, 1), np.float32),
    }
    support = {name: np.ones((1, 50), bool) for name in values}
    observed = {name: value.copy() for name, value in support.items()}
    support["text"][:, 0] = False
    observed["text"][:, 0] = False
    support["text"][:, 6:] = False
    observed["text"][:, 6:] = False
    observed["text"][:, 3] = False
    split = Problem2Split(
        ids=np.array(["v$_$0"]), values=values,
        physical_support=support, observed=observed,
        input_ids=np.array([[0, 11, 12, 0, 14, 15] + [0] * 44], dtype=np.int64),
        regression=np.array([1.0], np.float32), classification=np.array([2], np.int64),
        view="aligned_po", split="test",
    )
    adapted = FrozenMainMethodAdapter().reencode_text(split)

    assert adapted.input_ids[0, 0] == 101
    assert adapted.input_ids[0, 6] == 102
    assert adapted.input_ids[0, 3] == 0
    assert np.array_equal(adapted.values["text"], split.values["text"])
