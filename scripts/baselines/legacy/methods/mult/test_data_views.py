import numpy as np

from data_views import resolve_view, run_root
from runtime import Bundle, _positions, raw_q2_manifest


def test_mult_views_have_distinct_roots_and_contracts():
    raw = resolve_view("raw_unaligned")
    aligned = resolve_view("aligned_50_control")

    assert raw.canonical is False
    assert aligned.canonical is True
    assert raw.data_root != aligned.data_root
    assert run_root("raw_unaligned") != run_root("aligned_50_control")


def test_raw_manifest_records_one_entry_per_sample_and_condition():
    bundle = Bundle(
        ids=["a", "b"],
        values={name: np.zeros((2, 4, 1), dtype=np.float32) for name in ("text", "audio", "vision")},
        native={
            "text": np.array([[True, True, False, False], [True, True, True, False]]),
            "audio": np.array([[True, True, True, False], [True, False, False, False]]),
            "vision": np.array([[True, False, False, False], [True, True, False, False]]),
        },
        labels=np.zeros(2, dtype=np.float32),
    )
    manifest = raw_q2_manifest(bundle)

    assert len(manifest["entries"]) == 2 * 63
    entry = next(item for item in manifest["entries"] if item["sample_id"] == "a" and item["combination"] == "A" and item["position"] == "beginning" and item["requested_fraction"] == 0.5)
    assert entry["coordinate_lengths"]["audio"] == 3
    assert entry["synthetic_missing_counts"]["audio"] == 2
    assert entry["synthetic_start_positions"]["audio"] == 0
    assert entry["synthetic_end_positions"]["audio"] == 2


def test_aligned_q2_uses_manifest_count_instead_of_floor():
    mask = np.array([True, True, True, False])
    selected = _positions(mask, 0.5, "beginning", count=2)
    assert selected.tolist() == [0, 1]
