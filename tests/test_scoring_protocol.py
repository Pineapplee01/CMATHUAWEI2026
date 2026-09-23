"""Executable team protocol; official requirements are traced in docs/evaluation/."""
import copy
import json

import numpy as np
import pytest

import e_emotion.evaluation as evaluation
from e_emotion.cli.main import main
from e_emotion.contracts import Polarity


def test_output_adapter_is_fixed_not_batch_minmax():
    raw = np.array([-4., 0., 4.])
    output = evaluation.adapt_output(raw)
    np.testing.assert_array_equal(output.raw_intensity, raw)
    np.testing.assert_array_equal(output.intensity, [-3, 0, 3])
    np.testing.assert_array_equal(output.polarity, [0, 1, 2])
    np.testing.assert_array_equal(raw, [-4, 0, 4])
    assert evaluation.adapt_output([.2], scale=3).intensity[0] == pytest.approx(.6)
    assert evaluation.adapt_output([.2, .8], scale=3).intensity[0] == pytest.approx(.6)
    assert evaluation.adapt_output([0], scale=2, offset=1).intensity[0] == 1


def test_scoring_is_pure_and_rejects_invalid_final_output():
    y = np.array([-3., 0., 3.])
    p = np.array([-4., 0., 4.])
    assert evaluation.mae(y, p) == pytest.approx(2/3)  # raw diagnostic, not a valid submission
    with pytest.raises(ValueError, match="range"):
        evaluation.score_predictions(y, p, [0, 1, 2])
    np.testing.assert_array_equal(p, [-4, 0, 4])
    adapted = evaluation.adapt_output(p)
    result = evaluation.score_predictions(y, adapted.intensity, adapted.polarity)
    assert result["mae"] == 0
    assert result["accuracy"] == 1
    assert result["protocol_version"] == "e-competition-v1"


def test_truth_zero_is_neutral_predictions_use_submitted_class():
    # Separate classification heads are supported; scoring must not regenerate the class.
    result = evaluation.score_predictions([-1, 0, 1], [-1, .1, 1], [0, 1, 2])
    assert result["accuracy"] == 1
    with pytest.raises(ValueError, match="truth polarity"):
        evaluation.score_predictions([-1, 0, 1], [-1, .1, 1], [0, 1, 2], truth_polarity=[0, 2, 2])


def test_domain_polarity_enums_are_accepted_without_string_coercion():
    result = evaluation.score_predictions([-1, 0, 1], [-1, 0, 1], list(Polarity))
    assert result["accuracy"] == 1


def test_absent_class_rule_and_undefined_pearson():
    result = evaluation.score_predictions([0, 0], [0, 0], [1, 1])
    assert result["macro_f1"] == pytest.approx(1/3)
    assert result["weighted_f1"] == 1
    assert result["class_support"] == [0, 2, 0]
    assert result["pearson"] is None
    assert result["pearson_undefined_reason"] == "constant_truth"
    assert evaluation.pearson([1], [2]) is None
    assert evaluation.pearson([0, 1], [2, 2]) is None
    assert evaluation.pearson([.1, .1, .1], [1, 2, 3]) is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("y,p,c", [([], [], []), ([0], [0, 1], [1]),
    ([0], [float('nan')], [1]), ([float('inf')], [0], [1]),
    ([[0, 1]], [[0, 1]], [1, 2]), ([4], [3], [2]), ([0], [0], [3]), ([0], [0], [1.5])])
def test_invalid_scoring_inputs_fail(y, p, c):
    with pytest.raises(ValueError):
        evaluation.score_predictions(y, p, c)


def test_threshold_selection_is_validation_only():
    y, p = [-1, 0, 1], [-.8, .1, .9]
    assert evaluation.select_neutral_threshold(y, p, split="valid") == pytest.approx(.1)
    for split in ("train", "test", "attachment3", "attachment4"):
        with pytest.raises(ValueError, match="valid"):
            evaluation.select_neutral_threshold(y, p, split=split)
    np.testing.assert_array_equal(evaluation.adapt_output([-.1, 0, .1], threshold=.1).polarity, [1, 1, 1])
    for threshold in (-1, float("nan"), 4):
        with pytest.raises(ValueError):
            evaluation.adapt_output([0], threshold=threshold)


def example_rows():
    return [{"id": "v$_$1", "intensity": -1}, {"id": "v$_$2", "intensity": 1}]


def test_csv_roundtrip_and_id_join_use_identical_final_values(tmp_path):
    truth = example_rows()
    output = evaluation.adapt_output([4, -4])
    rows = evaluation.prediction_rows(["v$_$2", "v$_$1"], output)
    before = copy.deepcopy(rows)
    report = evaluation.score_records(truth, rows)
    assert report["mae"] == 2
    assert report["accuracy"] == 1
    path = tmp_path / "predictions.csv"
    evaluation.write_predictions_csv(path, rows)
    assert evaluation.score_records(truth, evaluation.read_records_csv(path)) == report
    assert rows == before
    with pytest.raises(FileExistsError):
        evaluation.write_predictions_csv(path, rows)


def test_csv_roundtrip_preserves_numpy_float32_values(tmp_path):
    truth = example_rows()
    rows = [{"id": "v$_$1", "intensity": np.float32(-.87654321), "polarity": 0},
            {"id": "v$_$2", "intensity": np.float32(.98765432), "polarity": 2}]
    before = evaluation.score_records(truth, rows)
    path = tmp_path / "float32.csv"
    evaluation.write_predictions_csv(path, rows)
    assert evaluation.score_records(truth, evaluation.read_records_csv(path)) == before


@pytest.mark.parametrize("p", [[0, 1e-160], [1, np.nextafter(1., 2.)]])
def test_pearson_is_stable_for_small_nonzero_variance(p):
    assert evaluation.pearson([0, 1], p) == pytest.approx(1, abs=1e-14)


@pytest.mark.parametrize("base", [.3, 1.5, 2.5])
def test_pearson_preserves_nearly_constant_equal_spacing(base):
    predicted = base + np.spacing(base) * np.arange(3)
    assert evaluation.pearson([0, 1, 2], predicted) == pytest.approx(1, abs=1e-14)


@pytest.mark.parametrize("ids", [["v$_$1", "v$_$1"], ["v$_$1", "v$_$3"], ["v$_$1"], ["", "v$_$2"]])
def test_duplicate_missing_extra_or_empty_ids_rejected(ids):
    rows = [{"id": i, "intensity": 0, "polarity": "Neutral"} for i in ids]
    with pytest.raises(ValueError):
        evaluation.score_records(example_rows(), rows)


def test_cli_scores_csv_and_restricts_labels_to_data(tmp_path, capsys):
    data = tmp_path / "data"
    data.mkdir()
    labels = data / "valid.csv"
    labels.write_text("id,intensity\nv$_$1,-1\nv$_$2,1\n", encoding="utf-8")
    config = tmp_path / "base.yaml"
    config.write_text("dataset:\n  data_root: data\n", encoding="utf-8")
    output = tmp_path / "artifacts" / "predictions.csv"
    evaluation.write_predictions_csv(output, evaluation.prediction_rows(
        ["v$_$1", "v$_$2"], evaluation.adapt_output([-1, 1])))
    args = ["score", "--config", str(config), "--truth", str(labels), "--predictions", str(output)]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["mae"] == 0
    labels_outside = tmp_path / "outside.csv"
    labels_outside.write_text(labels.read_text(), encoding="utf-8")
    args[4] = str(labels_outside)
    assert main(args) == 2


@pytest.mark.parametrize("raw,kwargs", [([float("inf")], {}), ([0], {"scale": 0}),
    ([0], {"offset": float("nan")}), ([1e308], {"scale": 3}),
    ([0, 1], {"predicted_polarity": [1]})])
def test_output_adapter_rejects_invalid_inputs(raw, kwargs):
    with pytest.raises(ValueError):
        evaluation.adapt_output(raw, **kwargs)
