from __future__ import annotations

import numpy as np
import pytest

from e_emotion.contracts import Polarity


def _aligned_split():
    from e_emotion.problem2_fair import Problem2Split

    ids = np.array(["v1$_$0", "v2$_$0"])
    values = {
        "text": np.ones((2, 50, 2), dtype=np.float32),
        "audio": np.ones((2, 50, 1), dtype=np.float32),
        "vision": np.ones((2, 50, 1), dtype=np.float32),
    }
    support = {name: np.ones((2, 50), dtype=bool) for name in values}
    observed = {name: np.ones((2, 50), dtype=bool) for name in values}
    return Problem2Split(
        ids=ids,
        values=values,
        physical_support=support,
        observed=observed,
        input_ids=np.tile(np.arange(50, dtype=np.int64), (2, 1)),
        regression=np.array([-1.0, 1.0], dtype=np.float32),
        classification=np.array([0, 2], dtype=np.int64),
        view="aligned_po",
        split="test",
    )


def _unaligned_split():
    from e_emotion.problem2_fair import Problem2Split

    ids = np.array(["v1$_$0"])
    values = {
        "text": np.ones((1, 50, 2), dtype=np.float32),
        "audio": np.ones((1, 500, 1), dtype=np.float32),
        "vision": np.ones((1, 500, 1), dtype=np.float32),
    }
    support = {
        "text": np.ones((1, 50), dtype=bool),
        "audio": np.ones((1, 500), dtype=bool),
        "vision": np.ones((1, 500), dtype=bool),
    }
    observed = {name: value.copy() for name, value in support.items()}
    return Problem2Split(
        ids=ids,
        values=values,
        physical_support=support,
        observed=observed,
        input_ids=np.tile(np.arange(50, dtype=np.int64), (1, 1)),
        regression=np.array([0.0], dtype=np.float32),
        classification=np.array([1], dtype=np.int64),
        view="unaligned_po",
        split="test",
    )


def _write_processed_po(
    root,
    *,
    unaligned: bool = False,
    invalid_observation: bool = False,
    missing_observation: bool = False,
    object_ids: bool = False,
):
    root.mkdir()
    for split, size in (("train", 2), ("valid", 1), ("test", 1)):
        ids = np.array([f"{split}{row}$_$0" for row in range(size)], dtype=object if object_ids else str)
        payload = {
            "XT": np.ones((size, 50, 2), dtype=np.float32),
            "I": np.tile(np.arange(50, dtype=np.int64), (size, 1)),
            "id": ids,
            "regression_labels": np.linspace(-1.0, 1.0, size, dtype=np.float32),
            "classification_labels": np.sign(np.linspace(-1.0, 1.0, size)).astype(np.int64) + 1,
        }
        if unaligned:
            payload |= {
                "XA": np.ones((size, 500, 1), dtype=np.float32),
                "XV": np.ones((size, 500, 1), dtype=np.float32),
                "P_T": np.ones((size, 50), dtype=bool),
                "P_A": np.ones((size, 500), dtype=bool),
                "P_V": np.ones((size, 500), dtype=bool),
                "O_T": np.ones((size, 50), dtype=bool),
                "O_A": np.ones((size, 500), dtype=bool),
                "O_V": np.ones((size, 500), dtype=bool),
            }
        else:
            payload |= {
                "XA": np.ones((size, 50, 1), dtype=np.float32),
                "XV": np.ones((size, 50, 1), dtype=np.float32),
                "P": np.ones((size, 3, 50), dtype=bool),
                "O": np.ones((size, 3, 50), dtype=bool),
            }
            if invalid_observation:
                payload["O"][0, 0, 0] = True
                payload["P"][0, 0, 0] = False
            if missing_observation:
                payload["O"][0, 1, 3] = False
                payload["O"][0, 0, 4] = False
        np.savez(root / f"{split}.npz", **payload)


def test_q2_v2_has_complete_plus_sixty_three_local_conditions():
    from e_emotion.problem2_fair import Q2_V2_CONDITIONS

    assert len(Q2_V2_CONDITIONS) == 64
    assert Q2_V2_CONDITIONS[0] == ("complete", None, 0.0)
    assert ("TAV", "middle", 0.5) in Q2_V2_CONDITIONS


def test_aligned_q2_window_hides_only_selected_observations():
    from e_emotion.problem2_fair import materialize_q2_condition

    condition = materialize_q2_condition(_aligned_split(), "A", "beginning", 0.1)

    assert not condition.observed["audio"][:, :5].any()
    assert condition.observed["text"].all()
    assert condition.observed["vision"].all()


def test_text_q2_window_also_removes_corresponding_token_ids():
    from e_emotion.problem2_fair import materialize_q2_condition

    source = _aligned_split()
    condition = materialize_q2_condition(source, "T", "beginning", 0.1)

    assert np.array_equal(condition.input_ids[:, :5], np.zeros((2, 5), dtype=np.int64))
    assert np.array_equal(condition.input_ids[:, 5:], source.input_ids[:, 5:])


def test_unaligned_q2_maps_one_text_window_to_tenfold_audio_and_vision_window():
    from e_emotion.problem2_fair import materialize_q2_condition

    condition = materialize_q2_condition(_unaligned_split(), "AV", "beginning", 0.1)

    assert not condition.observed["audio"][:, :50].any()
    assert not condition.observed["vision"][:, :50].any()
    assert condition.observed["audio"][:, 50:].all()
    assert condition.observed["vision"][:, 50:].all()


def test_q2_window_location_is_shared_across_modality_combinations():
    from e_emotion.problem2_fair import Problem2Split
    from e_emotion.problem2_fair.q2 import condition_windows

    split = _aligned_split()
    support = {name: np.zeros_like(mask) for name, mask in split.physical_support.items()}
    support["text"][:, :10] = True
    support["audio"][:, 30:40] = True
    support["vision"][:, 40:50] = True
    observed = {name: value.copy() for name, value in support.items()}
    split = Problem2Split(
        ids=split.ids,
        values=split.values,
        physical_support=support,
        observed=observed,
        input_ids=split.input_ids,
        regression=split.regression,
        classification=split.classification,
        view=split.view,
        split=split.split,
    )

    text_window = condition_windows(split, "T", "beginning", 0.1)[0]["text"]
    audio_window = condition_windows(split, "A", "beginning", 0.1)[0]["audio"]

    assert text_window == audio_window


def test_final_output_projects_intensity_using_method_polarity_without_changing_raw_value():
    from e_emotion.problem2_fair import finalize_prediction, project_intensity

    result = finalize_prediction(
        np.array([-1.5, 0.8, -0.4]),
        (Polarity.NEGATIVE, Polarity.NEUTRAL, Polarity.POSITIVE),
    )

    assert result.raw_intensity.tolist() == [-1.5, 0.8, -0.4]
    assert result.intensity[0] == pytest.approx(-1.5)
    assert result.intensity[1] == 0.0
    assert 0.0 < result.intensity[2] < 1e-4
    assert result.polarity == (Polarity.NEGATIVE, Polarity.NEUTRAL, Polarity.POSITIVE)
    assert np.array_equal(project_intensity(result.raw_intensity, result.polarity), result.intensity)


def test_aligned_processed_po_loader_preserves_physical_support_and_observation(tmp_path):
    from e_emotion.problem2_fair import load_problem2_dataset

    root = tmp_path / "aligned_po"
    _write_processed_po(root)

    dataset = load_problem2_dataset(root, view="aligned_po")

    assert dataset["train"].values["audio"].shape == (2, 50, 1)
    assert dataset["train"].physical_support["text"].shape == (2, 50)
    assert dataset["train"].observed["vision"].all()


def test_unaligned_processed_po_loader_preserves_fifty_five_hundred_layout(tmp_path):
    from e_emotion.problem2_fair import load_problem2_dataset

    root = tmp_path / "unaligned_po"
    _write_processed_po(root, unaligned=True)

    dataset = load_problem2_dataset(root, view="unaligned_po")

    assert dataset["train"].values["text"].shape == (2, 50, 2)
    assert dataset["train"].values["audio"].shape == (2, 500, 1)
    assert dataset["train"].physical_support["vision"].shape == (2, 500)


def test_processed_po_loader_accepts_object_id_arrays_but_keeps_feature_contract_strict(tmp_path):
    from e_emotion.problem2_fair import load_problem2_dataset

    root = tmp_path / "aligned_po"
    _write_processed_po(root, object_ids=True)

    dataset = load_problem2_dataset(root, view="aligned_po")

    assert dataset["train"].ids.tolist() == ["train0$_$0", "train1$_$0"]


def test_processed_po_loader_zeros_values_outside_observation_mask(tmp_path):
    from e_emotion.problem2_fair import load_problem2_dataset

    root = tmp_path / "aligned_po"
    _write_processed_po(root, missing_observation=True)

    dataset = load_problem2_dataset(root, view="aligned_po")

    assert dataset["train"].physical_support["audio"][0, 3]
    assert not dataset["train"].observed["audio"][0, 3]
    assert np.count_nonzero(dataset["train"].values["audio"][0, 3]) == 0
    assert dataset["train"].input_ids[0, 4] == 0


def test_processed_po_loader_rejects_observation_outside_physical_support(tmp_path):
    from e_emotion.problem2_fair import load_problem2_dataset

    root = tmp_path / "invalid"
    _write_processed_po(root, invalid_observation=True)

    with pytest.raises(ValueError, match="observation mask"):
        load_problem2_dataset(root, view="aligned_po")


def test_unaligned_window_pooling_happens_after_q2_observation_is_updated():
    from e_emotion.problem2_fair import Problem2Split, materialize_q2_condition, pool_unaligned_to_text_slots

    split = _unaligned_split()
    split = Problem2Split(
        ids=split.ids,
        values={
            **split.values,
            "audio": np.arange(500, dtype=np.float32).reshape(1, 500, 1),
        },
        physical_support=split.physical_support,
        observed=split.observed,
        input_ids=split.input_ids,
        regression=split.regression,
        classification=split.classification,
        view=split.view,
        split=split.split,
    )

    pooled = pool_unaligned_to_text_slots(materialize_q2_condition(split, "A", "beginning", 0.1))

    assert pooled.view == "unaligned_windowed"
    assert not pooled.observed["audio"][:, :5].any()
    assert np.all(pooled.values["audio"][:, :5] == 0.0)
    assert pooled.observed["audio"][:, 5:].all()
    assert pooled.values["audio"][0, 5, 0] == pytest.approx(54.5)


def test_q2_v2_manifest_covers_all_splits_and_has_stable_condition_count(tmp_path):
    from e_emotion.problem2_fair import build_q2_v2_manifest, load_problem2_dataset

    root = tmp_path / "aligned_po"
    _write_processed_po(root)
    dataset = load_problem2_dataset(root, view="aligned_po")

    manifest = build_q2_v2_manifest(dataset)

    assert manifest["protocol_version"] == "q2-continuous-local-v2"
    assert manifest["condition_count"] == 64
    assert len(manifest["entries"]) == 4 * 64
    assert len(manifest["mask_sha256"]) == 64


def test_q2_materialization_uses_frozen_manifest_windows(tmp_path):
    import copy

    from e_emotion.problem2_fair import Problem2Dataset, Problem2Split, build_q2_v2_manifest, materialize_q2_condition

    train = Problem2Split(**{**_aligned_split().__dict__, "split": "train"})
    valid = Problem2Split(**{**_aligned_split().__dict__, "split": "valid"})
    test = _aligned_split()
    dataset = Problem2Dataset({"train": train, "valid": valid, "test": test}, "aligned_po", tmp_path)
    manifest = build_q2_v2_manifest(dataset)
    entries = [copy.deepcopy(entry) for entry in manifest["entries"]
               if entry["split"] == "test" and entry["combination"] == "T"
               and entry["position"] == "beginning" and entry["requested_fraction"] == 0.1]
    entries[0]["window_by_modality"]["text"] = {"start": 8, "end": 10}
    entries[0]["synthetic_missing_counts"]["text"] = 2

    condition = materialize_q2_condition(test, "T", "beginning", 0.1, entries=entries)

    assert condition.observed["text"][0, :5].all()
    assert not condition.observed["text"][0, 8:10].any()
    assert not condition.observed["text"][1, :5].any()


def test_q2_v2_manifest_computes_windows_once_per_split_and_condition(tmp_path, monkeypatch):
    import e_emotion.problem2_fair.q2 as q2
    from e_emotion.problem2_fair import load_problem2_dataset

    root = tmp_path / "aligned_po"
    _write_processed_po(root)
    dataset = load_problem2_dataset(root, view="aligned_po")
    original = q2.condition_windows
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(q2, "condition_windows", counted)
    q2.build_q2_v2_manifest(dataset)

    assert calls == 3 * 64


def test_q2_v2_manifest_cache_reuses_matching_view_without_rebuilding(tmp_path, monkeypatch):
    import e_emotion.problem2_fair.q2 as q2
    from e_emotion.problem2_fair import ensure_q2_v2_manifest, load_problem2_dataset

    root = tmp_path / "aligned_po"
    _write_processed_po(root)
    dataset = load_problem2_dataset(root, view="aligned_po")
    cache = tmp_path / "q2.json"

    first = ensure_q2_v2_manifest(dataset, cache)
    monkeypatch.setattr(q2, "build_q2_v2_manifest", lambda _: (_ for _ in ()).throw(AssertionError("rebuilt")))
    second = ensure_q2_v2_manifest(dataset, cache)

    assert second["mask_sha256"] == first["mask_sha256"]


def test_q2_v2_manifest_cache_rejects_a_tampered_hash(tmp_path):
    from e_emotion.problem2_fair import ensure_q2_v2_manifest, load_problem2_dataset

    root = tmp_path / "aligned_po"
    _write_processed_po(root)
    dataset = load_problem2_dataset(root, view="aligned_po")
    cache = tmp_path / "q2.json"
    ensure_q2_v2_manifest(dataset, cache)
    payload = __import__("json").loads(cache.read_text(encoding="utf-8"))
    payload["mask_sha256"] = "0" * 64
    cache.write_text(__import__("json").dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="hash"):
        ensure_q2_v2_manifest(dataset, cache)


def test_q2_v2_never_turns_an_observed_modality_into_whole_modality_missingness():
    from e_emotion.problem2_fair import Problem2Split, materialize_q2_condition

    split = _aligned_split()
    observed = {name: np.zeros_like(mask) for name, mask in split.observed.items()}
    observed["text"][:, 0] = True
    split = Problem2Split(
        ids=split.ids,
        values=split.values,
        physical_support=split.physical_support,
        observed=observed,
        input_ids=split.input_ids,
        regression=split.regression,
        classification=split.classification,
        view=split.view,
        split=split.split,
    )

    condition = materialize_q2_condition(split, "T", "beginning", 0.5)

    assert condition.observed["text"].any(axis=1).all()


def test_baseline_run_trains_adapter_and_writes_clean_and_q2_v2_artifacts(tmp_path):
    from e_emotion.problem2_fair import BaselineRun, Problem2Dataset, Problem2Split, RawMethodPrediction

    class ToyAdapter:
        method_id = "toy"
        prepared_views = []

        def reencode_text(self, split):
            return split

        def train(self, dataset, *, seed, run_dir):
            self.trained = (dataset.view, seed, run_dir)
            checkpoint = run_dir / "toy.checkpoint"
            checkpoint.write_bytes(b"toy")
            return checkpoint

        def predict(self, split, checkpoint):
            polarity = tuple(Polarity.from_value(value) for value in split.classification)
            return RawMethodPrediction(
                raw_intensity=np.asarray(split.regression, dtype=np.float64),
                polarity=polarity,
                decision_source="native_logits_argmax",
            )

    train = _aligned_split()
    train = Problem2Split(**{**train.__dict__, "split": "train"})
    valid = Problem2Split(**{**_aligned_split().__dict__, "split": "valid"})
    test = _aligned_split()
    dataset = Problem2Dataset({"train": train, "valid": valid, "test": test}, "aligned_po", tmp_path)
    adapter = ToyAdapter()

    shared_manifest = tmp_path / "shared" / "aligned_po_q2_v2.json"
    result = BaselineRun(adapter).execute(
        dataset,
        seed=2026,
        run_dir=tmp_path / "run",
        manifest_path=shared_manifest,
    )

    assert adapter.trained[0:2] == ("aligned_po", 2026)
    assert result["clean"]["n"] == 2
    assert result["q2"]["n_conditions"] == 64
    assert result["q2"]["n_predictions"] == 128
    assert (tmp_path / "run" / "metrics.json").is_file()
    assert (tmp_path / "run" / "q2_v2" / "predictions.csv").is_file()
    assert shared_manifest.is_file()


def test_runner_refuses_text_missingness_for_adapter_without_required_reencoder(tmp_path):
    from e_emotion.problem2_fair import BaselineRun, Problem2Dataset, Problem2Split, RawMethodPrediction

    class MissingReencoderAdapter:
        method_id = "toy"
        requires_text_reencoding = True

        def train(self, dataset, *, seed, run_dir):
            checkpoint = run_dir / "checkpoint"
            checkpoint.write_bytes(b"checkpoint")
            return checkpoint

        def predict(self, split, checkpoint):
            return RawMethodPrediction(
                raw_intensity=np.asarray(split.regression, dtype=np.float64),
                polarity=tuple(Polarity.from_value(value) for value in split.classification),
                decision_source="native_logits_argmax",
            )

    train = Problem2Split(**{**_aligned_split().__dict__, "split": "train"})
    valid = Problem2Split(**{**_aligned_split().__dict__, "split": "valid"})
    dataset = Problem2Dataset({"train": train, "valid": valid, "test": _aligned_split()}, "aligned_po", tmp_path)

    with pytest.raises(ValueError, match="reencode_text"):
        BaselineRun(MissingReencoderAdapter()).execute(dataset, seed=2026, run_dir=tmp_path / "run")


def test_second_wave_method_directories_are_no_longer_frozen(tmp_path):
    from e_emotion.problem2_fair.catalog import is_frozen_output_path

    assert not is_frozen_output_path(tmp_path / "MulT" / "runs" / "problem2_fair_v1_seed1")
    assert not is_frozen_output_path(tmp_path / "CMAD" / "runs" / "aligned_po_q2_v2.json")


def test_windowed_adapter_uses_aligned_view_without_pooling(tmp_path):
    from e_emotion.problem2_fair import BaselineRun, Problem2Dataset, Problem2Split, RawMethodPrediction

    class WindowedAdapter:
        method_id = "toy"
        input_layout = "unaligned_windowed"

        def reencode_text(self, split):
            return split

        def train(self, dataset, *, seed, run_dir):
            checkpoint = run_dir / "checkpoint"
            checkpoint.write_bytes(b"checkpoint")
            return checkpoint

        def predict(self, split, checkpoint):
            assert split.view == "aligned_po"
            return RawMethodPrediction(np.asarray(split.regression), tuple(Polarity.from_value(value) for value in split.classification), "native")

    train = Problem2Split(**{**_aligned_split().__dict__, "split": "train"})
    valid = Problem2Split(**{**_aligned_split().__dict__, "split": "valid"})
    dataset = Problem2Dataset({"train": train, "valid": valid, "test": _aligned_split()}, "aligned_po", tmp_path)

    assert BaselineRun(WindowedAdapter()).execute(dataset, seed=2026, run_dir=tmp_path / "run")["q2"]["n_conditions"] == 64


def test_method_catalog_exposes_second_wave_methods_for_public_runs():
    from e_emotion.problem2_fair import active_method_specs, method_spec

    active = {spec.method_id for spec in active_method_specs()}

    assert {"tfn", "aumdf", "qa_moe", "ebmc"} <= active
    assert "problem2_main" not in active
    assert {"concat_mlp", "early_fusion_gru", "ef_lstm", "mult", "p_rmf", "cmad"} <= active
    assert method_spec("ef_lstm").status == "ready"
    assert method_spec("mult").status == "ready"
    assert method_spec("cica").status == "paper_only"
    assert method_spec("qa_moe").status == "ready"


def test_fair_protocol_view_roots_are_the_approved_processed_po_locations():
    from e_emotion.problem2_fair import default_view_root

    assert default_view_root("aligned_po").as_posix().endswith("对齐版本_retrain_20260925/processed_po")
    assert default_view_root("unaligned_po").as_posix().endswith("未对齐版本/processed_po")


def test_cli_baseline_validate_prints_q2_v2_manifest(tmp_path, capsys):
    from e_emotion.cli.main import main

    root = tmp_path / "aligned_po"
    _write_processed_po(root)

    assert main(["baseline", "validate", "--view", "aligned_po", "--data-root", str(root)]) == 0

    output = capsys.readouterr().out
    assert '"condition_count": 64' in output


def test_cli_baseline_validate_does_not_require_the_general_app_config(tmp_path, capsys):
    from e_emotion.cli.main import main

    root = tmp_path / "aligned_po"
    _write_processed_po(root)

    assert main([
        "baseline", "--config", str(tmp_path / "missing.yaml"), "validate",
        "--view", "aligned_po", "--data-root", str(root),
    ]) == 0

    assert '"condition_count": 64' in capsys.readouterr().out


def test_cli_baseline_run_allows_second_wave_method_to_reach_adapter_loading(tmp_path, capsys):
    from e_emotion.cli.main import main

    root = tmp_path / "aligned_po"
    _write_processed_po(root)

    assert main([
        "baseline", "run", "--method", "ef_lstm", "--view", "aligned_po",
        "--data-root", str(root), "--artifact-root", str(tmp_path / "artifacts"),
        "--run-dir", str(tmp_path / "wrong"),
    ]) == 2

    assert "ArtifactStore" in capsys.readouterr().out


def test_cli_baseline_run_refuses_non_ready_method_before_loading_adapter(tmp_path, capsys):
    from e_emotion.cli.main import main

    assert main([
        "baseline", "run", "--method", "cica", "--view", "aligned_po",
        "--data-root", str(tmp_path), "--artifact-root", str(tmp_path / "artifacts"),
    ]) == 2

    assert "paper_only" in capsys.readouterr().out


def test_cli_baseline_verify_rejects_missing_q2_manifest(tmp_path, capsys):
    from e_emotion.cli.main import main

    run = tmp_path / "run"
    q2 = run / "q2_v2"
    q2.mkdir(parents=True)
    (run / "metrics.json").write_text(
        __import__("json").dumps({"protocol_version": "problem2-fair-v1", "checkpoint_sha256": "a" * 64}),
        encoding="utf-8",
    )
    (q2 / "metrics.json").write_text(
        __import__("json").dumps({"n_conditions": 64, "n_predictions": 1, "mask_sha256": "a" * 64}),
        encoding="utf-8",
    )
    (q2 / "predictions.csv").write_text("id,intensity,polarity\ns,0.0,Neutral\n", encoding="utf-8")

    assert main(["baseline", "verify", "--run-dir", str(run)]) == 2

    assert "manifest" in capsys.readouterr().out


def test_cli_baseline_verify_accepts_complete_q2_records_with_repeated_sample_ids(tmp_path, capsys):
    from e_emotion.cli.main import main
    from e_emotion.problem2_fair import BaselineRun, Problem2Dataset, Problem2Split, RawMethodPrediction

    class ToyAdapter:
        method_id = "toy"

        def reencode_text(self, split):
            return split

        def train(self, dataset, *, seed, run_dir):
            checkpoint = run_dir / "checkpoint"
            checkpoint.write_bytes(b"checkpoint")
            return checkpoint

        def predict(self, split, checkpoint):
            return RawMethodPrediction(
                np.asarray(split.regression),
                tuple(Polarity.from_value(value) for value in split.classification),
                "native",
            )

    train = Problem2Split(**{**_aligned_split().__dict__, "split": "train"})
    valid = Problem2Split(**{**_aligned_split().__dict__, "split": "valid"})
    dataset = Problem2Dataset({"train": train, "valid": valid, "test": _aligned_split()}, "aligned_po", tmp_path)
    run = tmp_path / "run"
    BaselineRun(ToyAdapter()).execute(dataset, seed=2026, run_dir=run)

    assert main(["baseline", "verify", "--run-dir", str(run)]) == 0
    assert '"status": "ok"' in capsys.readouterr().out


def test_cli_baseline_verify_rejects_a_modified_checkpoint(tmp_path, capsys):
    from e_emotion.cli.main import main
    from e_emotion.problem2_fair import BaselineRun, Problem2Dataset, Problem2Split, RawMethodPrediction

    class ToyAdapter:
        method_id = "toy"

        def reencode_text(self, split):
            return split

        def train(self, dataset, *, seed, run_dir):
            checkpoint = run_dir / "checkpoint"
            checkpoint.write_bytes(b"checkpoint")
            return checkpoint

        def predict(self, split, checkpoint):
            return RawMethodPrediction(
                np.asarray(split.regression),
                tuple(Polarity.from_value(value) for value in split.classification),
                "native",
            )

    train = Problem2Split(**{**_aligned_split().__dict__, "split": "train"})
    valid = Problem2Split(**{**_aligned_split().__dict__, "split": "valid"})
    dataset = Problem2Dataset({"train": train, "valid": valid, "test": _aligned_split()}, "aligned_po", tmp_path)
    run = tmp_path / "run"
    BaselineRun(ToyAdapter()).execute(dataset, seed=2026, run_dir=run)
    (run / "checkpoint").write_bytes(b"modified")

    assert main(["baseline", "verify", "--run-dir", str(run)]) == 2
    assert "checkpoint" in capsys.readouterr().out


def test_cli_baseline_verify_rejects_modified_source_npz(tmp_path, capsys):
    from e_emotion.cli.main import main
    from e_emotion.problem2_fair import BaselineRun, RawMethodPrediction, load_problem2_dataset

    class ToyAdapter:
        method_id = "toy"

        def reencode_text(self, split):
            return split

        def train(self, dataset, *, seed, run_dir):
            checkpoint = run_dir / "checkpoint"
            checkpoint.write_bytes(b"checkpoint")
            return checkpoint

        def predict(self, split, checkpoint):
            return RawMethodPrediction(
                np.asarray(split.regression),
                tuple(Polarity.from_value(value) for value in split.classification),
                "native",
            )

    root = tmp_path / "aligned_po"
    _write_processed_po(root)
    dataset = load_problem2_dataset(root, view="aligned_po")
    run = tmp_path / "run"
    BaselineRun(ToyAdapter()).execute(dataset, seed=2026, run_dir=run)
    assert main(["baseline", "verify", "--run-dir", str(run)]) == 0
    with (root / "train.npz").open("ab") as stream:
        stream.write(b"modified")

    assert main(["baseline", "verify", "--run-dir", str(run)]) == 2
    assert "data hash" in capsys.readouterr().out


def test_cli_baseline_verify_recomputes_q2_metrics_from_source_truth(tmp_path, capsys):
    from e_emotion.cli.main import main
    from e_emotion.problem2_fair import BaselineRun, RawMethodPrediction, load_problem2_dataset

    class ToyAdapter:
        method_id = "toy"

        def reencode_text(self, split):
            return split

        def train(self, dataset, *, seed, run_dir):
            checkpoint = run_dir / "checkpoint"
            checkpoint.write_bytes(b"checkpoint")
            return checkpoint

        def predict(self, split, checkpoint):
            return RawMethodPrediction(np.asarray(split.regression),
                tuple(Polarity.from_value(value) for value in split.classification), "native")

    root = tmp_path / "aligned_po"
    _write_processed_po(root)
    run = tmp_path / "run"
    BaselineRun(ToyAdapter()).execute(load_problem2_dataset(root, view="aligned_po"), seed=2026, run_dir=run)
    assert main(["baseline", "verify", "--run-dir", str(run)]) == 0
    q2_path = run / "q2_v2" / "metrics.json"
    payload = __import__("json").loads(q2_path.read_text(encoding="utf-8"))
    payload["conditions"][0]["mae"] += 0.25
    q2_path.write_text(__import__("json").dumps(payload), encoding="utf-8")

    assert main(["baseline", "verify", "--run-dir", str(run)]) == 2
    assert "metric" in capsys.readouterr().out


def test_verify_run_reopens_hash_matched_default_view_for_early_artifacts(tmp_path, monkeypatch):
    import json

    from e_emotion.problem2_fair import BaselineRun, RawMethodPrediction, load_problem2_dataset
    from e_emotion.problem2_fair.verification import verify_run

    class ToyAdapter:
        method_id = "toy"

        def reencode_text(self, split):
            return split

        def train(self, dataset, *, seed, run_dir):
            checkpoint = run_dir / "checkpoint"
            checkpoint.write_bytes(b"checkpoint")
            return checkpoint

        def predict(self, split, checkpoint):
            return RawMethodPrediction(
                np.asarray(split.regression),
                tuple(Polarity.from_value(value) for value in split.classification),
                "native",
            )

    source = tmp_path / "aligned_po"
    _write_processed_po(source)
    run = tmp_path / "run"
    BaselineRun(ToyAdapter()).execute(load_problem2_dataset(source, view="aligned_po"), seed=2026, run_dir=run)
    for name in ("metrics.json", "protocol_manifest.json"):
        path = run / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("data_root")
        path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr("e_emotion.problem2_fair.verification.default_view_root", lambda view: source)

    assert verify_run(run, require_source=True)["status"] == "ok"


def test_cli_baseline_verify_rejects_intensity_inconsistent_with_polarity(tmp_path, capsys):
    import csv

    from e_emotion.cli.main import main
    from e_emotion.problem2_fair import BaselineRun, Problem2Dataset, Problem2Split, RawMethodPrediction

    class ToyAdapter:
        method_id = "toy"

        def reencode_text(self, split):
            return split

        def train(self, dataset, *, seed, run_dir):
            checkpoint = run_dir / "checkpoint"
            checkpoint.write_bytes(b"checkpoint")
            return checkpoint

        def predict(self, split, checkpoint):
            return RawMethodPrediction(np.asarray(split.regression),
                tuple(Polarity.from_value(value) for value in split.classification), "native")

    train = Problem2Split(**{**_aligned_split().__dict__, "split": "train"})
    valid = Problem2Split(**{**_aligned_split().__dict__, "split": "valid"})
    dataset = Problem2Dataset({"train": train, "valid": valid, "test": _aligned_split()}, "aligned_po", tmp_path)
    run = tmp_path / "run"
    BaselineRun(ToyAdapter()).execute(dataset, seed=2026, run_dir=run)
    q2_csv = run / "q2_v2" / "predictions.csv"
    with q2_csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["intensity"] = "0.5"
    with q2_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    assert main(["baseline", "verify", "--run-dir", str(run)]) == 2
    assert "project_intensity" in capsys.readouterr().out


def test_cli_report_requires_reopenable_source_truth(tmp_path, capsys):
    from e_emotion.cli.main import main
    from e_emotion.problem2_fair import BaselineRun, Problem2Dataset, Problem2Split, RawMethodPrediction

    class ToyAdapter:
        method_id = "toy"

        def reencode_text(self, split):
            return split

        def train(self, dataset, *, seed, run_dir):
            checkpoint = run_dir / "checkpoint"
            checkpoint.write_bytes(b"checkpoint")
            return checkpoint

        def predict(self, split, checkpoint):
            return RawMethodPrediction(
                np.asarray(split.regression),
                tuple(Polarity.from_value(value) for value in split.classification),
                "native",
            )

    train = Problem2Split(**{**_aligned_split().__dict__, "split": "train"})
    valid = Problem2Split(**{**_aligned_split().__dict__, "split": "valid"})
    dataset = Problem2Dataset({"train": train, "valid": valid, "test": _aligned_split()}, "aligned_po", tmp_path)
    runs = []
    for seed in (1, 2, 3):
        run = tmp_path / f"seed{seed}"
        BaselineRun(ToyAdapter()).execute(dataset, seed=seed, run_dir=run)
        runs.append(run)

    args = ["baseline", "report"]
    for run in runs:
        args.extend(("--run-dir", str(run)))
    assert main(args) == 2
    assert "source NPZ" in capsys.readouterr().out


def test_report_aggregates_only_matching_method_and_view_runs(tmp_path):
    from e_emotion.problem2_fair import FAIR_SEEDS, Q2_V2_CONDITIONS, summarize_runs

    paths = []
    for seed, score in zip(FAIR_SEEDS, (0.5, 0.55, 0.6), strict=True):
        root = tmp_path / str(seed)
        root.mkdir()
        (root / "metrics.json").write_text(
            __import__("json").dumps({
                "protocol_version": "problem2-fair-v1", "method": "aumdf", "view": "aligned_po", "seed": seed,
                "clean": {"accuracy": score, "macro_f1": score, "weighted_f1": score, "mae": 1 - score, "pearson": score},
            }),
            encoding="utf-8",
        )
        q2 = root / "q2_v2"
        q2.mkdir()
        (q2 / "metrics.json").write_text(
            __import__("json").dumps({
                "n_conditions": 64,
                "mask_sha256": "a" * 64,
                "method": "aumdf", "view": "aligned_po", "seed": seed,
                    "conditions": [
                        {"combination": combination, "position": position, "requested_fraction": fraction, "accuracy": score}
                        for combination, position, fraction in Q2_V2_CONDITIONS
                    ],
            }),
            encoding="utf-8",
        )
        paths.append(root)

    report = summarize_runs(paths)

    assert report["method"] == "aumdf"
    assert report["view"] == "aligned_po"
    assert report["seed_count"] == 3
    assert report["clean"]["accuracy"]["mean"] == pytest.approx(0.55)


def test_report_rejects_runs_with_different_q2_mask_hashes(tmp_path):
    from e_emotion.problem2_fair import FAIR_SEEDS, Q2_V2_CONDITIONS, summarize_runs

    paths = []
    for seed in FAIR_SEEDS:
        root = tmp_path / str(seed)
        root.mkdir()
        (root / "metrics.json").write_text(
            __import__("json").dumps({
                "protocol_version": "problem2-fair-v1", "method": "aumdf", "view": "aligned_po", "seed": seed,
                "clean": {"accuracy": 0.5, "macro_f1": 0.5, "weighted_f1": 0.5, "mae": 0.5, "pearson": 0.5},
            }), encoding="utf-8",
        )
        q2 = root / "q2_v2"
        q2.mkdir()
        (q2 / "metrics.json").write_text(
            __import__("json").dumps({
                "n_conditions": 64,
                "mask_sha256": ("a" if seed != 3 else "b") * 64,
                "method": "aumdf", "view": "aligned_po", "seed": seed,
                "conditions": [
                    {"combination": combination, "position": position, "requested_fraction": fraction, "accuracy": 0.5}
                    for combination, position, fraction in Q2_V2_CONDITIONS
                ],
            }), encoding="utf-8",
        )
        paths.append(root)

    with pytest.raises(ValueError, match="mask hash"):
        summarize_runs(paths)


def test_report_rejects_shared_but_nonstandard_q2_condition_set(tmp_path):
    from e_emotion.problem2_fair import FAIR_SEEDS, Q2_V2_CONDITIONS, summarize_runs

    wrong_conditions = list(Q2_V2_CONDITIONS)
    wrong_conditions[0] = ("T", None, 0.0)
    paths = []
    for seed in FAIR_SEEDS:
        root = tmp_path / str(seed)
        root.mkdir()
        (root / "metrics.json").write_text(
            __import__("json").dumps({
                "protocol_version": "problem2-fair-v1", "method": "aumdf", "view": "aligned_po", "seed": seed,
                "clean": {"accuracy": 0.5, "macro_f1": 0.5, "weighted_f1": 0.5, "mae": 0.5, "pearson": 0.5},
            }), encoding="utf-8",
        )
        q2 = root / "q2_v2"
        q2.mkdir()
        (q2 / "metrics.json").write_text(
            __import__("json").dumps({
                "n_conditions": 64, "mask_sha256": "a" * 64,
                "method": "aumdf", "view": "aligned_po", "seed": seed,
                "conditions": [
                    {"combination": combination, "position": position, "requested_fraction": fraction, "accuracy": 0.5}
                    for combination, position, fraction in wrong_conditions
                ],
            }), encoding="utf-8",
        )
        paths.append(root)

    with pytest.raises(ValueError, match="canonical"):
        summarize_runs(paths)


def test_report_rejects_q2_identity_that_disagrees_with_run_summary(tmp_path):
    from e_emotion.problem2_fair import FAIR_SEEDS, Q2_V2_CONDITIONS, summarize_runs

    paths = []
    for seed in FAIR_SEEDS:
        root = tmp_path / str(seed)
        root.mkdir()
        (root / "metrics.json").write_text(
            __import__("json").dumps({
                "protocol_version": "problem2-fair-v1", "method": "aumdf", "view": "aligned_po", "seed": seed,
                "clean": {"accuracy": 0.5, "macro_f1": 0.5, "weighted_f1": 0.5, "mae": 0.5, "pearson": 0.5},
            }), encoding="utf-8",
        )
        q2 = root / "q2_v2"
        q2.mkdir()
        (q2 / "metrics.json").write_text(
            __import__("json").dumps({
                "n_conditions": 64, "mask_sha256": "a" * 64,
                "method": "other" if seed == 3 else "aumdf", "view": "aligned_po", "seed": seed,
                "conditions": [
                    {"combination": combination, "position": position, "requested_fraction": fraction, "accuracy": 0.5}
                    for combination, position, fraction in Q2_V2_CONDITIONS
                ],
            }), encoding="utf-8",
        )
        paths.append(root)

    with pytest.raises(ValueError, match="identity"):
        summarize_runs(paths)
