from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from e_emotion.problem2_fair import Problem2Split


def _split(*, view="aligned_po", audio_length=50):
    text = np.ones((1, 50, 2), dtype=np.float32)
    audio = np.arange(audio_length, dtype=np.float32).reshape(1, audio_length, 1)
    vision = np.ones((1, audio_length, 1), dtype=np.float32)
    support = {
        "text": np.ones((1, 50), dtype=bool),
        "audio": np.ones((1, audio_length), dtype=bool),
        "vision": np.ones((1, audio_length), dtype=bool),
    }
    observed = {name: mask.copy() for name, mask in support.items()}
    observed["text"][0, 10:15] = False
    observed["audio"][0, : audio_length // 10] = False
    ids = np.arange(50, dtype=np.int64).reshape(1, 50) + 100
    ids[~observed["text"]] = 0
    return Problem2Split(
        ids=np.array(["sample$_$0"]),
        values={"text": text, "audio": audio, "vision": vision},
        physical_support=support,
        observed=observed,
        input_ids=ids,
        regression=np.array([1.0], dtype=np.float32),
        classification=np.array([2], dtype=np.int64),
        view=view,
        split="test",
    )


def _source(tmp_path):
    root = tmp_path / "MMSA"
    root.mkdir()
    config = {
        "datasetCommonParams": {
            "mosei": {
                "aligned": {"seq_lens": [50, 50, 50], "feature_dims": [2, 1, 1], "KeyEval": "Loss"},
                "unaligned": {"seq_lens": [50, 500, 500], "feature_dims": [2, 1, 1], "KeyEval": "Loss"},
            }
        },
    }
    for key, normalized in (("tfn", True), ("mfn", True), ("graph_mfn", False)):
        config[key] = {
            "commonParams": {"need_normalized": normalized, "early_stop": 8},
            "datasetParams": {"mosei": {"batch_size": 8}},
        }
    (root / "cpmcm_config.json").write_text(json.dumps(config), encoding="utf-8")
    return root


def test_factories_use_verified_native_mmsa_keys(tmp_path):
    from e_emotion.problem2_fair.methods.l2 import create_graph_mfn_adapter, create_mfn_adapter, create_tfn_adapter

    root = _source(tmp_path)
    for factory, method_id, native_class in (
        (create_tfn_adapter, "tfn", "TFN"),
        (create_mfn_adapter, "mfn", "MFN"),
        (create_graph_mfn_adapter, "graph_mfn_dfg", "Graph_MFN"),
    ):
        adapter = factory(source_root=root)
        assert adapter.method_id == method_id
        assert adapter.native_model_key in {"tfn", "mfn", "graph_mfn"}
        assert adapter.native_class_name == native_class
        assert adapter.input_layout == "unaligned_windowed"


def test_mfn_config_keeps_equal_length_sequence_for_native_model(tmp_path):
    from e_emotion.problem2_fair.methods.l2 import create_mfn_adapter

    adapter = create_mfn_adapter(source_root=_source(tmp_path))
    config = adapter.resolve_config(_split())
    batch = adapter.prepare_features(_split())

    assert config["need_normalized"] is False
    assert config["seq_lens"] == [50, 50, 50]
    assert batch["text"].shape[1] == batch["audio"].shape[1] == batch["vision"].shape[1] == 50
    assert batch["audio"][0, 0, 0] == 0.0


def test_tfn_uses_its_native_time_mean_after_observation_deletion(tmp_path):
    from e_emotion.problem2_fair.methods.l2 import create_tfn_adapter

    adapter = create_tfn_adapter(source_root=_source(tmp_path))
    batch = adapter.prepare_features(_split(view="unaligned_po", audio_length=500))

    assert batch["audio"].shape == (1, 1, 1)
    assert batch["audio"][0, 0, 0] == pytest.approx(np.arange(50, 500, dtype=np.float32).sum() / 500)


def test_text_reencoding_uses_deleted_ids_and_zeros_unobserved_output(tmp_path):
    from e_emotion.problem2_fair.methods.l2 import create_graph_mfn_adapter

    received = []

    def encoder(split):
        received.append(split.input_ids.copy())
        return np.full(split.values["text"].shape, 7, dtype=np.float32)

    adapter = create_graph_mfn_adapter(source_root=_source(tmp_path), text_reencoder=encoder)
    result = adapter.reencode_text(_split())

    assert not received[0][0, 10:15].any()
    assert not result.values["text"][0, 10:15].any()
    assert np.all(result.values["text"][0, :10] == 7)


def test_text_reencoding_restores_structural_tokens_around_physical_content(tmp_path):
    from e_emotion.problem2_fair.methods.l2 import create_mfn_adapter

    split = _split()
    support = {name: value.copy() for name, value in split.physical_support.items()}
    observed = {name: value.copy() for name, value in split.observed.items()}
    support["text"][:, 0] = False
    support["text"][:, 6:] = False
    observed["text"] &= support["text"]
    received = []
    split = Problem2Split(**{**split.__dict__, "physical_support": support, "observed": observed})
    adapter = create_mfn_adapter(source_root=_source(tmp_path), text_reencoder=lambda value: (received.append(value.input_ids.copy()) or np.zeros_like(value.values["text"])))

    adapter.reencode_text(split)

    assert received[0][0, 0] == 101
    assert received[0][0, 6] == 102


def test_adapter_refuses_predicting_before_frozen_text_encoder_is_available(tmp_path):
    from e_emotion.problem2_fair.methods.l2 import create_mfn_adapter

    adapter = create_mfn_adapter(source_root=_source(tmp_path))
    with pytest.raises(ValueError, match="text_reencoder"):
        adapter.reencode_text(_split())


def test_native_args_support_mmsa_mapping_and_attribute_access_without_easydict():
    from e_emotion.problem2_fair.methods.l2 import NativeArgs

    args = NativeArgs({"model_name": "mfn", "batch_size": 8})
    args.device = "cpu"

    assert args.model_name == args["model_name"] == "mfn"
    assert args.get("batch_size") == 8
    assert args["device"] == "cpu"


def test_native_loader_skips_eager_imports_of_unrelated_mmsa_models(tmp_path):
    from e_emotion.problem2_fair.methods.l2 import create_tfn_adapter

    root = _source(tmp_path)
    base = root / "src" / "MMSA"
    (base / "models" / "singleTask").mkdir(parents=True)
    (base / "models" / "subNets").mkdir(parents=True)
    (base / "trains" / "singleTask").mkdir(parents=True)
    (base / "utils").mkdir(parents=True)
    (base / "__init__.py").write_text("raise RuntimeError('unrelated eager import')\n", encoding="utf-8")
    (base / "models" / "subNets" / "FeatureNets.py").write_text(
        "class SubNet: pass\nclass TextSubNet: pass\n", encoding="utf-8"
    )
    (base / "models" / "singleTask" / "TFN.py").write_text("class TFN: pass\n", encoding="utf-8")
    (base / "utils" / "__init__.py").write_text("raise RuntimeError('unrelated GPU helper')\n", encoding="utf-8")
    (base / "utils" / "metricsTop.py").write_text("class MetricsTop: pass\n", encoding="utf-8")
    (base / "trains" / "singleTask" / "TFN.py").write_text(
        "from ...utils import MetricsTop, dict_to_str\nclass TFN: pass\n", encoding="utf-8"
    )

    before = set(sys.modules)
    try:
        model_class = create_tfn_adapter(source_root=root)._native_class(trainer=False)
        assert model_class.__name__ == "TFN"
        assert model_class.__module__ == "MMSA.models.singleTask.TFN"
        trainer_class = create_tfn_adapter(source_root=root)._native_class(trainer=True)
        assert trainer_class.__module__ == "MMSA.trains.singleTask.TFN"
    finally:
        for name in set(sys.modules) - before:
            if name == "MMSA" or name.startswith("MMSA."):
                sys.modules.pop(name, None)
