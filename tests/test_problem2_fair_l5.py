from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from e_emotion.problem2_fair.core import Problem2Split


def _split(*, view="aligned_po", length=50):
    values = {
        "text": np.ones((2, 50, 4), dtype=np.float32),
        "audio": np.ones((2, length, 2), dtype=np.float32),
        "vision": np.ones((2, length, 3), dtype=np.float32),
    }
    support = {name: np.ones(value.shape[:2], dtype=bool) for name, value in values.items()}
    observed = {name: value.copy() for name, value in support.items()}
    observed["text"][:, 10:15] = False
    observed["audio"][:, : length // 10] = False
    tokens = np.tile(np.arange(100, 150, dtype=np.int64), (2, 1))
    return Problem2Split(
        ids=np.array(["a", "b"]), values=values, physical_support=support,
        observed=observed, input_ids=tokens,
        regression=np.array([-1.0, 1.0], dtype=np.float32),
        classification=np.array([0, 2], dtype=np.int64),
        view=view, split="test",
    )


def _source(tmp_path, method):
    keys = {"emoe": "emoe", "cacr": "CACR", "tlra": "TLRA", "mruf": "mruf"}
    root = tmp_path / method.upper()
    root.mkdir()
    config = {
        "datasetCommonParams": {"mosei": {"aligned": {
            "feature_dims": [768, 74, 35], "seq_lens": [50, 50, 50],
            "KeyEval": "F1_score" if method == "tlra" else "Loss",
        }}},
        keys[method]: {
            "commonParams": {"need_data_aligned": True, "use_bert": True, "use_finetune": True},
            "datasetParams": {"mosei": {"batch_size": 8, "learning_rate": 0.001}},
        },
    }
    (root / "cpmcm_config.json").write_text(json.dumps(config), encoding="utf-8")
    return root


@pytest.mark.parametrize("method", ["emoe", "cacr", "tlra", "mruf"])
def test_l5_config_uses_original_method_key_and_frozen_text_path(tmp_path, method):
    from e_emotion.problem2_fair.methods.l5 import L5Adapter

    adapter = L5Adapter(method, source_root=_source(tmp_path, method))
    args = adapter.resolve_config(_split())

    assert adapter.input_layout == "unaligned_windowed"
    assert args["model_name"] == {"emoe": "emoe", "cacr": "CACR", "tlra": "TLRA", "mruf": "mruf"}[method]
    assert args["feature_dims"] == [4, 2, 3]
    assert args["seq_lens"] == [50, 50, 50]
    assert args["need_data_aligned"] is True
    assert args["use_bert"] is False
    assert args["dataset_name"] == ("cmumosei" if method == "cacr" else "mosei")
    assert args["KeyEval"] == ("F1_score" if method == "tlra" else "Loss")


@pytest.mark.parametrize("method", ["emoe", "cacr", "tlra", "mruf"])
def test_l5_reencoder_receives_deleted_tokens_and_masks_outputs(tmp_path, method):
    from e_emotion.problem2_fair.methods.l5 import L5Adapter

    received = []

    def encoder(split):
        received.append(split.input_ids.copy())
        return np.full(split.values["text"].shape, 7, dtype=np.float32)

    adapter = L5Adapter(method, source_root=_source(tmp_path, method), text_reencoder=encoder)
    result = adapter.reencode_text(_split())

    assert not received[0][:, 10:15].any()
    assert not result.values["text"][:, 10:15].any()
    assert np.all(result.values["text"][:, :10] == 7)
    assert not result.values["audio"][:, :5].any()
    assert np.all(result.values["vision"] == 1)


def test_l5_unaligned_masking_pools_after_native_observation_deletion(tmp_path):
    from e_emotion.problem2_fair.methods.l5 import L5Adapter

    split = _split(view="unaligned_po", length=500)
    audio = np.arange(500, dtype=np.float32)[None, :, None].repeat(2, axis=0)
    audio = np.repeat(audio, 2, axis=2)
    split = Problem2Split(**{**split.__dict__, "values": {**split.values, "audio": audio}})
    adapter = L5Adapter("mruf", source_root=_source(tmp_path, "mruf"),
                        text_reencoder=lambda value: value.values["text"])
    prepared = adapter.prepare_features(split)

    assert prepared["audio"].shape == (2, 50, 2)
    assert not prepared["audio"][:, :5].any()
    assert prepared["audio"][0, 5, 0] == pytest.approx(54.5)


def test_l5_reencoder_fails_closed_when_missing(tmp_path):
    from e_emotion.problem2_fair.methods.l5 import L5Adapter

    adapter = L5Adapter("emoe", source_root=_source(tmp_path, "emoe"))
    with pytest.raises(ValueError, match="text_reencoder"):
        adapter.reencode_text(_split())


def test_l5_native_prediction_uses_method_specific_key_and_tlra_missing_branch(tmp_path):
    from e_emotion.problem2_fair.methods.l5 import L5Adapter

    calls = []

    class Model:
        def __call__(self, *features, **kwargs):
            calls.append(kwargs)
            return {"output_logit": np.array([[0.5]]), "logits_c": np.array([[0.75]])}

    for method in ("emoe", "cacr", "tlra", "mruf"):
        adapter = L5Adapter(method, source_root=_source(tmp_path, method))
        output = adapter.native_forward(Model(), (1, 2, 3))
        expected = 0.75 if method == "emoe" else 0.5
        assert float(output.reshape(-1)[0]) == expected
    assert calls[2] == {"role": "missing", "keep_modes": "TAV", "freeze_bank": True}
    assert calls[0] == calls[1] == calls[3] == {}


def test_l5_valid_interval_is_deterministic_and_uses_valid_only():
    from e_emotion.problem2_fair.methods.l5 import valid_interval

    raw = np.array([-2.0, -0.2, 0.0, 0.2, 2.0])
    truth = np.array([0, 0, 1, 2, 2])
    first = valid_interval(raw, truth)

    assert first == valid_interval(raw, truth)
    assert first["fit_split"] == "valid"
    assert first["objective"] == "macro_f1"
    assert first["lower"] <= 0 <= first["upper"]


def test_l5_rejects_missing_native_source(tmp_path):
    from e_emotion.problem2_fair.methods.l5 import L5Adapter

    adapter = L5Adapter("emoe", source_root=tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="native source"):
        adapter.resolve_config(_split())


def test_cacr_native_nontext_and_loss_branches_match_mosei():
    root = Path(__file__).resolve().parents[1] / "references" / "vendor" / "CACR" / "trains" / "singleTask"
    files = (root / "model" / "CACR.py", root / "CACR.py")
    checked = 0
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.If) or node.lineno <= 24:
                continue
            expression = ast.unparse(node.test)
            if "dataset_name" not in expression:
                continue
            outcomes = []
            for name in ("mosei", "cmumosei"):
                args = SimpleNamespace(dataset_name=name)
                value = eval(compile(ast.Expression(node.test), str(path), "eval"),
                             {"args": args, "self": SimpleNamespace(args=args)})
                outcomes.append(bool(value))
            assert outcomes[0] == outcomes[1], f"{path}:{node.lineno} diverges"
            checked += 1
    assert checked >= 7


def test_l5_native_workdir_contains_relative_trainer_artifacts(tmp_path):
    from e_emotion.problem2_fair.methods.l5 import native_workdir

    before = Path.cwd()
    with native_workdir(tmp_path):
        assert Path.cwd() == tmp_path
        Path("pt").mkdir()
        Path("pt" , "best.pth").write_bytes(b"weights")
    assert Path.cwd() == before
    assert (tmp_path / "pt" / "best.pth").read_bytes() == b"weights"


def test_l5_native_workdir_restores_cwd_on_failure(tmp_path):
    from e_emotion.problem2_fair.methods.l5 import native_workdir

    before = Path.cwd()
    with pytest.raises(RuntimeError, match="broken"):
        with native_workdir(tmp_path):
            raise RuntimeError("broken")
    assert Path.cwd() == before
