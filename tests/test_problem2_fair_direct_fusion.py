from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from e_emotion.problem2_fair.methods.masked_train import _take
from e_emotion.problem2_fair.views import Problem2Dataset
from tests.test_problem2_fair_masked_train import _encoder, _split


@pytest.mark.parametrize("method_id", ("concat_mlp", "early_fusion_gru"))
@pytest.mark.parametrize("view", ("aligned_po", "unaligned_po"))
def test_direct_fusion_adapter_trains_with_po_masks_and_uses_native_logits(tmp_path, method_id, view):
    from e_emotion.problem2_fair.methods.direct_fusion import DirectFusionAdapter

    dataset = Problem2Dataset(
        {name: _split(view, split=name, size=2) for name in ("train", "valid", "test")},
        view,
        tmp_path,
    )
    adapter = DirectFusionAdapter(
        method_id,
        text_reencoder=_encoder,
        device="cpu",
        epochs=2,
        patience=1,
        batch_size=2,
        hidden_dim=8,
    )
    checkpoint = adapter.train(dataset, seed=1, run_dir=tmp_path)
    prepared = adapter._prepare(dataset["test"])
    output = adapter.predict(prepared, checkpoint)
    metadata = json.loads((tmp_path / "direct_fusion.json").read_text(encoding="utf-8"))

    assert checkpoint.is_file()
    assert prepared.view == ("unaligned_windowed" if view == "unaligned_po" else "aligned_po")
    assert output.raw_intensity.shape == (2,)
    assert output.decision_source == "native_logits_argmax"
    assert metadata["selection"]["split"] == "valid"
    assert metadata["training"]["seed"] == 1
