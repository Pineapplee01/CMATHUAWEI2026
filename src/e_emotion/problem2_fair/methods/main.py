"""Evaluate the already trained Problem 2 main checkpoints under one protocol."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np

from e_emotion.contracts import Polarity
from e_emotion.problem2_fair.core import Problem2Split
from e_emotion.problem2_fair.run import RawMethodPrediction
from e_emotion.problem2_fair.views import Problem2Dataset


CPMCM_ROOT = Path("/user_home/gaojianan/CPMCM")
_CHECKPOINT_GROUPS = {
    "aligned_po": {
        2026: "retrain_f1_clssep_avP0_s2026",
        2027: "retrain_f1_clssep_avP0_s2027_2028",
        2028: "retrain_f1_clssep_avP0_s2027_2028",
        2029: "retrain_f1_clssep_avP0_s2029_2030",
        2030: "retrain_f1_clssep_avP0_s2029_2030",
    },
    "unaligned_po": {
        2026: "retrain_v2_unaligned_modalgate_s2026",
        2027: "retrain_v2_unaligned_modalgate_s2027_2030",
        2028: "retrain_v2_unaligned_modalgate_s2027_2030",
        2029: "retrain_v2_unaligned_modalgate_s2027_2030",
        2030: "retrain_v2_unaligned_modalgate_s2027_2030",
    },
}
HISTORICAL_MAIN_SEEDS = tuple(sorted(_CHECKPOINT_GROUPS["aligned_po"]))


def checkpoint_for(view: str, seed: int, *, root: Path = CPMCM_ROOT) -> Path:
    if view not in _CHECKPOINT_GROUPS or seed not in HISTORICAL_MAIN_SEEDS:
        raise ValueError(f"unsupported historical main-method checkpoint view/seed: {view}/{seed}")
    group = _CHECKPOINT_GROUPS[view][seed]
    return root / "AAAmodel" / "checkpoints" / "runs" / group / f"retrain_processed_po_seed{seed}.pt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenMainMethodAdapter:
    """Reuse verified native checkpoints; only the public evaluation is new."""

    method_id = "problem2_main"
    input_layout = "native"

    def __init__(self, *, cpmcm_root: Path = CPMCM_ROOT, device: str = "cuda:0", batch_size: int = 16) -> None:
        self.cpmcm_root = Path(cpmcm_root)
        self.device = device
        self.batch_size = batch_size
        self._view: str | None = None
        self._loaded_checkpoint: Path | None = None
        self._model = None

    def train(self, dataset: Problem2Dataset, *, seed: int, run_dir: Path) -> Path:
        """Adopt an existing checkpoint after checking its original training evidence."""
        import torch

        checkpoint = checkpoint_for(dataset.view, seed, root=self.cpmcm_root)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = payload.get("config", {})
        expected_format = "aligned_v4" if dataset.view == "aligned_po" else "unaligned_v1"
        if payload.get("format") != expected_format:
            raise ValueError("main checkpoint format does not match data view")
        if int(config.get("seed", -1)) != seed or Path(config.get("data_dir", "")).resolve() != dataset.root.resolve():
            raise ValueError("main checkpoint seed or processed_po root differs from fair dataset")
        if not config.get("finetune_bert", False):
            raise ValueError("main checkpoint must use online BERT text encoding")
        expected_hashes = {name: _sha256(dataset.root / f"{name}.npz") for name in ("train", "valid")}
        if payload.get("data_identity") != expected_hashes:
            raise ValueError("main checkpoint train/valid hashes differ from fair dataset")
        (run_dir / "checkpoint_source.json").write_text(
            json.dumps({
                "training_source": "existing_frozen_checkpoint",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "view": dataset.view,
                "seed": seed,
                "train_valid_sha256": expected_hashes,
                "selection_rule": config.get("selection_rule"),
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        self._view = dataset.view
        return checkpoint

    def reencode_text(self, split: Problem2Split) -> Problem2Split:
        """Restore structural tokens; the native model runs its BERT online."""
        tokens = np.array(split.input_ids, copy=True)
        tokens[~split.observed["text"]] = 0
        for row, support in enumerate(split.physical_support["text"]):
            positions = np.flatnonzero(support)
            if not len(positions):
                continue
            first, last = int(positions[0]), int(positions[-1])
            if first > 0:
                tokens[row, first - 1] = 101
            if last + 1 < tokens.shape[1]:
                tokens[row, last + 1] = 102
        return replace(split, input_ids=tokens)

    def _restore_model(self, checkpoint: Path, view: str):
        if self._loaded_checkpoint == checkpoint and self._model is not None:
            return self._model
        if str(self.cpmcm_root) not in sys.path:
            sys.path.insert(0, str(self.cpmcm_root))
        module = "problem2_retrain.training" if view == "aligned_po" else "problem2_retrain_v2.training"
        restore = importlib.import_module(module).restore
        model, config = restore(checkpoint, self.device)
        if not config.get("finetune_bert", False):
            raise ValueError("main checkpoint does not perform online text reencoding")
        self._model = model.eval()
        self._loaded_checkpoint = checkpoint
        return self._model

    def predict(self, split: Problem2Split, checkpoint: Path) -> RawMethodPrediction:
        import torch

        checkpoint = Path(checkpoint).resolve()
        model = self._restore_model(checkpoint, split.view)
        raw_values: list[float] = []
        class_values: list[int] = []
        model_dtype = next(model.parameters()).dtype
        with torch.no_grad():
            for start in range(0, split.size, self.batch_size):
                end = min(start + self.batch_size, split.size)
                batch = {
                    "XT": torch.as_tensor(split.values["text"][start:end], device=self.device, dtype=model_dtype),
                    "XA": torch.as_tensor(split.values["audio"][start:end], device=self.device, dtype=model_dtype),
                    "XV": torch.as_tensor(split.values["vision"][start:end], device=self.device, dtype=model_dtype),
                    "I": torch.as_tensor(split.input_ids[start:end], device=self.device, dtype=torch.int64),
                }
                if split.view == "aligned_po":
                    batch["P"] = torch.as_tensor(np.stack([split.physical_support[name][start:end] for name in ("text", "audio", "vision")], axis=1), device=self.device)
                    batch["O"] = torch.as_tensor(np.stack([split.observed[name][start:end] for name in ("text", "audio", "vision")], axis=1), device=self.device)
                else:
                    for code, name in zip("TAV", ("text", "audio", "vision")):
                        batch[f"P_{code}"] = torch.as_tensor(split.physical_support[name][start:end], device=self.device)
                        batch[f"O_{code}"] = torch.as_tensor(split.observed[name][start:end], device=self.device)
                output = model(batch)
                raw_values.extend(output["intensity"].reshape(-1).detach().cpu().tolist())
                class_values.extend(output["logits"].argmax(-1).reshape(-1).detach().cpu().tolist())
        raw = np.asarray(raw_values, dtype=np.float64)
        if raw.shape != (split.size,) or not np.isfinite(raw).all():
            raise ValueError("main method returned nonfinite or incomplete intensity predictions")
        return RawMethodPrediction(
            raw_intensity=raw,
            polarity=tuple(Polarity.from_value(value) for value in class_values),
            decision_source="native_logits_argmax",
        )


def create_adapter() -> FrozenMainMethodAdapter:
    return FrozenMainMethodAdapter()


__all__ = ["HISTORICAL_MAIN_SEEDS", "FrozenMainMethodAdapter", "checkpoint_for", "create_adapter"]
