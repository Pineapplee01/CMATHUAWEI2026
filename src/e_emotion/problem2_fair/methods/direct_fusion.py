"""Problem 2 adapters for the two project-owned direct-fusion controls."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import random
from typing import Callable

import numpy as np

from e_emotion.baselines.direct_fusion import ConcatMLP, EarlyFusionGRU
from e_emotion.contracts import Polarity
from e_emotion.problem2_fair.core import MODALITIES, Problem2Split
from e_emotion.problem2_fair.methods.l2 import DEFAULT_BERT_ROOT, FrozenBertTextReencoder
from e_emotion.problem2_fair.q2 import pool_unaligned_to_text_slots
from e_emotion.problem2_fair.run import RawMethodPrediction
from e_emotion.problem2_fair.views import Problem2Dataset


def _take(split: Problem2Split, indices: np.ndarray) -> Problem2Split:
    return replace(
        split,
        ids=split.ids[indices],
        values={name: split.values[name][indices] for name in MODALITIES},
        physical_support={name: split.physical_support[name][indices] for name in MODALITIES},
        observed={name: split.observed[name][indices] for name in MODALITIES},
        input_ids=split.input_ids[indices],
        regression=split.regression[indices],
        classification=split.classification[indices],
    )


class DirectFusionAdapter:
    """Train one direct-fusion model through the shared P/O and Q2 protocol."""

    input_layout = "unaligned_windowed"

    def __init__(
        self,
        method_id: str,
        *,
        bert_root: str | Path = DEFAULT_BERT_ROOT,
        device: str = "auto",
        text_reencoder: Callable[[Problem2Split], np.ndarray] | None = None,
        epochs: int = 30,
        patience: int = 5,
        batch_size: int = 32,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
    ) -> None:
        if method_id not in {"concat_mlp", "early_fusion_gru"}:
            raise ValueError(f"unknown direct-fusion method: {method_id}")
        if min(epochs, patience, batch_size, hidden_dim) < 1 or learning_rate <= 0:
            raise ValueError("training arguments must be positive")
        self.method_id = method_id
        self.bert_root = Path(bert_root)
        self.device = device
        self.text_reencoder = text_reencoder
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self._dataset_root: Path | None = None
        self._loaded_checkpoint: Path | None = None
        self._loaded_model = None

    def _device(self):
        import torch

        return torch.device("cuda:0" if self.device == "auto" and torch.cuda.is_available() else
                            "cpu" if self.device == "auto" else self.device)

    def _model(self):
        return ConcatMLP(hidden_dim=self.hidden_dim) if self.method_id == "concat_mlp" else EarlyFusionGRU(hidden_dim=self.hidden_dim)

    def _ensure_encoder(self) -> None:
        if self.text_reencoder is not None:
            return
        if self._dataset_root is None:
            raise ValueError("text_reencoder requires the processed_po source root")
        self.text_reencoder = FrozenBertTextReencoder(
            self.bert_root,
            self._dataset_root / "scaler_params.npz",
            device=str(self._device()),
            batch_size=self.batch_size,
        )

    def reencode_text(self, split: Problem2Split) -> Problem2Split:
        self._ensure_encoder()
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
        refreshed = replace(split, input_ids=tokens)
        encoded = np.asarray(self.text_reencoder(refreshed), dtype=np.float32)
        if encoded.shape != split.values["text"].shape or not np.isfinite(encoded).all():
            raise ValueError("text_reencoder must return finite XT-shaped features")
        encoded[~split.observed["text"]] = 0.0
        return replace(refreshed, values={**split.values, "text": encoded})

    def _prepare(self, split: Problem2Split) -> Problem2Split:
        pooled = pool_unaligned_to_text_slots(split) if split.view == "unaligned_po" else split
        return self.reencode_text(pooled)

    @staticmethod
    def _batch(split: Problem2Split, device):
        import torch

        features = {name: torch.as_tensor(split.values[name], dtype=torch.float32, device=device) for name in MODALITIES}
        masks = {name: torch.as_tensor(split.observed[name], dtype=torch.bool, device=device) for name in MODALITIES}
        classes = torch.as_tensor(split.classification, dtype=torch.long, device=device)
        intensity = torch.as_tensor(split.regression, dtype=torch.float32, device=device)
        return features, masks, classes, intensity

    @staticmethod
    def _loss(output, classes, intensity):
        import torch.nn.functional as functional

        return functional.cross_entropy(output.polarity_logits, classes) + functional.mse_loss(output.intensity, intensity)

    def _validation_loss(self, model, split: Problem2Split, device) -> float:
        import torch

        total = 0.0
        model.eval()
        with torch.no_grad():
            for start in range(0, split.size, self.batch_size):
                batch = _take(split, np.arange(start, min(start + self.batch_size, split.size)))
                features, masks, classes, intensity = self._batch(batch, device)
                loss = self._loss(model(features, masks), classes, intensity)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite direct-fusion validation loss")
                total += float(loss) * batch.size
        return total / split.size

    def train(self, dataset: Problem2Dataset, *, seed: int, run_dir: Path) -> Path:
        import torch

        self._dataset_root = Path(dataset.root)
        self._loaded_checkpoint = None
        self._loaded_model = None
        self._ensure_encoder()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        device = self._device()
        train, valid = self._prepare(dataset["train"]), self._prepare(dataset["valid"])
        model = self._model().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        checkpoint = Path(run_dir) / "best.pt"
        best_loss, best_epoch, stale = float("inf"), 0, 0
        for epoch in range(1, self.epochs + 1):
            model.train()
            order = np.random.default_rng(seed + epoch).permutation(train.size)
            for start in range(0, train.size, self.batch_size):
                batch = _take(train, order[start:start + self.batch_size])
                features, masks, classes, intensity = self._batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss = self._loss(model(features, masks), classes, intensity)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite direct-fusion training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
                optimizer.step()
            valid_loss = self._validation_loss(model, valid, device)
            if valid_loss < best_loss:
                best_loss, best_epoch, stale = valid_loss, epoch, 0
                torch.save(model.state_dict(), checkpoint)
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if not checkpoint.is_file():
            raise RuntimeError("direct-fusion training did not produce a validation checkpoint")
        (Path(run_dir) / "direct_fusion.json").write_text(
            json.dumps({
                "method_id": self.method_id,
                "view": dataset.view,
                "model": {"hidden_dim": self.hidden_dim},
                "selection": {"split": "valid", "rule": "minimum_cross_entropy_plus_mse", "best_epoch": best_epoch, "best_loss": best_loss},
                "training": {"seed": seed, "epochs_max": self.epochs, "batch_size": self.batch_size, "learning_rate": self.learning_rate},
                "polarity_decision": "native_logits_argmax",
                "text_encoder": {"model_root": str(self.bert_root), "scaler": str(self._dataset_root / "scaler_params.npz")},
            }, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return checkpoint

    def predict(self, split: Problem2Split, checkpoint: Path) -> RawMethodPrediction:
        import torch

        metadata = json.loads(Path(checkpoint).with_name("direct_fusion.json").read_text(encoding="utf-8"))
        if metadata["method_id"] != self.method_id:
            raise ValueError("checkpoint belongs to a different direct-fusion method")
        checkpoint = Path(checkpoint).resolve()
        if self._loaded_checkpoint != checkpoint or self._loaded_model is None:
            model = self._model().to(self._device())
            model.load_state_dict(torch.load(checkpoint, map_location=self._device(), weights_only=True), strict=True)
            self._loaded_model, self._loaded_checkpoint = model.eval(), checkpoint
        raw, labels = [], []
        with torch.no_grad():
            for start in range(0, split.size, self.batch_size):
                batch = _take(split, np.arange(start, min(start + self.batch_size, split.size)))
                features, masks, _, _ = self._batch(batch, self._device())
                output = self._loaded_model(features, masks)
                if not bool(torch.isfinite(output.intensity).all()) or not bool(torch.isfinite(output.polarity_logits).all()):
                    raise FloatingPointError("non-finite direct-fusion prediction")
                raw.extend(output.intensity.cpu().tolist())
                labels.extend(output.polarity_logits.argmax(dim=-1).cpu().tolist())
        return RawMethodPrediction(np.asarray(raw, dtype=np.float64), tuple(Polarity.from_value(label) for label in labels), "native_logits_argmax")


def create_concat_mlp_adapter() -> DirectFusionAdapter:
    return DirectFusionAdapter("concat_mlp")


def create_early_fusion_gru_adapter() -> DirectFusionAdapter:
    return DirectFusionAdapter("early_fusion_gru")


__all__ = ["DirectFusionAdapter", "create_concat_mlp_adapter", "create_early_fusion_gru_adapter"]
