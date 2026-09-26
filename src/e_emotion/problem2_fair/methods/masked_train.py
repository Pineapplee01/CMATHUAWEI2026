"""Train the project-owned EarlyFusionGRU with Q2-v2 local-missingness augmentation."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import random
from typing import Callable

import numpy as np

from e_emotion.baselines.direct_fusion import EarlyFusionGRU
from e_emotion.contracts import Polarity
from e_emotion.problem2_fair.core import MODALITIES, Problem2Split, Q2_V2_CONDITIONS
from e_emotion.problem2_fair.methods.l2 import DEFAULT_BERT_ROOT, FrozenBertTextReencoder
from e_emotion.problem2_fair.q2 import materialize_q2_condition, pool_unaligned_to_text_slots
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


class MaskedTrainAdapter:
    """Q2-augmented training control with an unchanged EarlyFusionGRU network."""

    method_id = "masked_train"
    input_layout = "unaligned_windowed"
    requires_text_reencoding = True

    def __init__(
        self,
        *,
        bert_root: str | Path = DEFAULT_BERT_ROOT,
        device: str = "auto",
        text_reencoder: Callable[[Problem2Split], np.ndarray] | None = None,
        epochs: int = 30,
        patience: int = 5,
        batch_size: int = 32,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        cross_entropy_weight: float = 1.0,
        mse_weight: float = 1.0,
    ) -> None:
        if epochs < 1 or patience < 1 or batch_size < 1 or hidden_dim < 1:
            raise ValueError("epochs, patience, batch_size and hidden_dim must be positive")
        if learning_rate <= 0 or cross_entropy_weight < 0 or mse_weight < 0:
            raise ValueError("learning_rate must be positive and loss weights nonnegative")
        if cross_entropy_weight + mse_weight == 0:
            raise ValueError("at least one loss weight must be positive")
        self.bert_root = Path(bert_root)
        self.device = device
        self.text_reencoder = text_reencoder
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.cross_entropy_weight = cross_entropy_weight
        self.mse_weight = mse_weight
        self._dataset_root: Path | None = None
        self._loaded_checkpoint: Path | None = None
        self._loaded_model: EarlyFusionGRU | None = None

    def _device(self):
        import torch

        if self.device == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _build_model(self) -> EarlyFusionGRU:
        return EarlyFusionGRU(hidden_dim=self.hidden_dim)

    def _ensure_encoder(self) -> None:
        if self.text_reencoder is not None:
            return
        if self._dataset_root is None:
            raise ValueError("text_reencoder requires a training dataset with a view scaler")
        self.text_reencoder = FrozenBertTextReencoder(
            self.bert_root,
            self._dataset_root / "scaler_params.npz",
            device=str(self._device()),
            batch_size=self.batch_size,
        )

    def reencode_text(self, split: Problem2Split) -> Problem2Split:
        """Encode surviving tokens with the frozen BERT and current view scaler."""
        self._ensure_encoder()
        tokens = np.array(split.input_ids, copy=True)
        tokens[~split.observed["text"]] = 0
        for row, support in enumerate(split.physical_support["text"]):
            positions = np.flatnonzero(support)
            if len(positions) == 0:
                continue
            first, last = int(positions[0]), int(positions[-1])
            if first > 0:
                tokens[row, first - 1] = 101
            if last + 1 < tokens.shape[1]:
                tokens[row, last + 1] = 102
        sanitized = replace(split, input_ids=tokens)
        encoded = np.asarray(self.text_reencoder(sanitized), dtype=np.float32)
        if encoded.shape != split.values["text"].shape or not np.isfinite(encoded).all():
            raise ValueError("text_reencoder must return finite XT-shaped features")
        text = encoded.copy()
        text[~split.observed["text"]] = 0
        return replace(sanitized, values={**split.values, "text": text})

    def augment_training_split(
        self,
        split: Problem2Split,
        *,
        rng: np.random.Generator,
        condition: tuple[str, str | None, float] | None = None,
    ) -> Problem2Split:
        """Apply one Q2-v2 condition before any unaligned physical-window pooling."""
        if split.split != "train":
            raise ValueError("training augmentation may only consume the train split")
        if condition is None:
            condition = Q2_V2_CONDITIONS[int(rng.integers(len(Q2_V2_CONDITIONS)))]
        if condition not in Q2_V2_CONDITIONS:
            raise ValueError("training augmentation requires a canonical Q2-v2 condition")
        masked = materialize_q2_condition(split, *condition)
        prepared = pool_unaligned_to_text_slots(masked) if masked.view == "unaligned_po" else masked
        return self.reencode_text(prepared)

    def _prepare_clean(self, split: Problem2Split) -> Problem2Split:
        prepared = pool_unaligned_to_text_slots(split) if split.view == "unaligned_po" else split
        return self.reencode_text(prepared)

    @staticmethod
    def _batch(split: Problem2Split, device):
        import torch

        features = {
            name: torch.as_tensor(split.values[name], dtype=torch.float32, device=device)
            for name in MODALITIES
        }
        masks = {
            name: torch.as_tensor(split.observed[name], dtype=torch.bool, device=device)
            for name in MODALITIES
        }
        classes = torch.as_tensor(split.classification, dtype=torch.long, device=device)
        intensity = torch.as_tensor(split.regression, dtype=torch.float32, device=device)
        return features, masks, classes, intensity

    def _loss(self, output, classes, intensity):
        import torch.nn.functional as F

        return (
            self.cross_entropy_weight * F.cross_entropy(output.polarity_logits, classes)
            + self.mse_weight * F.mse_loss(output.intensity, intensity)
        )

    def _validation_loss(self, model, split: Problem2Split, device) -> float:
        import torch

        model.eval()
        weighted_sum = 0.0
        with torch.no_grad():
            for start in range(0, split.size, self.batch_size):
                indices = np.arange(start, min(start + self.batch_size, split.size))
                features, masks, classes, intensity = self._batch(_take(split, indices), device)
                loss = self._loss(model(features, masks), classes, intensity)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite Masked-Train validation loss")
                weighted_sum += float(loss) * len(indices)
        return weighted_sum / split.size

    def train(self, dataset: Problem2Dataset, *, seed: int, run_dir: Path) -> Path:
        import torch

        if dataset.view not in {"aligned_po", "unaligned_po"}:
            raise ValueError("Masked-Train needs an aligned_po or unaligned_po dataset")
        if dataset["train"].size == 0 or dataset["valid"].size == 0:
            raise ValueError("train and valid splits must be nonempty")
        self._dataset_root = Path(dataset.root)
        self._loaded_checkpoint = None
        self._loaded_model = None
        self._ensure_encoder()
        device = self._device()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        model = self._build_model().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        valid = self._prepare_clean(dataset["valid"])
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = run_dir / "best.pt"
        best_loss = float("inf")
        best_epoch = 0
        stale_epochs = 0
        for epoch in range(1, self.epochs + 1):
            model.train()
            order = rng.permutation(dataset["train"].size)
            for start in range(0, len(order), self.batch_size):
                batch = _take(dataset["train"], order[start:start + self.batch_size])
                augmented = self.augment_training_split(batch, rng=rng)
                features, masks, classes, intensity = self._batch(augmented, device)
                optimizer.zero_grad(set_to_none=True)
                loss = self._loss(model(features, masks), classes, intensity)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite Masked-Train training loss")
                loss.backward()
                optimizer.step()
            valid_loss = self._validation_loss(model, valid, device)
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                stale_epochs = 0
                torch.save(model.state_dict(), checkpoint)
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break
        metadata = {
            "method_id": self.method_id,
            "model_class": "EarlyFusionGRU",
            "training_choices": "method_defined_control_settings_not_q2_protocol_requirements",
            "seed": seed,
            "view": dataset.view,
            "model": {"hidden_dim": self.hidden_dim},
            "training": {
                "epochs_max": self.epochs,
                "epochs_run": epoch,
                "patience": self.patience,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
            },
            "augmentation": {
                "protocol": "q2-continuous-local-v2",
                "condition_count": len(Q2_V2_CONDITIONS),
                "sampling": "uniform_one_condition_per_train_batch",
                "rng_seed": seed,
                "applied_before_unaligned_pooling": True,
            },
            "loss_weights": {"cross_entropy": self.cross_entropy_weight, "mse": self.mse_weight},
            "selection": {
                "split": "valid",
                "rule": "minimum_weighted_cross_entropy_plus_mse",
                "best_epoch": best_epoch,
                "best_loss": best_loss,
            },
            "polarity_decision": "native_logits_argmax",
            "text_encoder": {"model_root": str(self.bert_root), "scaler": str(self._dataset_root / "scaler_params.npz")},
        }
        (run_dir / "masked_train.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return checkpoint

    def predict(self, split: Problem2Split, checkpoint: Path) -> RawMethodPrediction:
        import torch

        checkpoint = Path(checkpoint).resolve()
        metadata = json.loads(checkpoint.with_name("masked_train.json").read_text(encoding="utf-8"))
        if metadata["method_id"] != self.method_id:
            raise ValueError("checkpoint belongs to a different method")
        if metadata["view"] != split.view and not (metadata["view"] == "unaligned_po" and split.view == "unaligned_windowed"):
            raise ValueError("checkpoint and split views differ")
        device = self._device()
        if self._loaded_checkpoint != checkpoint or self._loaded_model is None:
            model = EarlyFusionGRU(hidden_dim=int(metadata["model"]["hidden_dim"])).to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
            self._loaded_model = model.eval()
            self._loaded_checkpoint = checkpoint
        model = self._loaded_model
        raw: list[float] = []
        classes: list[int] = []
        with torch.no_grad():
            for start in range(0, split.size, self.batch_size):
                indices = np.arange(start, min(start + self.batch_size, split.size))
                features, masks, _, _ = self._batch(_take(split, indices), device)
                output = model(features, masks)
                if not bool(torch.isfinite(output.intensity).all()) or not bool(torch.isfinite(output.polarity_logits).all()):
                    raise FloatingPointError("non-finite Masked-Train prediction")
                raw.extend(output.intensity.cpu().tolist())
                classes.extend(output.polarity_logits.argmax(-1).cpu().tolist())
        return RawMethodPrediction(
            raw_intensity=np.asarray(raw, dtype=np.float64),
            polarity=tuple(Polarity.from_value(value) for value in classes),
            decision_source="native_logits_argmax",
        )


def create_adapter() -> MaskedTrainAdapter:
    return MaskedTrainAdapter()


__all__ = ["MaskedTrainAdapter", "create_adapter"]
