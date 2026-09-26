"""Problem 2 bridge for the P-RMF proxy-modality network."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import random
import sys
from typing import Callable

import numpy as np

from e_emotion.baselines.workspace import PROJECT_ROOT, bert_root, vendor_root
from e_emotion.contracts import Polarity
from e_emotion.problem2_fair.core import MODALITIES, Problem2Split
from e_emotion.problem2_fair.methods.l2 import DEFAULT_BERT_ROOT, FrozenBertTextReencoder, _valid_interval
from e_emotion.problem2_fair.q2 import pool_unaligned_to_text_slots
from e_emotion.problem2_fair.run import RawMethodPrediction
from e_emotion.problem2_fair.views import Problem2Dataset


DEFAULT_PRMF_ROOT = vendor_root() / "P-RMF"


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


class PRMFAdapter:
    """Use the P-RMF model and loss with public data, masks and outputs."""

    method_id = "p_rmf"
    input_layout = "unaligned_windowed"

    def __init__(
        self,
        *,
        source_root: str | Path = DEFAULT_PRMF_ROOT,
        bert_root: str | Path = DEFAULT_BERT_ROOT,
        device: str = "auto",
        text_reencoder: Callable[[Problem2Split], np.ndarray] | None = None,
        epochs: int = 60,
        patience: int = 8,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
    ) -> None:
        if min(epochs, patience, batch_size) < 1 or learning_rate <= 0:
            raise ValueError("invalid P-RMF training configuration")
        self.source_root = Path(source_root)
        self.bert_root = Path(bert_root)
        self.device = device
        self.text_reencoder = text_reencoder
        self.epochs, self.patience, self.batch_size, self.learning_rate = epochs, patience, batch_size, learning_rate
        self._dataset_root: Path | None = None
        self._loaded_checkpoint: Path | None = None
        self._loaded_model = None

    def _device(self):
        import torch

        return torch.device("cuda:0" if self.device == "auto" and torch.cuda.is_available() else
                            "cpu" if self.device == "auto" else self.device)

    def _native(self):
        if not (self.source_root / "models" / "P_RMF.py").is_file():
            raise FileNotFoundError(self.source_root / "models" / "P_RMF.py")
        root = str(self.source_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        import yaml
        from core.losses import MultimodalLoss
        from models.P_RMF import build_model

        config = yaml.safe_load((PROJECT_ROOT / "configs" / "baselines" / "p_rmf.yaml").read_text(encoding="utf-8"))
        extractor = config["model"]["feature_extractor"]
        extractor["input_length"] = [50, 50, 50]
        extractor["input_dims"] = [768, 35, 74]
        extractor["bert_pretrained"] = str(self.bert_root)
        return config, build_model, MultimodalLoss

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
        return self.reencode_text(pool_unaligned_to_text_slots(split) if split.view == "unaligned_po" else split)

    @staticmethod
    def _inputs(split: Problem2Split, device):
        import torch

        values = {name: torch.as_tensor(split.values[name], dtype=torch.float32, device=device) for name in MODALITIES}
        masks = {name: torch.as_tensor(split.observed[name], dtype=torch.bool, device=device) for name in MODALITIES}
        labels = torch.as_tensor(split.regression, dtype=torch.float32, device=device).reshape(-1, 1)
        return values, masks, labels

    @staticmethod
    def _forward(model, values, masks, *, complete: bool):
        full = (values["vision"], values["audio"], values["text"])
        mask_tuple = (masks["vision"], masks["audio"], masks["text"])
        return model(full if complete else (None, None, None), full, complete_masks=mask_tuple if complete else None, observed_masks=mask_tuple)

    def _predict_raw(self, model, split: Problem2Split, device) -> np.ndarray:
        import torch

        result: list[float] = []
        model.eval()
        with torch.no_grad():
            for start in range(0, split.size, self.batch_size):
                batch = _take(split, np.arange(start, min(start + self.batch_size, split.size)))
                values, masks, _ = self._inputs(batch, device)
                output = self._forward(model, values, masks, complete=False)["sentiment_preds"].reshape(-1)
                if not bool(torch.isfinite(output).all()):
                    raise FloatingPointError("non-finite P-RMF prediction")
                result.extend(output.cpu().tolist())
        return np.asarray(result, dtype=np.float64)

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
        config, build_model, loss_type = self._native()
        device = self._device()
        train, valid = self._prepare(dataset["train"]), self._prepare(dataset["valid"])
        model = build_model(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=float(config["base"].get("weight_decay", 0.0)))
        objective = loss_type(config)
        checkpoint = Path(run_dir) / "best.pt"
        best, best_epoch, stale = float("inf"), 0, 0
        for epoch in range(1, self.epochs + 1):
            model.train()
            order = np.random.default_rng(seed + epoch).permutation(train.size)
            for start in range(0, train.size, self.batch_size):
                batch = _take(train, order[start:start + self.batch_size])
                values, masks, labels = self._inputs(batch, device)
                optimizer.zero_grad(set_to_none=True)
                output = self._forward(model, values, masks, complete=True)
                loss = objective(output, {"sentiment_labels": labels})["loss"]
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite P-RMF training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
                optimizer.step()
            raw_valid = self._predict_raw(model, valid, device)
            valid_mae = float(np.abs(raw_valid - valid.regression).mean())
            if valid_mae < best:
                best, best_epoch, stale = valid_mae, epoch, 0
                torch.save(model.state_dict(), checkpoint)
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if not checkpoint.is_file():
            raise RuntimeError("P-RMF training did not produce a validation checkpoint")
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
        interval = _valid_interval(self._predict_raw(model, valid, device), valid.classification)
        (Path(run_dir) / "p_rmf.json").write_text(
            json.dumps({
                "method_id": self.method_id,
                "view": dataset.view,
                "source_root": str(self.source_root),
                "config": config,
                "training": {"seed": seed, "epochs_max": self.epochs, "batch_size": self.batch_size, "learning_rate": self.learning_rate},
                "selection": {"split": "valid", "rule": "minimum_mae", "best_epoch": best_epoch, "best_mae": best},
                "polarity_interval": interval,
            }, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return checkpoint

    def predict(self, split: Problem2Split, checkpoint: Path) -> RawMethodPrediction:
        import torch

        checkpoint = Path(checkpoint).resolve()
        metadata = json.loads(checkpoint.with_name("p_rmf.json").read_text(encoding="utf-8"))
        if metadata["method_id"] != self.method_id:
            raise ValueError("checkpoint belongs to a different P-RMF method")
        if self._loaded_checkpoint != checkpoint or self._loaded_model is None:
            _, build_model, _ = self._native()
            model = build_model(metadata["config"]).to(self._device())
            model.load_state_dict(torch.load(checkpoint, map_location=self._device(), weights_only=True), strict=True)
            self._loaded_checkpoint, self._loaded_model = checkpoint, model.eval()
        raw = self._predict_raw(self._loaded_model, split, self._device())
        interval = metadata["polarity_interval"]
        classes = np.where(raw < interval["lower"], 0, np.where(raw > interval["upper"], 2, 1))
        return RawMethodPrediction(raw, tuple(Polarity.from_value(value) for value in classes),
                                   f"valid macro_f1 interval [{interval['lower']:g}, {interval['upper']:g}]")


def create_adapter() -> PRMFAdapter:
    return PRMFAdapter()


__all__ = ["DEFAULT_PRMF_ROOT", "PRMFAdapter", "create_adapter"]
