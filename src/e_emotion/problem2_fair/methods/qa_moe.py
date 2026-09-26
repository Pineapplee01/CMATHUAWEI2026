"""QA-MoE bridge using processed_po tokens and the original model/trainer."""

from __future__ import annotations

from dataclasses import replace
import importlib
import json
from pathlib import Path
import random
import sys
from unittest.mock import patch

import numpy as np

from e_emotion.baselines.workspace import PROJECT_ROOT, bert_root, vendor_root
from e_emotion.contracts import Polarity
from e_emotion.problem2_fair.core import Problem2Split
from e_emotion.problem2_fair.q2 import pool_unaligned_to_text_slots
from e_emotion.problem2_fair.run import RawMethodPrediction
from e_emotion.problem2_fair.views import Problem2Dataset

from .l2 import NativeArgs, _valid_interval


DEFAULT_SOURCE_ROOT = vendor_root() / "QA-MoE"
DEFAULT_BERT_ROOT = bert_root()
NATIVE_DROP_RATE = 0.1
NATIVE_VALID_SEED = 1111
NATIVE_EARLY_STOP_DELTA = 1e-4


def _validation_modes(size: int, *, drop_rate: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return np.asarray([
        6 if rng.rand() >= drop_rate else int(rng.randint(0, 6))
        for _ in range(size)
    ], dtype=np.int64)


def _mae_improved(current: float, best: float) -> bool:
    return current < best - NATIVE_EARLY_STOP_DELTA


def _augment_sample(
    sample: dict[str, np.ndarray],
    observed: dict[str, np.ndarray],
    *,
    noise_level: float,
    missing_mode: int,
    rng,
) -> dict[str, np.ndarray]:
    augmented = {name: np.array(value, copy=True) for name, value in sample.items()}
    if noise_level > 0:
        for name in ("audio", "vision"):
            noise = np.asarray(rng.normal(0.0, noise_level, size=augmented[name].shape), dtype=np.float32)
            augmented[name] += noise * observed[name][:, None]
        keep = rng.rand(1, augmented["text"].shape[1]) > noise_level
        augmented["text"][0] *= keep[0]
    dropped = {
        0: ("vision",), 1: ("text",), 2: ("audio",),
        3: ("text", "vision"), 4: ("audio", "vision"),
        5: ("text", "audio"), 6: (),
    }[missing_mode]
    for name in ("text", "audio", "vision"):
        if name in dropped:
            augmented[name].fill(0)
        augmented[f"missing_{name}"] = np.bool_(not observed[name].any() or name in dropped)
    return augmented


def _tokens(split: Problem2Split) -> np.ndarray:
    tokens = np.array(split.input_ids, dtype=np.int64, copy=True)
    tokens[~split.observed["text"]] = 0
    for row, support in enumerate(split.physical_support["text"]):
        if not split.observed["text"][row].any():
            continue
        positions = np.flatnonzero(support)
        if not len(positions):
            continue
        first, last = int(positions[0]), int(positions[-1])
        if first > 0:
            tokens[row, first - 1] = 101
        if last + 1 < tokens.shape[1]:
            tokens[row, last + 1] = 102
    return tokens


class QAMoEFairAdapter:
    """Retain QA-MoE's native BERT, optimizer, loss and valid-MAE selection."""

    method_id = "qa_moe"
    input_layout = "unaligned_windowed"
    requires_text_reencoding = True

    def __init__(
        self,
        *,
        source_root: str | Path = DEFAULT_SOURCE_ROOT,
        bert_root: str | Path = DEFAULT_BERT_ROOT,
        device: str = "auto",
        batch_size: int | None = None,
    ) -> None:
        self.source_root = Path(source_root)
        self.bert_root = Path(bert_root)
        self.device = device
        self.batch_size = batch_size
        self._loaded_checkpoint: Path | None = None
        self._model = None

    def reencode_text(self, split: Problem2Split) -> Problem2Split:
        """Pass only surviving token semantics; native BERT encodes online."""
        tokens = _tokens(split)
        values = {name: np.array(value, copy=True) for name, value in split.values.items()}
        values["text"].fill(0)
        for name in ("audio", "vision"):
            values[name][~split.observed[name]] = 0
        return replace(split, input_ids=tokens, values=values)

    def prepare_arrays(self, split: Problem2Split) -> dict[str, np.ndarray]:
        """Build the native model's input fields without reading legacy data."""
        prepared = pool_unaligned_to_text_slots(split) if split.view == "unaligned_po" else split
        if any(prepared.values[name].shape[1] != 50 for name in ("audio", "vision")):
            raise ValueError("QA-MoE requires 50 text-anchored slots")
        tokens = _tokens(prepared)
        attention = (tokens != 0).astype(np.int64)
        text = np.stack((tokens, attention, np.zeros_like(tokens)), axis=1)
        arrays: dict[str, np.ndarray] = {"text": text}
        for name in ("audio", "vision"):
            values = np.array(prepared.values[name], dtype=np.float32, copy=True)
            values[~prepared.observed[name]] = 0
            arrays[name] = values
        for name in ("text", "audio", "vision"):
            arrays[f"missing_{name}"] = ~prepared.observed[name].any(axis=1)
        arrays["regression_labels"] = np.asarray(prepared.regression, dtype=np.float32)
        return arrays

    def _device(self):
        import torch

        if self.device == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _native(self, module: str):
        source = self.source_root.resolve()
        if not (source / "model" / "model.py").is_file():
            raise FileNotFoundError(source / "model" / "model.py")
        existing = sys.modules.get("model")
        if existing is not None:
            paths = tuple(Path(item).resolve() for item in getattr(existing, "__path__", ()))
            if source / "model" not in paths:
                raise RuntimeError("a different native 'model' source tree is already imported")
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        sys.dont_write_bytecode = True
        return importlib.import_module(module)

    def _config(self, split: Problem2Split, *, data_root: Path) -> NativeArgs:
        import yaml

        path = PROJECT_ROOT / "configs" / "baselines" / "qa_moe.yaml"
        with path.open(encoding="utf-8") as stream:
            config = NativeArgs(yaml.safe_load(stream))
        config.dataset = "MOSEI"
        config.data_path = str(data_root)
        config.audio_input_dim = int(split.values["audio"].shape[-1])
        config.vision_input_dim = int(split.values["vision"].shape[-1])
        config.finetune_bert = True
        if self.batch_size is not None:
            config.batch_size = self.batch_size
        config.device = str(self._device()) if self.device == "auto" else self.device
        return config

    def _new_model(self, config: NativeArgs):
        from transformers import BertModel

        if not self.bert_root.is_dir():
            raise FileNotFoundError(self.bert_root)
        native = self._native("model.model")
        original_loader = BertModel.from_pretrained
        with patch.object(
            BertModel, "from_pretrained",
            side_effect=lambda *_args, **_kwargs: original_loader(
                str(self.bert_root), local_files_only=True
            ),
        ):
            return native.MoEMultimodalEmotionModel(config).to(self._device())

    def _loader(
        self, split: Problem2Split, *, batch_size: int, shuffle: bool,
        augmentation: str | None = None, noise_level: float = 0.0,
    ):
        import torch
        from torch.utils.data import DataLoader, Dataset

        prepared = pool_unaligned_to_text_slots(split) if split.view == "unaligned_po" else split
        arrays = self.prepare_arrays(prepared)
        if augmentation not in (None, "train", "valid"):
            raise ValueError(f"unknown QA-MoE augmentation split: {augmentation}")
        fixed_modes = (
            _validation_modes(prepared.size, drop_rate=NATIVE_DROP_RATE, seed=NATIVE_VALID_SEED)
            if augmentation == "valid" else None
        )

        class SplitDataset(Dataset):
            def __len__(self):
                return split.size

            def __getitem__(self, index):
                sample = {name: values[index] for name, values in arrays.items()}
                if augmentation is not None:
                    mode = (
                        int(fixed_modes[index]) if fixed_modes is not None else
                        6 if random.random() >= NATIVE_DROP_RATE else random.randint(0, 5)
                    )
                    rng = np.random if augmentation == "train" else np.random.RandomState(NATIVE_VALID_SEED + index)
                    sample = _augment_sample(
                        sample, {name: prepared.observed[name][index] for name in ("text", "audio", "vision")},
                        noise_level=noise_level, missing_mode=mode, rng=rng,
                    )
                return {
                    name: torch.as_tensor(value)
                    for name, value in sample.items()
                }

        return DataLoader(SplitDataset(), batch_size=batch_size, shuffle=shuffle, num_workers=0)

    def _raw_predict(
        self, model, split: Problem2Split, *, batch_size: int,
        augmentation: str | None = None, noise_level: float = 0.0,
    ) -> np.ndarray:
        import torch

        model.eval()
        output = []
        with torch.no_grad():
            for batch in self._loader(
                split, batch_size=batch_size, shuffle=False,
                augmentation=augmentation, noise_level=noise_level,
            ):
                prediction = model(*(
                    batch[name].to(self._device())
                    for name in ("text", "audio", "vision", "missing_text", "missing_audio", "missing_vision")
                ))["output"].reshape(-1)
                if not bool(torch.isfinite(prediction).all()):
                    raise FloatingPointError("non-finite QA-MoE prediction")
                output.extend(prediction.cpu().tolist())
        raw = np.asarray(output, dtype=np.float64)
        if raw.shape != (split.size,):
            raise ValueError("QA-MoE prediction count differs from input split")
        return raw

    def train(self, dataset: Problem2Dataset, *, seed: int, run_dir: Path) -> Path:
        import torch

        train = self.reencode_text(
            pool_unaligned_to_text_slots(dataset["train"])
            if dataset.view == "unaligned_po" else dataset["train"]
        )
        valid = self.reencode_text(
            pool_unaligned_to_text_slots(dataset["valid"])
            if dataset.view == "unaligned_po" else dataset["valid"]
        )
        config = self._config(train, data_root=dataset.root)
        config.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = self._new_model(config)
        RegressionMoELoss = self._native("losses.regression_loss").RegressionMoELoss
        EmotionContrastiveLoss = self._native("losses.contrastive_loss").EmotionContrastiveLoss
        train_one_epoch = self._native("utils.trainer").train_one_epoch
        optimizer = torch.optim.AdamW(
            [
                {"params": model.text_encoder.bert.parameters(), "lr": 1e-5},
                {"params": [p for name, p in model.named_parameters() if "bert" not in name], "lr": config.lr},
            ],
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        loss = RegressionMoELoss(
            reg_weight=config.reg_weight,
            quality_weight=config.quality_weight,
            entropy_weight=config.entropy_weight,
        )
        contrastive = EmotionContrastiveLoss() if config.use_contrastive else None
        train_loader = self._loader(
            train, batch_size=config.batch_size, shuffle=True,
            augmentation="train", noise_level=float(config.noise_level),
        )
        checkpoint = Path(run_dir) / "best.pth"
        best, wait = float("inf"), 0
        history = []
        for epoch in range(1, int(config.epochs) + 1):
            for parameter in model.text_encoder.parameters():
                parameter.requires_grad = epoch >= 6
            if config.use_contrastive and epoch - 1 > config.contrastive_freeze_epoch:
                loss.contrastive_weight = 0.1
            summary = train_one_epoch(
                config, epoch, model, train_loader, optimizer, loss,
                contrastive_loss_fn=contrastive, device=config.device,
                use_contrastive=config.use_contrastive,
            )
            valid_raw = self._raw_predict(
                model, valid, batch_size=config.batch_size,
                augmentation="valid", noise_level=float(config.noise_level),
            )
            mae = float(np.mean(np.abs(valid.regression - valid_raw)))
            if not np.isfinite(mae):
                raise FloatingPointError("non-finite QA-MoE validation MAE")
            history.append({"epoch": epoch, "valid_mae": mae, "train": summary})
            scheduler.step(mae)
            if _mae_improved(mae, best):
                best, wait = mae, 0
                torch.save(model.state_dict(), checkpoint)
            else:
                wait += 1
            if wait >= 7:
                break
        if not checkpoint.is_file():
            raise RuntimeError("QA-MoE did not save a valid-selected checkpoint")
        model.load_state_dict(torch.load(checkpoint, map_location=self._device(), weights_only=True), strict=True)
        valid_raw = self._raw_predict(model, valid, batch_size=config.batch_size)
        interval = _valid_interval(valid_raw, np.asarray(valid.classification, dtype=np.int64))
        (Path(run_dir) / "qa_moe.json").write_text(
            json.dumps({
                "method_id": self.method_id, "seed": seed, "view": dataset.view,
                "data_root": str(dataset.root), "source_root": str(self.source_root),
                "bert_root": str(self.bert_root), "config": dict(config),
                "selection": {
                    "split": "valid", "metric": "mae", "rule": "minimum, patience 7",
                    "delta": NATIVE_EARLY_STOP_DELTA,
                    "drop_rate": NATIVE_DROP_RATE, "noise_level": float(config.noise_level),
                    "validation_augmentation_seed": NATIVE_VALID_SEED,
                },
                "history": history, "polarity_interval": interval,
            }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        self._model = model.eval()
        self._loaded_checkpoint = checkpoint.resolve()
        return checkpoint

    def predict(self, split: Problem2Split, checkpoint: Path) -> RawMethodPrediction:
        import torch

        checkpoint = Path(checkpoint).resolve()
        with checkpoint.with_name("qa_moe.json").open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        if metadata["method_id"] != self.method_id:
            raise ValueError("checkpoint method differs from QA-MoE adapter")
        if self._loaded_checkpoint != checkpoint or self._model is None:
            model = self._new_model(NativeArgs(metadata["config"]))
            model.load_state_dict(torch.load(checkpoint, map_location=self._device(), weights_only=True), strict=True)
            self._model = model.eval()
            self._loaded_checkpoint = checkpoint
        raw = self._raw_predict(
            self._model, split,
            batch_size=int(self.batch_size or metadata["config"]["batch_size"]),
        )
        interval = metadata["polarity_interval"]
        classes = np.where(raw < interval["lower"], 0, np.where(raw > interval["upper"], 2, 1))
        return RawMethodPrediction(
            raw_intensity=raw,
            polarity=tuple(Polarity.from_value(int(value)) for value in classes),
            decision_source=f"valid macro_f1 interval [{interval['lower']:g}, {interval['upper']:g}]",
        )


def create_adapter(**kwargs) -> QAMoEFairAdapter:
    return QAMoEFairAdapter(**kwargs)


__all__ = ["QAMoEFairAdapter", "create_adapter"]
