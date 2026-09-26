"""Problem 2 bridge for CMAD's complete-teacher, missing-student training path."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Callable

import numpy as np

from e_emotion.baselines.workspace import PROJECT_ROOT, bert_root, vendor_root
from e_emotion.contracts import Polarity
from e_emotion.problem2_fair.core import MODALITIES, Problem2Split
from e_emotion.problem2_fair.methods.l2 import DEFAULT_BERT_ROOT, FrozenBertTextReencoder, _valid_interval
from e_emotion.problem2_fair.q2 import pool_unaligned_to_text_slots
from e_emotion.problem2_fair.run import RawMethodPrediction
from e_emotion.problem2_fair.views import Problem2Dataset


DEFAULT_CMAD_ROOT = vendor_root() / "CMAD"


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


class CMADAdapter:
    """Use CMAD's native teacher/student objectives with the shared P/O split."""

    method_id = "cmad"
    input_layout = "unaligned_windowed"

    def __init__(
        self,
        *,
        source_root: str | Path = DEFAULT_CMAD_ROOT,
        bert_root: str | Path = DEFAULT_BERT_ROOT,
        device: str = "auto",
        text_reencoder: Callable[[Problem2Split], np.ndarray] | None = None,
        teacher_epochs: int = 30,
        student_epochs: int = 30,
        patience: int = 8,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
    ) -> None:
        if min(teacher_epochs, student_epochs, patience, batch_size) < 1 or learning_rate <= 0:
            raise ValueError("invalid CMAD training configuration")
        self.source_root = Path(source_root)
        self.bert_root = Path(bert_root)
        self.device = device
        self.text_reencoder = text_reencoder
        self.teacher_epochs, self.student_epochs = teacher_epochs, student_epochs
        self.patience, self.batch_size, self.learning_rate = patience, batch_size, learning_rate
        self._dataset_root: Path | None = None
        self._loaded_checkpoint: Path | None = None
        self._loaded_model = None

    def _device(self):
        import torch

        return torch.device("cuda:0" if self.device == "auto" and torch.cuda.is_available() else
                            "cpu" if self.device == "auto" else self.device)

    def _native(self):
        from e_emotion.baselines import cmad_launcher as module

        config = json.loads((PROJECT_ROOT / "configs" / "baselines" / "cmad.json").read_text(encoding="utf-8"))
        config["bert"] = str(self.bert_root)
        return module, config

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
    def _batch(split: Problem2Split, device):
        import torch

        return {
            "text_embeddings": torch.as_tensor(split.values["text"], dtype=torch.float32, device=device),
            "attention_mask": torch.as_tensor(split.input_ids != 0, dtype=torch.long, device=device),
            "vision": torch.as_tensor(split.values["vision"], dtype=torch.float32, device=device),
            "audio": torch.as_tensor(split.values["audio"], dtype=torch.float32, device=device),
            "regression": torch.as_tensor(split.regression, dtype=torch.float32, device=device),
            "mT": torch.as_tensor(split.physical_support["text"], dtype=torch.bool, device=device),
            "mA": torch.as_tensor(split.physical_support["audio"], dtype=torch.bool, device=device),
            "mV": torch.as_tensor(split.physical_support["vision"], dtype=torch.bool, device=device),
            "observed_mT": torch.as_tensor(split.observed["text"], dtype=torch.bool, device=device),
            "observed_mA": torch.as_tensor(split.observed["audio"], dtype=torch.bool, device=device),
            "observed_mV": torch.as_tensor(split.observed["vision"], dtype=torch.bool, device=device),
        }

    def _predict_raw(self, native, model, split: Problem2Split, device) -> np.ndarray:
        import torch

        values: list[float] = []
        model.eval()
        with torch.no_grad():
            for start in range(0, split.size, self.batch_size):
                batch = _take(split, np.arange(start, min(start + self.batch_size, split.size)))
                output, _ = native.forward(model, self._batch(batch, device), device)
                prediction = output[0].reshape(-1)
                if not bool(torch.isfinite(prediction).all()):
                    raise FloatingPointError("non-finite CMAD prediction")
                values.extend(prediction.cpu().tolist())
        return np.asarray(values, dtype=np.float64)

    @staticmethod
    def _optimizer(native, model, args, stage: str):
        return native.optimizer_for(model, args, stage)

    def _fit_teacher(self, native, teacher, args, train: Problem2Split, valid: Problem2Split, device, run_dir: Path):
        import torch
        import torch.nn.functional as functional

        optimizer = self._optimizer(native, teacher, args, "teacher")
        checkpoint, best, best_epoch = run_dir / "teacher_best.pt", float("inf"), 0
        for epoch in range(1, self.teacher_epochs + 1):
            teacher.train()
            order = np.random.default_rng(args.seed + epoch).permutation(train.size)
            for start in range(0, train.size, self.batch_size):
                batch = _take(train, order[start:start + self.batch_size])
                optimizer.zero_grad(set_to_none=True)
                output, payload = native.forward(teacher, self._batch(batch, device), device)
                loss = functional.l1_loss(output[0].reshape(-1), payload["regression"].reshape(-1))
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite CMAD teacher loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), args.clip, error_if_nonfinite=True)
                optimizer.step()
            mae = float(np.abs(self._predict_raw(native, teacher, valid, device) - valid.regression).mean())
            if mae < best:
                best, best_epoch = mae, epoch
                torch.save(teacher.state_dict(), checkpoint)
        if not checkpoint.is_file():
            raise RuntimeError("CMAD teacher did not produce a validation checkpoint")
        teacher.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
        return checkpoint, best_epoch, best

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
        native, config = self._native()
        config["training"]["seed"] = seed
        config["training"]["learning_rate"] = self.learning_rate
        args = SimpleNamespace(**config["training"])
        device = self._device()
        train, valid = self._prepare(dataset["train"]), self._prepare(dataset["valid"])
        teacher, student = native.build_models(device, config)
        teacher_ckpt, teacher_epoch, teacher_mae = self._fit_teacher(native, teacher, args, train, valid, device, Path(run_dir))
        teacher.eval()
        teacher.requires_grad_(False)
        optimizer = self._optimizer(native, student, args, "student")
        state = native.MARState(args, device)
        checkpoint, best, best_epoch, stale = Path(run_dir) / "best.pt", float("inf"), 0, 0
        for epoch in range(1, self.student_epochs + 1):
            student.train()
            state.begin(epoch - 1)
            order = np.random.default_rng(seed + self.teacher_epochs + epoch).permutation(train.size)
            for start in range(0, train.size, self.batch_size):
                batch = _take(train, order[start:start + self.batch_size])
                payload = self._batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    teacher_output, _ = native.forward(teacher, payload, device)
                student_output, student_payload = native.forward(student, payload, device)
                if len(student_output) < 3 or len(teacher_output) < 2:
                    raise RuntimeError("CMAD native outputs do not contain prediction, hidden state and modality condition")
                loss = native.student_objective(
                    student_output[0], student_output[1], teacher_output[0], teacher_output[1],
                    student_output[2], student_payload["regression"], state,
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite CMAD student loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip, error_if_nonfinite=True)
                optimizer.step()
            mae = float(np.abs(self._predict_raw(native, student, valid, device) - valid.regression).mean())
            if mae < best:
                best, best_epoch, stale = mae, epoch, 0
                torch.save(student.state_dict(), checkpoint)
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if not checkpoint.is_file():
            raise RuntimeError("CMAD student did not produce a validation checkpoint")
        student.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
        interval = _valid_interval(self._predict_raw(native, student, valid, device), valid.classification)
        (Path(run_dir) / "cmad.json").write_text(
            json.dumps({
                "method_id": self.method_id,
                "view": dataset.view,
                "source_root": str(self.source_root),
                "config": config,
                "training": {"seed": seed, "teacher_epochs": self.teacher_epochs, "student_epochs": self.student_epochs, "batch_size": self.batch_size, "learning_rate": self.learning_rate},
                "teacher": {"checkpoint": str(teacher_ckpt), "best_epoch": teacher_epoch, "best_mae": teacher_mae},
                "selection": {"split": "valid", "rule": "minimum_mae", "best_epoch": best_epoch, "best_mae": best},
                "polarity_interval": interval,
            }, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return checkpoint

    def predict(self, split: Problem2Split, checkpoint: Path) -> RawMethodPrediction:
        import torch

        checkpoint = Path(checkpoint).resolve()
        metadata = json.loads(checkpoint.with_name("cmad.json").read_text(encoding="utf-8"))
        if metadata["method_id"] != self.method_id:
            raise ValueError("checkpoint belongs to a different CMAD method")
        native, config = self._native()
        if self._loaded_checkpoint != checkpoint or self._loaded_model is None:
            _, student = native.build_models(self._device(), config)
            student.load_state_dict(torch.load(checkpoint, map_location=self._device(), weights_only=True), strict=True)
            self._loaded_checkpoint, self._loaded_model = checkpoint, student.eval()
        raw = self._predict_raw(native, self._loaded_model, split, self._device())
        interval = metadata["polarity_interval"]
        classes = np.where(raw < interval["lower"], 0, np.where(raw > interval["upper"], 2, 1))
        return RawMethodPrediction(raw, tuple(Polarity.from_value(value) for value in classes),
                                   f"valid macro_f1 interval [{interval['lower']:g}, {interval['upper']:g}]")


def create_adapter() -> CMADAdapter:
    return CMADAdapter()


__all__ = ["CMADAdapter", "DEFAULT_CMAD_ROOT", "create_adapter"]
