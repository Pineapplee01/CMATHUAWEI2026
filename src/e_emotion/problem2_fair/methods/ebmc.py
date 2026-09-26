"""Processed-po bridge for EBMC's original two-stage sequence model."""

from __future__ import annotations

from dataclasses import replace
import importlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from typing import Callable, Mapping

import numpy as np

from e_emotion.baselines.workspace import bert_root, vendor_root
from e_emotion.contracts import Polarity
from e_emotion.problem2_fair.core import Problem2Split
from e_emotion.problem2_fair.q2 import pool_unaligned_to_text_slots
from e_emotion.problem2_fair.run import RawMethodPrediction
from e_emotion.problem2_fair.views import Problem2Dataset

from .l2 import FrozenBertTextReencoder


DEFAULT_SOURCE_ROOT = vendor_root() / "EBMC"
DEFAULT_BERT_ROOT = bert_root()
_MODALITIES = ("audio", "text", "vision")
_VALID_RATE = 0.5


def native_missing_arrays(
    arrays: Mapping[str, np.ndarray], *, mode: str, seed: int, rng=None
) -> dict[str, np.ndarray]:
    """Apply the native sequence loader's frame deletion on top of observed P/O data."""
    if mode not in {"train", "valid"}:
        raise ValueError(f"unsupported EBMC missing mode: {mode}")
    result = {name: np.array(value, copy=True) for name, value in arrays.items()}
    training_rng = np.random if rng is None else rng
    for row in range(len(result["label"])):
        if mode == "train":
            generator = training_rng
            rates = tuple(
                0.0 if generator.rand() < 0.5 else float(generator.uniform(0, 1))
                for _ in _MODALITIES
            )
        else:
            generator = np.random.RandomState(seed * 1_000_003 + row)
            rates = (_VALID_RATE,) * len(_MODALITIES)
        for name, rate in zip(_MODALITIES, rates):
            positions = np.flatnonzero(result[f"{name}_mask"][row])
            keep = generator.uniform(size=len(positions)) > rate
            if name == "text" and len(positions):
                keep[0] = keep[-1] = True
            result[name][row, positions[~keep]] = 0.0
    return result


def split_stage2_state(state: Mapping[str, object]) -> tuple[dict, dict]:
    """Separate the student from the lazy teacher embedded in native best.pth."""
    prefix = "teacher_model."
    student = {key: value for key, value in state.items() if not key.startswith(prefix)}
    teacher = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not student or not teacher:
        raise ValueError("Stage-II checkpoint requires both student and teacher weights")
    if set(student) != set(teacher):
        raise ValueError("Stage-II student and teacher weight keys differ")
    return student, teacher


def _weighted_nonzero_f1(truth: np.ndarray, raw: np.ndarray) -> float:
    selected = truth != 0
    if not np.any(selected):
        raise ValueError("valid split has no nonzero labels for native EBMC F1 selection")
    actual = truth[selected] > 0
    predicted = raw[selected] > 0
    total = len(actual)
    score = 0.0
    for label in (False, True):
        support = int(np.count_nonzero(actual == label))
        predicted_count = int(np.count_nonzero(predicted == label))
        hits = int(np.count_nonzero((actual == label) & (predicted == label)))
        denominator = support + predicted_count
        score += support / total * (2 * hits / denominator if denominator else 0.0)
    return score


class EBMCAdapter:
    """Run native EBMC on P/O-aware 50-slot processed_po features."""

    method_id = "ebmc"
    input_layout = "unaligned_windowed"
    requires_text_reencoding = True

    def __init__(
        self,
        *,
        source_root: str | Path = DEFAULT_SOURCE_ROOT,
        bert_root: str | Path = DEFAULT_BERT_ROOT,
        device: str = "auto",
        text_reencoder: Callable[[Problem2Split], np.ndarray] | None = None,
        batch_size: int = 16,
        epochs: int = 100,
        stage_epoch: int = 50,
        hidden: int = 256,
        depth: int = 4,
        num_heads: int = 2,
    ) -> None:
        if batch_size < 2 or not 0 < stage_epoch < epochs:
            raise ValueError("EBMC needs batch_size >= 2 and 0 < stage_epoch < epochs")
        self.source_root = Path(source_root)
        self.bert_root = Path(bert_root)
        self.device = device
        self.text_reencoder = text_reencoder
        self.batch_size = batch_size
        self.epochs = epochs
        self.stage_epoch = stage_epoch
        self.hidden = hidden
        self.depth = depth
        self.num_heads = num_heads

    def _device(self):
        import torch

        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if self.device == "auto" else torch.device(self.device)

    def _native_model_class(self):
        source = self.source_root.resolve()
        native = source / "EBMC"
        if not (native / "ebmc.py").is_file() or not (source / "config.py").is_file():
            raise FileNotFoundError(f"EBMC native source is incomplete: {source}")
        existing = sys.modules.get("config")
        if existing is not None and Path(getattr(existing, "__file__", "")).resolve() != source / "config.py":
            raise RuntimeError("another native config module is already imported")
        for path in (str(native), str(source)):
            if path not in sys.path:
                sys.path.insert(0, path)
        sys.dont_write_bytecode = True
        return importlib.import_module("ebmc").EBMC

    def _ensure_reencoder(self, dataset_root: Path, device) -> None:
        if self.text_reencoder is None:
            self.text_reencoder = FrozenBertTextReencoder(
                self.bert_root, dataset_root / "scaler_params.npz", device=str(device), batch_size=32
            )

    def reencode_text(self, split: Problem2Split) -> Problem2Split:
        if self.text_reencoder is None:
            raise ValueError("EBMC fair adapter requires text_reencoder")
        tokens = np.array(split.input_ids, copy=True)
        tokens[~split.observed["text"]] = 0
        for row, support in enumerate(split.physical_support["text"]):
            positions = np.flatnonzero(support)
            if not len(positions):
                continue
            first, last = int(positions[0]), int(positions[-1])
            if first > 0:
                tokens[row, first - 1] = 101
            if last + 1 < 50:
                tokens[row, last + 1] = 102
        sanitized = replace(split, input_ids=tokens)
        text = np.asarray(self.text_reencoder(sanitized), dtype=np.float32)
        if text.shape != split.values["text"].shape or not np.isfinite(text).all():
            raise ValueError("text_reencoder must return finite XT-shaped features")
        values = {name: np.array(array, copy=True) for name, array in split.values.items()}
        values["text"] = text.copy()
        for name in values:
            values[name][~split.observed[name]] = 0.0
        return replace(sanitized, values=values)

    def prepare_arrays(self, split: Problem2Split) -> dict[str, np.ndarray]:
        prepared = pool_unaligned_to_text_slots(split) if split.view == "unaligned_po" else split
        arrays: dict[str, np.ndarray] = {}
        for name in _MODALITIES:
            values = np.asarray(prepared.values[name], dtype=np.float32).copy()
            if values.shape[1] != 50:
                raise ValueError("EBMC needs shared 50-slot input")
            values[~prepared.observed[name]] = 0.0
            arrays[name] = values
            arrays[f"{name}_mask"] = prepared.physical_support[name].astype(np.float32, copy=True)
        arrays["label"] = prepared.regression.astype(np.float32, copy=True).reshape(-1, 1)
        return arrays

    @staticmethod
    def _torch_batch(arrays: Mapping[str, np.ndarray], indices: np.ndarray, device):
        import torch

        feature = torch.cat(
            [torch.as_tensor(arrays[name][indices], device=device).transpose(0, 1) for name in _MODALITIES], dim=2
        )
        mask = torch.cat(
            [torch.as_tensor(arrays[f"{name}_mask"][indices], device=device).transpose(0, 1).unsqueeze(-1)
             for name in _MODALITIES], dim=2
        )
        umask = torch.as_tensor(
            np.logical_or.reduce([arrays[f"{name}_mask"][indices] > 0 for name in _MODALITIES]),
            dtype=torch.float32, device=device,
        )
        label = torch.as_tensor(arrays["label"][indices], device=device)
        return feature, mask, umask, label

    def _args(self, dimensions: tuple[int, int, int], device):
        return SimpleNamespace(
            dataset="CMUMOSEI", test_condition="atv", frame_seq=True, device=device, no_cuda=device.type == "cpu",
            n_classes=1, n_speakers=2, hidden=self.hidden, depth=self.depth, num_heads=self.num_heads,
            drop_rate=0.5, attn_drop_rate=0.0, lambda_msd=0.5, lambda_cce=0.1,
            lambda_emc=0.1, lambda_imtd=0.1, adim=dimensions[0], tdim=dimensions[1], vdim=dimensions[2],
        )

    def _new_model(self, args):
        return self._native_model_class()(
            args, args.adim, args.tdim, args.vdim, args.hidden, n_classes=1,
            depth=args.depth, num_heads=args.num_heads, mlp_ratio=1,
            drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate,
            no_cuda=args.no_cuda,
        ).to(args.device)

    def raw_predict(
        self, model, split: Problem2Split, *, batch_size: int | None = None,
        arrays: Mapping[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        import torch

        arrays = self.prepare_arrays(split) if arrays is None else arrays
        device = self._device()
        model.eval()
        predictions = []
        width = batch_size or self.batch_size
        for start in range(0, split.size, width):
            indices = np.arange(start, min(split.size, start + width))
            features, masks, umask, _ = self._torch_batch(arrays, indices, device)
            pseudo_label = torch.zeros((len(indices), 1), dtype=torch.float32, device=device)
            # Native EMC calls autograd.grad even in eval mode.
            with torch.enable_grad():
                output = model(features, masks, umask, False, pseudo_label, 0)[1]
            values = output.reshape(-1).detach()
            if values.numel() != len(indices) or not bool(torch.isfinite(values).all()):
                raise FloatingPointError("EBMC produced non-finite or incomplete predictions")
            predictions.extend(values.cpu().tolist())
        return np.asarray(predictions, dtype=np.float64)

    def _train_epoch(self, model, optimizer, arrays, *, first_stage: bool, device):
        import torch

        model.train()
        order = np.random.permutation(len(arrays["label"]))
        for start in range(0, len(order), self.batch_size):
            indices = order[start:start + self.batch_size]
            if len(indices) < 2:
                continue  # Native teacher variance is undefined for a singleton.
            sampled = native_missing_arrays(
                {name: value[indices] for name, value in arrays.items()}, mode="train", seed=0
            )
            features, masks, umask, label = self._torch_batch(sampled, np.arange(len(indices)), device)
            optimizer.zero_grad()
            output = model(features, masks, umask, first_stage, label, 0)
            _, logits_c, logits_a, logits_t, logits_v, losses, _ = output
            reg = lambda value: ((value.reshape(-1, 1) - label) ** 2).mean()
            loss = (
                reg(logits_a) + reg(logits_t) + reg(logits_v)
                if first_stage else reg(logits_c)
            )
            loss = loss + 0.5 * losses["disentangle"] + 0.1 * losses["cfd"]
            if not first_stage:
                loss = loss + 0.1 * losses["emc"] + 0.1 * losses["imtd"]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("EBMC native training loss is non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    def train(self, dataset: Problem2Dataset, *, seed: int, run_dir: Path) -> Path:
        import torch

        device = self._device()
        self._ensure_reencoder(dataset.root, device)
        train = self.reencode_text(dataset["train"])
        valid = self.reencode_text(dataset["valid"])
        if train.size < 2:
            raise ValueError("EBMC native Stage-II requires at least two training samples")
        train_arrays = self.prepare_arrays(train)
        valid_arrays = native_missing_arrays(self.prepare_arrays(valid), mode="valid", seed=seed)
        dimensions = tuple(train_arrays[name].shape[-1] for name in _MODALITIES)
        args = self._args(dimensions, device)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = self._new_model(args)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        run_dir = Path(run_dir).resolve()
        stage1 = run_dir / "stage1_teacher.pth"
        best = run_dir / "best.pth"
        best_f1 = -1.0
        selected_epoch = None
        for epoch in range(self.epochs):
            first_stage = epoch < self.stage_epoch
            if epoch == self.stage_epoch:
                teacher = self._new_model(args)
                teacher.load_state_dict(torch.load(stage1, map_location=device, weights_only=True), strict=True)
                teacher.requires_grad_(False)
                teacher.eval()
                model.teacher_model = teacher
            self._train_epoch(model, optimizer, train_arrays, first_stage=first_stage, device=device)
            if epoch == self.stage_epoch - 1:
                torch.save(model.state_dict(), stage1)
            if not first_stage:
                raw = self.raw_predict(model, valid, arrays=valid_arrays)
                score = _weighted_nonzero_f1(valid.regression, raw)
                if score > best_f1:
                    best_f1, selected_epoch = score, epoch
                    torch.save(model.state_dict(), best)
        if selected_epoch is None:
            raise RuntimeError("EBMC Stage-II produced no validation-selected checkpoint")
        metadata = {
            "method_id": self.method_id, "view": dataset.view, "seed": seed,
            "dataset_root": str(dataset.root), "source_root": str(self.source_root.resolve()),
            "stage1_checkpoint": str(stage1), "selected_epoch": selected_epoch,
            "selection": {"split": "valid", "missing_rate": _VALID_RATE,
                          "key": "nonzero_weighted_f1", "score": best_f1, "rule": "native >"},
            "dimensions": dimensions,
            "architecture": {"hidden": self.hidden, "depth": self.depth, "num_heads": self.num_heads},
        }
        (run_dir / "ebmc.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return best

    def predict(self, split: Problem2Split, checkpoint: Path) -> RawMethodPrediction:
        import torch

        checkpoint = Path(checkpoint).resolve()
        metadata = json.loads((checkpoint.parent / "ebmc.json").read_text(encoding="utf-8"))
        if metadata["method_id"] != self.method_id or metadata["view"] not in {
            split.view, "unaligned_po" if split.view == "unaligned_windowed" else split.view
        }:
            raise ValueError("EBMC checkpoint method or view differs from requested split")
        if Path(metadata["source_root"]).resolve() != self.source_root.resolve():
            raise ValueError("EBMC checkpoint native source differs from adapter")
        device = self._device()
        self._ensure_reencoder(Path(metadata["dataset_root"]), device)
        dimensions = tuple(metadata["dimensions"])
        architecture = metadata["architecture"]
        self.hidden, self.depth, self.num_heads = (
            architecture["hidden"], architecture["depth"], architecture["num_heads"]
        )
        args = self._args(dimensions, device)
        student = self._new_model(args)
        teacher = self._new_model(args)
        student_state, teacher_state = split_stage2_state(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
        student.load_state_dict(student_state, strict=True)
        teacher.load_state_dict(teacher_state, strict=True)
        teacher.requires_grad_(False)
        teacher.eval()
        student.teacher_model = teacher
        raw = self.raw_predict(student, split)
        classes = np.where(raw < 0, 0, np.where(raw > 0, 2, 1))
        return RawMethodPrediction(
            raw_intensity=raw,
            polarity=tuple(Polarity.from_value(int(value)) for value in classes),
            decision_source="native EBMC regression sign (valid-selected nonzero weighted F1 checkpoint)",
        )


def create_ebmc_adapter(**kwargs) -> EBMCAdapter:
    return EBMCAdapter(**kwargs)


__all__ = ["EBMCAdapter", "create_ebmc_adapter", "native_missing_arrays", "split_stage2_state"]
