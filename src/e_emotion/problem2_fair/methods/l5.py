"""Fair Problem 2 bridges for the original EMOE, CACR, TLRA and MRUF trainers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Callable

import numpy as np

from e_emotion.baselines.workspace import PROJECT_ROOT, bert_root, vendor_root
from e_emotion.contracts import Polarity
from e_emotion.problem2_fair.core import Problem2Split
from e_emotion.problem2_fair.q2 import pool_unaligned_to_text_slots
from e_emotion.problem2_fair.run import RawMethodPrediction
from e_emotion.problem2_fair.views import Problem2Dataset

from .l2 import FrozenBertTextReencoder, NativeArgs, _valid_interval as valid_interval


DEFAULT_VENDOR_ROOT = vendor_root()
DEFAULT_BERT_ROOT = bert_root()
_NATIVE = {
    "emoe": ("EMOE", "emoe", "EMOE", "EMOE", "logits_c", "pt/emoe.pth"),
    "cacr": ("CACR", "CACR", "CACR", "CACR", "output_logit", "pt/CACRcmumosei.pth"),
    "tlra": ("TLRA", "TLRA", "TLRA", "TLRATrainer", "output_logit", "pt/best.pth"),
    "mruf": ("MRUF", "mruf", "MRUF", "MRUF", "output_logit", "pt/mosei-aligned/best.pth"),
}


@contextmanager
def native_workdir(run_dir: Path):
    """Keep upstream trainers' relative checkpoint paths inside this run."""
    previous = Path.cwd()
    try:
        os.chdir(run_dir)
        yield
    finally:
        os.chdir(previous)


class L5Adapter:
    """Bind one unchanged native network and trainer to processed_po P/O inputs."""

    input_layout = "unaligned_windowed"
    requires_text_reencoding = True

    def __init__(
        self,
        method_id: str,
        *,
        source_root: str | Path | None = None,
        bert_root: str | Path = DEFAULT_BERT_ROOT,
        device: str = "auto",
        text_reencoder: Callable[[Problem2Split], np.ndarray] | None = None,
    ) -> None:
        if method_id not in _NATIVE:
            raise ValueError(f"unknown L5 method: {method_id}")
        self.method_id = method_id
        self.directory, self.native_model_key, self.model_class, self.trainer_class, self.output_key, self.checkpoint_relpath = _NATIVE[method_id]
        self.source_root = Path(source_root) if source_root is not None else DEFAULT_VENDOR_ROOT / self.directory
        self.bert_root = Path(bert_root)
        self.device = device
        self.text_reencoder = text_reencoder

    def resolve_config(self, split: Problem2Split) -> dict:
        if not self.source_root.is_dir():
            raise FileNotFoundError(f"native source is missing: {self.source_root}")
        path = PROJECT_ROOT / "configs" / "baselines" / "l5" / f"{self.method_id}.json"
        with path.open(encoding="utf-8") as stream:
            source = json.load(stream)
        native = source[self.native_model_key]
        config = dict(source["datasetCommonParams"]["mosei"]["aligned"])
        config.update(native["commonParams"])
        config.update(native["datasetParams"]["mosei"])
        config.update(
            feature_dims=[int(split.values[name].shape[-1]) for name in ("text", "audio", "vision")],
            seq_lens=[50, 50, 50],
            dataset_name="cmumosei" if self.method_id == "cacr" else "mosei",
            model_name=self.native_model_key,
            train_mode="regression",
            need_data_aligned=True,
            use_bert=False,
            use_finetune=False,
        )
        return config

    def prepare_features(self, split: Problem2Split) -> dict[str, np.ndarray]:
        prepared = pool_unaligned_to_text_slots(split) if split.view == "unaligned_po" else split
        features = {}
        for name in ("text", "audio", "vision"):
            source = np.asarray(prepared.values[name], dtype=np.float32)
            if source.shape[1] != 50:
                raise ValueError(f"{self.method_id} requires 50 text-anchored slots for {name}")
            values = source.copy()
            values[~prepared.observed[name]] = 0.0
            features[name] = values
        return features

    def reencode_text(self, split: Problem2Split) -> Problem2Split:
        if self.text_reencoder is None:
            raise ValueError("L5 fair adapter requires text_reencoder")
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
        sanitized = replace(split, input_ids=tokens)
        text = np.asarray(self.text_reencoder(sanitized), dtype=np.float32)
        if text.shape != split.values["text"].shape or not np.isfinite(text).all():
            raise ValueError("text_reencoder must return finite XT-shaped features")
        values = {name: array.copy() for name, array in split.values.items()}
        values["text"] = text.copy()
        for name in values:
            values[name][~split.observed[name]] = 0.0
        return replace(sanitized, values=values)

    def _device(self):
        import torch

        if self.device == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _native_class(self, *, trainer: bool):
        base = self.source_root.resolve()
        if not (base / "trains" / "singleTask").is_dir():
            raise FileNotFoundError(base / "trains" / "singleTask")
        existing = sys.modules.get("trains")
        if existing is not None:
            paths = tuple(Path(path).resolve() for path in getattr(existing, "__path__", ()))
            if base / "trains" not in paths:
                raise RuntimeError("a different native 'trains' source tree is already imported")
        if str(base) not in sys.path:
            sys.path.insert(0, str(base))
        sys.dont_write_bytecode = True
        module = (
            f"trains.singleTask.{self.directory}"
            if trainer else f"trains.singleTask.model.{self.native_model_key}"
        )
        class_name = self.trainer_class if trainer else self.model_class
        return getattr(importlib.import_module(module), class_name)

    def _loader(self, split: Problem2Split, *, batch_size: int, shuffle: bool):
        import torch
        from torch.utils.data import DataLoader, Dataset

        features = self.prepare_features(split)

        class SplitDataset(Dataset):
            def __len__(self):
                return split.size

            def __getitem__(self, index):
                return {
                    "text": torch.from_numpy(features["text"][index]),
                    "audio": torch.from_numpy(features["audio"][index]),
                    "vision": torch.from_numpy(features["vision"][index]),
                    "id": str(split.ids[index]),
                    "labels": {"M": torch.tensor([float(split.regression[index])], dtype=torch.float32)},
                }

        return DataLoader(SplitDataset(), batch_size=batch_size, shuffle=shuffle, num_workers=0)

    def native_forward(self, model, features):
        kwargs = {"role": "missing", "keep_modes": "TAV", "freeze_bank": True} if self.method_id == "tlra" else {}
        return model(*features, **kwargs)[self.output_key]

    def _raw_predict(self, model, split: Problem2Split, *, batch_size: int, device) -> np.ndarray:
        import torch

        model.eval()
        predictions = []
        with torch.no_grad():
            for batch in self._loader(split, batch_size=batch_size, shuffle=False):
                features = tuple(batch[name].to(device) for name in ("text", "audio", "vision"))
                output = self.native_forward(model, features).reshape(-1)
                if not bool(torch.isfinite(output).all()):
                    raise FloatingPointError(f"non-finite {self.method_id} prediction")
                predictions.extend(output.cpu().tolist())
        raw = np.asarray(predictions, dtype=np.float64)
        if raw.shape != (split.size,):
            raise ValueError("native prediction count differs from input split")
        return raw

    def _model_group(self, model, args, device):
        if self.method_id == "emoe":
            return model
        if self.method_id != "mruf":
            return [model]
        from trains.singleTask.distillnets import get_distillation_kernel, get_distillation_kernel_homo
        from trains.singleTask.misc import softmax

        modules = [model]
        for module, hidden_size, width, prior in (
            (get_distillation_kernel_homo, 64, args.dst_feature_dim_nheads[0], [0, 0, 1, 0, 1, 0]),
            (get_distillation_kernel, 32, args.dst_feature_dim_nheads[0] * 2, [0, 0, 1, 0, 1, 1]),
        ):
            modules.append(module.DistillationKernel(
                n_classes=1, hidden_size=width, gd_size=hidden_size,
                to_idx=[0, 1, 2], from_idx=[0, 1, 2], gd_prior=softmax(prior, .25),
                gd_reg=10, w_losses=[1, 10], metric="l1", alpha=1 / 8,
                hyp_params=args,
            ).to(device))
        return modules

    def _ensure_reencoder(self, dataset_root: Path, device) -> None:
        if self.text_reencoder is None:
            self.text_reencoder = FrozenBertTextReencoder(
                self.bert_root, Path(dataset_root) / "scaler_params.npz",
                device=str(device), batch_size=32,
            )

    def train(self, dataset: Problem2Dataset, *, seed: int, run_dir: Path) -> Path:
        import torch

        device = self._device()
        self._ensure_reencoder(dataset.root, device)
        train_input = pool_unaligned_to_text_slots(dataset["train"]) if dataset.view == "unaligned_po" else dataset["train"]
        valid_input = pool_unaligned_to_text_slots(dataset["valid"]) if dataset.view == "unaligned_po" else dataset["valid"]
        train_split = self.reencode_text(train_input)
        valid_split = self.reencode_text(valid_input)
        config = self.resolve_config(train_split)
        run_dir = Path(run_dir).resolve()
        checkpoint = run_dir / self.checkpoint_relpath
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        args = NativeArgs(config)
        args.update(
            device=device, cur_seed=seed, model_save_path=str(checkpoint),
            feature_T="", feature_A="", feature_V="", custom_feature="",
            is_training=True, is_distill=True, mode="train",
        )
        loaders = {
            "train": self._loader(train_split, batch_size=int(args.batch_size), shuffle=True),
            "valid": self._loader(valid_split, batch_size=int(args.batch_size), shuffle=False),
        }
        # Native trainers evaluate 'test' every epoch; alias valid to prevent test leakage.
        loaders["test"] = loaders["valid"]
        with native_workdir(run_dir):
            model = self._native_class(trainer=False)(args).to(device)
            trainer = self._native_class(trainer=True)(args)
            trainer.do_train(self._model_group(model, args, device), loaders, return_epoch_results=False)
        if not checkpoint.is_file():
            raise RuntimeError("native trainer did not save a validation-selected checkpoint")
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
        phase2 = bool(getattr(model, "phase2", False))
        raw_valid = self._raw_predict(model, valid_split, batch_size=int(args.batch_size), device=device)
        interval = valid_interval(raw_valid, valid_split.classification)
        metadata = {
            "method_id": self.method_id,
            "seed": seed,
            "view": dataset.view,
            "dataset_root": str(dataset.root),
            "source_root": str(self.source_root.resolve()),
            "selection": {"split": "valid", "key": config["KeyEval"], "rule": "native trainer"},
            "polarity_interval": interval,
            "phase2": phase2,
            "config": config,
        }
        (run_dir / "l5.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return checkpoint

    def predict(self, split: Problem2Split, checkpoint: Path) -> RawMethodPrediction:
        import torch

        checkpoint = Path(checkpoint).resolve()
        with (checkpoint.parents[len(Path(self.checkpoint_relpath).parts) - 1] / "l5.json").open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        if metadata["method_id"] != self.method_id or metadata["view"] not in {split.view, "unaligned_po" if split.view == "unaligned_windowed" else split.view}:
            raise ValueError("checkpoint method or data view differs from requested L5 adapter")
        if Path(metadata["source_root"]).resolve() != self.source_root.resolve():
            raise ValueError("checkpoint native source differs from requested L5 adapter")
        device = self._device()
        self._ensure_reencoder(Path(metadata["dataset_root"]), device)
        args = NativeArgs(metadata["config"])
        args.device = device
        model = self._native_class(trainer=False)(args).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
        if self.method_id == "tlra":
            model.phase2 = bool(metadata["phase2"])
        raw = self._raw_predict(model, split, batch_size=int(args.batch_size), device=device)
        interval = metadata["polarity_interval"]
        classes = np.where(raw < interval["lower"], 0, np.where(raw > interval["upper"], 2, 1))
        return RawMethodPrediction(
            raw_intensity=raw,
            polarity=tuple(Polarity.from_value(int(value)) for value in classes),
            decision_source=f"valid macro_f1 interval [{interval['lower']:g}, {interval['upper']:g}]",
        )


def create_emoe_adapter(**kwargs) -> L5Adapter:
    return L5Adapter("emoe", **kwargs)


def create_cacr_adapter(**kwargs) -> L5Adapter:
    return L5Adapter("cacr", **kwargs)


def create_tlra_adapter(**kwargs) -> L5Adapter:
    return L5Adapter("tlra", **kwargs)


def create_mruf_adapter(**kwargs) -> L5Adapter:
    return L5Adapter("mruf", **kwargs)


__all__ = [
    "L5Adapter", "create_emoe_adapter", "create_cacr_adapter", "create_tlra_adapter",
    "create_mruf_adapter", "native_workdir", "valid_interval",
]
