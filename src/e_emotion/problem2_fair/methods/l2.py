"""Fair Problem 2 bridges for the original MMSA L2 model implementations."""

from __future__ import annotations

from dataclasses import replace
import importlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import types
from typing import Callable

import numpy as np

from e_emotion.baselines.workspace import bert_root, vendor_root
from e_emotion.contracts import Polarity
from e_emotion.problem2_fair.core import Problem2Split
from e_emotion.problem2_fair.q2 import pool_unaligned_to_text_slots
from e_emotion.problem2_fair.run import RawMethodPrediction
from e_emotion.problem2_fair.views import Problem2Dataset


DEFAULT_MMSA_ROOT = vendor_root() / "MMSA"
DEFAULT_BERT_ROOT = bert_root()
_NATIVE_CLASSES = {
    "tfn": ("tfn", "TFN"),
    "mfn": ("mfn", "MFN"),
    "graph_mfn_dfg": ("graph_mfn", "Graph_MFN"),
}


class NativeArgs(dict):
    """Attribute access expected by MMSA without a runtime easydict dependency."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def _prepared_split(split: Problem2Split) -> Problem2Split:
    return pool_unaligned_to_text_slots(split) if split.view == "unaligned_po" else split


def _masked_features(split: Problem2Split, method_id: str) -> dict[str, np.ndarray]:
    prepared = _prepared_split(split)
    features: dict[str, np.ndarray] = {}
    for name in ("text", "audio", "vision"):
        source = np.asarray(prepared.values[name], dtype=np.float32)
        if source.shape[1] != 50:
            raise ValueError(f"{method_id} needs 50 text-anchored slots for {name}")
        values = np.array(source, copy=True)
        values[~prepared.observed[name]] = 0.0
        if method_id == "tfn" and name != "text":
            # MMSA TFN's need_normalized path averages A/V over the time axis.
            values = values.mean(axis=1, keepdims=True)
        features[name] = values
    return features


class FrozenBertTextReencoder:
    """Rebuild standardized XT from surviving token IDs and the view scaler."""

    def __init__(self, model_root: Path, scaler_path: Path, *, device: str, batch_size: int) -> None:
        import torch
        from transformers import AutoModel

        if not model_root.is_dir() or not scaler_path.is_file():
            raise FileNotFoundError(f"BERT model or processed_po scaler is missing: {model_root}, {scaler_path}")
        with np.load(scaler_path, allow_pickle=False) as payload:
            self.mean = np.asarray(payload["mu_T"], dtype=np.float32)
            self.std = np.asarray(payload["sigma_T"], dtype=np.float32)
        if self.mean.ndim != 1 or self.std.shape != self.mean.shape or np.any(self.std <= 0):
            raise ValueError("invalid view text scaler")
        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model = AutoModel.from_pretrained(model_root, local_files_only=True).to(self.device).eval()
        self.model.requires_grad_(False)

    def __call__(self, split: Problem2Split) -> np.ndarray:
        torch = self.torch
        result = np.zeros(split.values["text"].shape, dtype=np.float32)
        if result.shape[-1] != len(self.mean):
            raise ValueError("view text scaler dimension differs from XT")
        with torch.no_grad():
            for start in range(0, split.size, self.batch_size):
                end = min(split.size, start + self.batch_size)
                token_ids = torch.as_tensor(split.input_ids[start:end].copy(), dtype=torch.long, device=self.device)
                attention = token_ids.ne(0)
                empty = ~attention.any(dim=1)
                token_ids[empty, 0] = 101
                attention[empty, 0] = True
                result[start:end] = self.model(
                    input_ids=token_ids, attention_mask=attention.long()
                ).last_hidden_state.cpu().numpy()
        result = (result - self.mean) / self.std
        result[~split.observed["text"]] = 0.0
        return result


def _macro_f1(truth: np.ndarray, predictions: np.ndarray) -> float:
    scores = []
    for label in (0, 1, 2):
        true = truth == label
        pred = predictions == label
        tp = np.count_nonzero(true & pred)
        denominator = np.count_nonzero(true) + np.count_nonzero(pred)
        scores.append(0.0 if denominator == 0 else 2.0 * tp / denominator)
    return float(np.mean(scores))


def _valid_interval(raw: np.ndarray, truth: np.ndarray) -> dict[str, float | str]:
    candidates = []
    for lower in np.linspace(-1.5, 0.0, 31):
        for upper in np.linspace(0.0, 1.5, 31):
            classes = np.where(raw < lower, 0, np.where(raw > upper, 2, 1))
            candidates.append((-_macro_f1(truth, classes), float(upper - lower), float(lower), float(upper)))
    score, _, lower, upper = min(candidates)
    return {
        "lower": lower,
        "upper": upper,
        "fit_split": "valid",
        "objective": "macro_f1",
        "validation_macro_f1": -score,
    }


class MMSAL2Adapter:
    """Use the verified MMSA models/trainers with public P/O-aware inputs."""

    input_layout = "unaligned_windowed"
    requires_text_reencoding = True

    def __init__(
        self,
        method_id: str,
        *,
        source_root: str | Path = DEFAULT_MMSA_ROOT,
        bert_root: str | Path = DEFAULT_BERT_ROOT,
        device: str = "auto",
        text_reencoder: Callable[[Problem2Split], np.ndarray] | None = None,
    ) -> None:
        if method_id not in _NATIVE_CLASSES:
            raise ValueError(f"unknown MMSA L2 method: {method_id}")
        self.method_id = method_id
        self.native_model_key, self.native_class_name = _NATIVE_CLASSES[method_id]
        self.source_root = Path(source_root)
        self.bert_root = Path(bert_root)
        self.device = device
        self.text_reencoder = text_reencoder

    def resolve_config(self, split: Problem2Split) -> dict:
        path = self.source_root / "cpmcm_config.json"
        with path.open(encoding="utf-8") as stream:
            source = json.load(stream)
        native = source[self.native_model_key]
        # All three bridges consume the approved equal-length view after the
        # shared raw 500-to-50 pooling step.
        config = dict(source["datasetCommonParams"]["mosei"]["aligned"])
        config.update(native["commonParams"])
        config.update(native["datasetParams"]["mosei"])
        config["feature_dims"] = [int(split.values[name].shape[-1]) for name in ("text", "audio", "vision")]
        config["seq_lens"] = [50, 50, 50]
        config["dataset_name"] = "mosei"
        config["model_name"] = self.native_model_key
        config["train_mode"] = "regression"
        config["need_data_aligned"] = True
        if self.method_id == "mfn":
            # Its forward loops over text time, so 1-step normalized A/V
            # from the stock config cannot be fed to this original network.
            config["need_normalized"] = False
        return config

    def prepare_features(self, split: Problem2Split) -> dict[str, np.ndarray]:
        return _masked_features(split, self.method_id)

    def reencode_text(self, split: Problem2Split) -> Problem2Split:
        if self.text_reencoder is None:
            raise ValueError("MMSA L2 fair adapter requires text_reencoder")
        token_ids = np.array(split.input_ids, copy=True)
        token_ids[~split.observed["text"]] = 0
        for row, support in enumerate(split.physical_support["text"]):
            positions = np.flatnonzero(support)
            if not len(positions):
                continue
            first, last = int(positions[0]), int(positions[-1])
            if first > 0:
                token_ids[row, first - 1] = 101
            if last + 1 < token_ids.shape[1]:
                token_ids[row, last + 1] = 102
        sanitized = replace(split, input_ids=token_ids)
        encoded = np.asarray(self.text_reencoder(sanitized), dtype=np.float32)
        if encoded.shape != split.values["text"].shape or not np.isfinite(encoded).all():
            raise ValueError("text_reencoder must return finite XT-shaped features")
        text = encoded.copy()
        text[~split.observed["text"]] = 0.0
        return replace(sanitized, values={**split.values, "text": text})

    def _device(self):
        import torch

        return torch.device("cuda:0" if self.device == "auto" and torch.cuda.is_available() else
                            "cpu" if self.device == "auto" else self.device)

    def _native_class(self, *, trainer: bool):
        base = self.source_root / "src" / "MMSA"
        if not base.is_dir():
            raise FileNotFoundError(base)
        existing = sys.modules.get("MMSA")
        if existing is not None:
            paths = getattr(existing, "__path__", ())
            if not paths or Path(paths[0]).resolve() != base.resolve():
                raise RuntimeError("a different MMSA source tree is already imported")
        else:
            # MMSA/__init__.py eagerly imports unrelated models and optional
            # dependencies.  Load only the original L2 modules and their
            # FeatureNets dependency under their proper package names.
            for suffix in ("", "models", "models.singleTask", "models.subNets", "trains", "trains.singleTask", "utils"):
                name = "MMSA" + ("." + suffix if suffix else "")
                package = types.ModuleType(name)
                package.__path__ = [str(base.joinpath(*suffix.split(".")))] if suffix else [str(base)]
                sys.modules[name] = package
            name = "MMSA.models.subNets.FeatureNets"
            path = base / "models" / "subNets" / "FeatureNets.py"
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load native MMSA FeatureNets: {path}")
            feature_module = importlib.util.module_from_spec(spec)
            sys.modules[name] = feature_module
            spec.loader.exec_module(feature_module)
            subnet_package = sys.modules["MMSA.models.subNets"]
            subnet_package.SubNet = feature_module.SubNet
            subnet_package.TextSubNet = feature_module.TextSubNet
            native_metrics = importlib.import_module("MMSA.utils.metricsTop")
            utils_package = sys.modules["MMSA.utils"]
            utils_package.MetricsTop = native_metrics.MetricsTop
            # Native trainers use this formatter only for logging.  Importing
            # MMSA.utils.functions would require optional GPU discovery pynvml.
            utils_package.dict_to_str = lambda values: "".join(
                " %s: %.4f " % (key, value) for key, value in values.items()
            )
        kind = "trains" if trainer else "models"
        module = importlib.import_module(f"MMSA.{kind}.singleTask.{self.native_class_name}")
        return getattr(module, self.native_class_name)

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

    def _raw_predict(self, model, split: Problem2Split, *, batch_size: int, device) -> np.ndarray:
        import torch

        model.eval()
        result = []
        with torch.no_grad():
            for batch in self._loader(split, batch_size=batch_size, shuffle=False):
                outputs = model(*(batch[name].to(device) for name in ("text", "audio", "vision")))["M"]
                if not bool(torch.isfinite(outputs).all()):
                    raise FloatingPointError("non-finite MMSA L2 prediction")
                result.extend(outputs.reshape(-1).cpu().tolist())
        return np.asarray(result, dtype=np.float64)

    def train(self, dataset: Problem2Dataset, *, seed: int, run_dir: Path) -> Path:
        import torch

        if self.text_reencoder is None:
            self.text_reencoder = FrozenBertTextReencoder(
                self.bert_root, Path(dataset.root) / "scaler_params.npz",
                device=str(self._device()), batch_size=32,
            )
        train = self.reencode_text(_prepared_split(dataset["train"]))
        valid = self.reencode_text(_prepared_split(dataset["valid"]))
        config = self.resolve_config(train)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        device = self._device()
        run_dir = Path(run_dir)
        checkpoint = run_dir / "best.pth"
        args = NativeArgs(config)
        args.update(device=device, cur_seed=seed, model_save_path=str(checkpoint))
        model = self._native_class(trainer=False)(args).to(device)
        trainer = self._native_class(trainer=True)(args)
        if self.method_id == "graph_mfn_dfg" and not any("graph_mfn.networks." in k for k in model.state_dict()):
            raise RuntimeError("Graph-MFN DFG node networks are not registered in the checkpoint")
        loaders = {
            "train": self._loader(train, batch_size=int(args.batch_size), shuffle=True),
            "valid": self._loader(valid, batch_size=int(args.batch_size), shuffle=False),
        }
        loaders["test"] = loaders["valid"]  # Native trainer receives no test data during checkpoint selection.
        trainer.do_train(model, loaders, return_epoch_results=False)
        if not checkpoint.is_file():
            raise RuntimeError("native trainer did not save a validation-selected checkpoint")
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
        raw_valid = self._raw_predict(model, valid, batch_size=int(args.batch_size), device=device)
        interval = _valid_interval(raw_valid, np.asarray(valid.classification, dtype=np.int64))
        metadata = {
            "method_id": self.method_id,
            "native_model_key": self.native_model_key,
            "native_class_name": self.native_class_name,
            "seed": seed,
            "view": dataset.view,
            "selection": {"split": "valid", "key": config["KeyEval"], "rule": "native MMSA trainer"},
            "polarity_interval": interval,
            "config": config,
        }
        (run_dir / "mmsa_l2.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return checkpoint

    def predict(self, split: Problem2Split, checkpoint: Path) -> RawMethodPrediction:
        import torch

        checkpoint = Path(checkpoint)
        metadata_path = checkpoint.with_name("mmsa_l2.json")
        with metadata_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        if metadata["method_id"] != self.method_id:
            raise ValueError("checkpoint method differs from requested L2 adapter")
        device = self._device()
        args = NativeArgs(metadata["config"])
        args.device = device
        model = self._native_class(trainer=False)(args).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
        raw = self._raw_predict(model, split, batch_size=int(args.batch_size), device=device)
        interval = metadata["polarity_interval"]
        classes = np.where(raw < interval["lower"], 0, np.where(raw > interval["upper"], 2, 1))
        return RawMethodPrediction(
            raw_intensity=raw,
            polarity=tuple(Polarity.from_value(int(value)) for value in classes),
            decision_source=f"valid macro_f1 interval [{interval['lower']:g}, {interval['upper']:g}]",
        )


def create_tfn_adapter(**kwargs) -> MMSAL2Adapter:
    return MMSAL2Adapter("tfn", **kwargs)


def create_mfn_adapter(**kwargs) -> MMSAL2Adapter:
    return MMSAL2Adapter("mfn", **kwargs)


def create_graph_mfn_adapter(**kwargs) -> MMSAL2Adapter:
    return MMSAL2Adapter("graph_mfn_dfg", **kwargs)


__all__ = [
    "DEFAULT_MMSA_ROOT",
    "FrozenBertTextReencoder",
    "MMSAL2Adapter",
    "NativeArgs",
    "create_graph_mfn_adapter",
    "create_mfn_adapter",
    "create_tfn_adapter",
]
