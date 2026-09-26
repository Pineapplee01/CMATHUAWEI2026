"""Pure Problem 2 fair-protocol input conversion for the AUMDF model."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from e_emotion.contracts import Polarity
from e_emotion.problem2_fair import Problem2Dataset, Problem2Split, pool_unaligned_to_text_slots
from e_emotion.problem2_fair.run import RawMethodPrediction

from e_emotion.baselines.aumdf.model import MODALITIES


DEFAULT_BERT_ROOT = Path("/user_home/gaojianan/CPMCM/AAAmodel/bert-base-uncased")


class FrozenBertTextReencoder:
    """Rebuild organizer-standardized XT from the surviving token context."""

    def __init__(
        self,
        model_dir: str | Path,
        scaler_path: str | Path,
        *,
        device: str = "auto",
        batch_size: int = 32,
    ) -> None:
        from transformers import AutoModel

        model_dir = Path(model_dir)
        scaler_path = Path(scaler_path)
        if not model_dir.is_dir():
            raise FileNotFoundError(model_dir)
        if not scaler_path.is_file():
            raise FileNotFoundError(scaler_path)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        with np.load(scaler_path, allow_pickle=False) as scaler:
            self.mean = np.asarray(scaler["mu_T"], dtype=np.float32)
            self.std = np.asarray(scaler["sigma_T"], dtype=np.float32)
        if self.mean.ndim != 1 or self.std.shape != self.mean.shape or np.any(self.std <= 0):
            raise ValueError("invalid frozen text scaler")
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                                   "cpu" if device == "auto" else device)
        self.batch_size = batch_size
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(self.device).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, split: Problem2Split) -> np.ndarray:
        if split.values["text"].shape[-1] != len(self.mean):
            raise ValueError("text scaler dimension does not match XT")
        encoded = np.zeros(split.values["text"].shape, dtype=np.float32)
        for start in range(0, split.size, self.batch_size):
            end = min(start + self.batch_size, split.size)
            ids = torch.as_tensor(split.input_ids[start:end].copy(), dtype=torch.long, device=self.device)
            attention = ids.ne(0)
            empty = ~attention.any(dim=1)
            ids[empty, 0] = 101
            attention[empty, 0] = True
            hidden = self.model(input_ids=ids, attention_mask=attention.long()).last_hidden_state
            encoded[start:end] = hidden.cpu().numpy().astype(np.float32)
        encoded = (encoded - self.mean) / self.std
        encoded[~split.observed["text"]] = 0.0
        return encoded


def adapt_problem2_split(split: Problem2Split) -> dict[str, object]:
    """Return a batched AUMDF input, pooling raw A/V after Q2 deletion."""
    prepared = pool_unaligned_to_text_slots(split) if split.view == "unaligned_po" else split
    features: dict[str, torch.Tensor] = {}
    valid: dict[str, torch.Tensor] = {}
    observed: dict[str, torch.Tensor] = {}
    for modality in MODALITIES:
        values = np.asarray(prepared.values[modality], dtype=np.float32)
        if values.shape[1] != 50:
            raise ValueError(f"AUMDF requires 50 slots for {modality}")
        physical = np.asarray(prepared.physical_support[modality], dtype=bool)
        native_observed = np.asarray(prepared.observed[modality], dtype=bool)
        masked = values.copy()
        masked[~native_observed] = 0.0
        features[modality] = torch.from_numpy(masked)
        valid[modality] = torch.from_numpy(physical.copy())
        observed[modality] = torch.from_numpy(native_observed.copy())
    return {
        "id": tuple(str(sample_id) for sample_id in prepared.ids),
        "features": features,
        "valid": valid,
        "observed": observed,
        "target": torch.from_numpy(np.asarray(prepared.regression, dtype=np.float32).copy()),
    }


class AUMDFFairDataset(Dataset):
    """Sample-wise view of the same P/O-aware batch used at inference."""

    def __init__(self, split: Problem2Split) -> None:
        self.batch = adapt_problem2_split(split)

    def __len__(self) -> int:
        return len(self.batch["id"])

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "id": self.batch["id"][index],
            "features": {name: self.batch["features"][name][index] for name in MODALITIES},
            "valid": {name: self.batch["valid"][name][index] for name in MODALITIES},
            "observed": {name: self.batch["observed"][name][index] for name in MODALITIES},
            "target": self.batch["target"][index],
        }


class AUMDFFairAdapter:
    """AUMDF training and inference bridge for the shared fair runner."""

    method_id = "aumdf"
    input_layout = "unaligned_windowed"
    requires_text_reencoding = True

    def __init__(
        self,
        config: Mapping[str, object] | None = None,
        *,
        device: str = "auto",
        text_reencoder: Callable[[Problem2Split], np.ndarray] | None = None,
        bert_dir: str | Path = DEFAULT_BERT_ROOT,
    ) -> None:
        self.config = copy.deepcopy(dict(config)) if config is not None else None
        self.device = device
        self.text_reencoder = text_reencoder
        self.bert_dir = Path(bert_dir)

    def reencode_text(self, split: Problem2Split) -> Problem2Split:
        if self.text_reencoder is None:
            raise ValueError("AUMDF fair bridge requires text_reencoder")
        input_ids = np.array(split.input_ids, copy=True)
        input_ids[~split.observed["text"]] = 0
        for row, support in enumerate(split.physical_support["text"]):
            positions = np.flatnonzero(support)
            if not len(positions):
                continue
            first, last = int(positions[0]), int(positions[-1])
            if first > 0:
                input_ids[row, first - 1] = 101
            if last + 1 < input_ids.shape[1]:
                input_ids[row, last + 1] = 102
        sanitized = replace(split, input_ids=input_ids)
        encoded = np.asarray(self.text_reencoder(sanitized), dtype=np.float32)
        if encoded.shape != split.values["text"].shape or not np.isfinite(encoded).all():
            raise ValueError("text_reencoder must return finite text features matching XT shape")
        text = encoded.copy()
        text[~split.observed["text"]] = 0.0
        return replace(sanitized, values={**split.values, "text": text})

    def _config(self) -> dict:
        if self.config is not None:
            return copy.deepcopy(self.config)
        path = Path(__file__).resolve().parents[4] / "configs" / "aumdf" / "adapted.yaml"
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def train(self, dataset: Problem2Dataset, *, seed: int, run_dir: Path) -> Path:
        if self.text_reencoder is None:
            config = self._config()
            self.text_reencoder = FrozenBertTextReencoder(
                self.bert_dir,
                Path(dataset.root) / "scaler_params.npz",
                device=self.device,
                batch_size=int(config["training"].get("batch_size", 32)),
            )
        from e_emotion.baselines.aumdf.engine import train_run

        config = self._config()
        config.pop("data_file", None)
        config["fair_view"] = dataset.view
        config["fair_data_root"] = str(dataset.root)
        config["training"]["seed"] = seed
        fair_datasets = {name: AUMDFFairDataset(dataset[name]) for name in ("train", "valid")}
        output = Path(run_dir) / "aumdf"
        train_run(config, output, device=self.device, fair_datasets=fair_datasets)
        return output / "student.pt"

    @torch.no_grad()
    def predict(self, split: Problem2Split, checkpoint: Path) -> RawMethodPrediction:
        from e_emotion.baselines.aumdf.engine import choose_device, restore

        device = choose_device(self.device)
        model, payload = restore(checkpoint, device)
        model.eval()
        batch_size = int(payload["config"]["training"].get("batch_size", 32))
        loader = DataLoader(AUMDFFairDataset(split), batch_size=batch_size, shuffle=False)
        scores = []
        classes = []
        for batch in loader:
            features = {name: batch["features"][name].to(device) for name in MODALITIES}
            valid = {name: batch["valid"][name].to(device) for name in MODALITIES}
            observed = {name: batch["observed"][name].to(device) for name in MODALITIES}
            output = model(features, valid, observed)
            scores.extend(output.score.cpu().tolist())
            if output.logits is not None:
                classes.extend((output.logits.argmax(-1) - 3).sign().add(1).cpu().tolist())
        raw = np.asarray(scores, dtype=np.float64)
        if model.config.readout == "classification":
            polarity = tuple(Polarity.from_value(value) for value in classes)
            decision_source = "logits.argmax()"
        else:
            threshold = float(payload["neutral_threshold"])
            clipped = np.clip(raw, -3.0, 3.0)
            polarity = tuple(
                Polarity.from_value(value)
                for value in np.where(clipped < -threshold, 0, np.where(clipped > threshold, 2, 1))
            )
            decision_source = f"valid neutral interval ({threshold:g})"
        return RawMethodPrediction(raw_intensity=raw, polarity=polarity, decision_source=decision_source)


def create_adapter() -> AUMDFFairAdapter:
    """Registry factory for the Problem 2 Baseline runner."""
    return AUMDFFairAdapter()


__all__ = [
    "AUMDFFairAdapter",
    "AUMDFFairDataset",
    "FrozenBertTextReencoder",
    "adapt_problem2_split",
    "create_adapter",
]
