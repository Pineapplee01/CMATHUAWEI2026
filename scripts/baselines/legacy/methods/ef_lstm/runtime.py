"""Shared canonical-NPZ data and scoring helpers for the EF-LSTM adapter."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parent
REFERENCE_ROOT = ROOT.parent
CANONICAL_ROOT = Path(
    "/user_home/gaojianan/CPMCM/AAAdata/Appendix_2/标准化/对齐版本/processed"
)
EXPECTED_MASK_SHA256 = "bddfc02985e9528bcab58a252ebb9beaa8c6a9812b609b6b88c7a547fd623916"
COMBINATIONS = ("complete", "T", "A", "V", "TA", "TV", "AV")
POSITIONS = ("beginning", "middle", "end")
FRACTIONS = (0.1, 0.3, 0.5)
MODALITY = {"T": "text", "A": "audio", "V": "vision"}


@dataclass(frozen=True)
class Bundle:
    ids: list[str]
    values: dict[str, np.ndarray]
    native: dict[str, np.ndarray]
    labels: np.ndarray


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_bundle(split: str) -> Bundle:
    import sys

    protocol = str(REFERENCE_ROOT / "protocol_src")
    if protocol not in sys.path:
        sys.path.insert(0, protocol)
    from e_emotion.data import load_processed_dataset

    dataset = load_processed_dataset(CANONICAL_ROOT)
    block = dataset[split]
    return Bundle(
        ids=[str(value) for value in block.ids.tolist()],
        values={name: np.asarray(block.features[name], dtype=np.float32) for name in ("text", "audio", "vision")},
        native={name: np.asarray(block.native_valid_mask[name], dtype=bool) for name in ("text", "audio", "vision")},
        labels=np.asarray(block.regression, dtype=np.float32),
    )


def batches(bundle: Bundle, batch_size: int, shuffle: bool, seed: int):
    order = np.arange(len(bundle.ids))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        yield {
            "ids": [bundle.ids[int(index)] for index in indices],
            "text": bundle.values["text"][indices],
            "audio": bundle.values["audio"][indices],
            "vision": bundle.values["vision"][indices],
            "masks": {name: bundle.native[name][indices] for name in bundle.native},
            "labels": bundle.labels[indices],
        }


def predict(model, bundle: Bundle, batch_size: int, device: torch.device) -> np.ndarray:
    values = []
    model.eval()
    with torch.no_grad():
        for batch in batches(bundle, batch_size, False, 0):
            inputs = {
                name: torch.as_tensor(batch[name], dtype=torch.float32, device=device)
                for name in ("text", "audio", "vision")
            }
            masks = {
                name: torch.as_tensor(batch["masks"][name], dtype=torch.bool, device=device)
                for name in ("text", "audio", "vision")
            }
            output = model(**inputs, masks=masks).reshape(-1).detach().cpu().numpy()
            if not np.isfinite(output).all():
                raise FloatingPointError("EF-LSTM produced non-finite predictions")
            values.append(output)
    return np.concatenate(values)


def calibrate(prediction: np.ndarray, target: np.ndarray) -> dict:
    target_class = np.sign(target).astype(int) + 1
    best = None
    for lower in np.linspace(-1.5, 0.0, 31):
        for upper in np.linspace(0.0, 1.5, 31):
            predicted = np.where(prediction < lower, 0, np.where(prediction > upper, 2, 1))
            score = float(f1_score(target_class, predicted, labels=[0, 1, 2], average="macro", zero_division=0))
            candidate = (-score, float(upper - lower), float(lower), float(upper))
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return {
        "lower": best[2],
        "upper": best[3],
        "fit_split": "valid",
        "objective": "macro_f1",
        "validation_macro_f1": -best[0],
        "final_intensity_rule": "clip(raw_intensity, -3, 3)",
    }


def score(target: np.ndarray, raw_prediction: np.ndarray, threshold: dict) -> dict:
    prediction = np.clip(np.asarray(raw_prediction, dtype=np.float64), -3.0, 3.0)
    target = np.asarray(target, dtype=np.float64)
    actual = np.sign(target).astype(int) + 1
    predicted = np.where(prediction < threshold["lower"], 0, np.where(prediction > threshold["upper"], 2, 1))
    return {
        "n": int(len(target)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=[0, 1, 2], average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(actual, predicted, labels=[0, 1, 2], average="weighted", zero_division=0)),
        "mae": float(np.abs(target - prediction).mean()),
        "pearson": float(np.corrcoef(target, prediction)[0, 1]) if np.std(target) and np.std(prediction) else None,
    }


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_common_q2_index() -> dict:
    index_path = REFERENCE_ROOT / "q2_test_manifest_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("mask_sha256") != EXPECTED_MASK_SHA256 or index.get("split") != "test":
        raise RuntimeError("common Q2 index does not match the fixed test protocol")
    return index


def bind_common_q2(run: Path) -> None:
    index = load_common_q2_index()
    save_json(run / "q2_mask_manifest.json", {
        "protocol_version": index["protocol_version"],
        "source_index": str(REFERENCE_ROOT / "q2_test_manifest_index.json"),
        "mask_sha256": EXPECTED_MASK_SHA256,
        "feature_version": "aligned_50",
        "data_hashes": index["data_hashes"],
        "n_conditions": 63,
    })


def q2_condition(bundle: Bundle, index: dict, combination: str, position: str, fraction: float):
    entries = {
        (str(entry["sample_id"]), entry["combination"], entry["position"], float(entry["requested_fraction"])): entry
        for entry in index["entries"]
    }
    values = {name: np.array(value, copy=True) for name, value in bundle.values.items()}
    observed = {name: np.array(mask, copy=True) for name, mask in bundle.native.items()}
    coordinate_masks = {
        "text": bundle.native["text"],
        "audio": bundle.native["text"] & bundle.native["audio"],
        "vision": bundle.native["text"] & bundle.native["vision"],
    }
    removed = eligible = 0
    for code in combination if combination != "complete" else ():
        name = MODALITY[code]
        for row, sample_id in enumerate(bundle.ids):
            entry = entries[(sample_id, combination, position, float(fraction))]
            coordinates = np.flatnonzero(coordinate_masks[name][row])
            count = int(entry["synthetic_missing_counts"][name])
            if count > len(coordinates):
                raise RuntimeError("Q2 deletion count exceeds coordinate domain")
            if count:
                start = (len(coordinates) - count) // 2
                selected = coordinates[:count] if position == "beginning" else coordinates[-count:] if position == "end" else coordinates[start:start + count]
                values[name][row, selected] = 0.0
                observed[name][row, selected] = False
            removed += count
            eligible += len(coordinates)
    return Bundle(bundle.ids, values, observed, bundle.labels), removed, eligible
