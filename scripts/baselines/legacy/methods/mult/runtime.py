"""Data, scoring, and Q2 helpers for the two standalone MulT views."""
from __future__ import annotations

from dataclasses import dataclass
import json
import pickle
from pathlib import Path
import random
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch

from data_views import resolve_view


ROOT = Path(__file__).resolve().parent
REFERENCE_ROOT = ROOT.parent
EXPECTED_ALIGNED_MASK_SHA256 = "bddfc02985e9528bcab58a252ebb9beaa8c6a9812b609b6b88c7a547fd623916"
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


def _load_aligned(split: str) -> Bundle:
    protocol = str(REFERENCE_ROOT / "protocol_src")
    if protocol not in sys.path:
        sys.path.insert(0, protocol)
    from e_emotion.data import load_processed_dataset

    block = load_processed_dataset(resolve_view("aligned_50_control").data_root)[split]
    return Bundle(
        [str(value) for value in block.ids.tolist()],
        {name: np.asarray(block.features[name], dtype=np.float32) for name in ("text", "audio", "vision")},
        {name: np.asarray(block.native_valid_mask[name], dtype=bool) for name in ("text", "audio", "vision")},
        np.asarray(block.regression, dtype=np.float32),
    )


def _load_raw(split: str) -> Bundle:
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    path = resolve_view("raw_unaligned").data_root / f"{split}.pkl"
    with path.open("rb") as stream:
        raw = pickle.load(stream)
    text = np.asarray(raw["text"], dtype=np.float32)
    audio = np.asarray(raw["audio"], dtype=np.float32)
    vision = np.asarray(raw["vision"], dtype=np.float32)
    text_mask = np.asarray(raw["text_bert"], dtype=np.float32)[:, 1, :] > 0
    def prefix(lengths, width):
        lengths = np.clip(np.asarray(lengths, dtype=np.int64), 0, width)
        return np.arange(width)[None, :] < lengths[:, None]
    return Bundle(
        [str(value) for value in raw["id"]],
        {"text": np.nan_to_num(text), "audio": np.nan_to_num(audio), "vision": np.nan_to_num(vision)},
        {"text": text_mask, "audio": prefix(raw["audio_lengths"], audio.shape[1]), "vision": prefix(raw["vision_lengths"], vision.shape[1])},
        np.asarray(raw["regression_labels"], dtype=np.float32),
    )


def load_bundle(view_name: str, split: str) -> Bundle:
    return _load_aligned(split) if resolve_view(view_name).canonical else _load_raw(split)


def batches(bundle: Bundle, batch_size: int, shuffle: bool, seed: int):
    order = np.arange(len(bundle.ids))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        yield {
            "ids": [bundle.ids[int(index)] for index in indices],
            "text": bundle.values["text"][indices], "audio": bundle.values["audio"][indices], "vision": bundle.values["vision"][indices],
            "masks": {name: bundle.native[name][indices] for name in bundle.native}, "labels": bundle.labels[indices],
        }


def predict(model, bundle: Bundle, batch_size: int, device: torch.device) -> np.ndarray:
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in batches(bundle, batch_size, False, 0):
            values = {name: torch.as_tensor(batch[name], dtype=torch.float32, device=device) for name in ("text", "audio", "vision")}
            masks = {name: torch.as_tensor(batch["masks"][name], dtype=torch.bool, device=device) for name in ("text", "audio", "vision")}
            result = model(**values, masks=masks).reshape(-1).detach().cpu().numpy()
            if not np.isfinite(result).all():
                raise FloatingPointError("MulT produced non-finite predictions")
            predictions.append(result)
    return np.concatenate(predictions)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def calibrate(prediction: np.ndarray, target: np.ndarray) -> dict:
    actual = np.sign(target).astype(int) + 1
    best = None
    for lower in np.linspace(-1.5, 0.0, 31):
        for upper in np.linspace(0.0, 1.5, 31):
            predicted = np.where(prediction < lower, 0, np.where(prediction > upper, 2, 1))
            score = float(f1_score(actual, predicted, labels=[0, 1, 2], average="macro", zero_division=0))
            candidate = (-score, float(upper - lower), float(lower), float(upper))
            if best is None or candidate < best: best = candidate
    assert best is not None
    return {"lower": best[2], "upper": best[3], "fit_split": "valid", "objective": "macro_f1", "validation_macro_f1": -best[0], "final_intensity_rule": "clip(raw_intensity, -3, 3)"}


def score(target: np.ndarray, raw: np.ndarray, threshold: dict) -> dict:
    prediction = np.clip(np.asarray(raw, dtype=np.float64), -3, 3)
    target = np.asarray(target, dtype=np.float64)
    actual = np.sign(target).astype(int) + 1
    predicted = np.where(prediction < threshold["lower"], 0, np.where(prediction > threshold["upper"], 2, 1))
    return {"n": int(len(target)), "accuracy": float(accuracy_score(actual, predicted)), "macro_f1": float(f1_score(actual, predicted, labels=[0,1,2], average="macro", zero_division=0)), "weighted_f1": float(f1_score(actual, predicted, labels=[0,1,2], average="weighted", zero_division=0)), "mae": float(np.abs(target-prediction).mean()), "pearson": float(np.corrcoef(target,prediction)[0,1]) if np.std(target) and np.std(prediction) else None}


def aligned_index() -> dict:
    index = json.loads((REFERENCE_ROOT / "q2_test_manifest_index.json").read_text(encoding="utf-8"))
    if index.get("mask_sha256") != EXPECTED_ALIGNED_MASK_SHA256 or index.get("split") != "test":
        raise RuntimeError("aligned Q2 index does not match the fixed protocol")
    return index


def bind_run(run: Path, view_name: str) -> None:
    view = resolve_view(view_name)
    payload = {"input_view": view.name, "feature_version": view.feature_version, "data_root": str(view.data_root), "n_conditions": 63}
    if view.canonical:
        index = aligned_index()
        payload.update(mask_sha256=EXPECTED_ALIGNED_MASK_SHA256, source_index=str(REFERENCE_ROOT / "q2_test_manifest_index.json"), data_hashes=index["data_hashes"])
    save_json(run / "q2_mask_manifest.json", payload)


def _positions(mask: np.ndarray, fraction: float, position: str, count: int | None = None) -> np.ndarray:
    coordinates = np.flatnonzero(mask)
    if count is None:
        count = int(round(len(coordinates) * fraction))
    if not 0 <= count <= len(coordinates):
        raise ValueError("Q2 deletion count is outside the coordinate domain")
    if position == "beginning": return coordinates[:count]
    if position == "end": return coordinates[-count:]
    start = (len(coordinates) - count) // 2
    return coordinates[start:start + count]


def q2_condition(bundle: Bundle, view_name: str, combination: str, position: str, fraction: float, index: dict | None = None):
    values = {name: np.array(value, copy=True) for name, value in bundle.values.items()}
    observed = {name: np.array(mask, copy=True) for name, mask in bundle.native.items()}
    removed = eligible = 0
    entries = None
    coordinate_masks = bundle.native
    if resolve_view(view_name).canonical:
        if index is None: raise ValueError("aligned Q2 requires the common index")
        entries = {(str(item["sample_id"]), item["combination"], item["position"], float(item["requested_fraction"])): item for item in index["entries"]}
        coordinate_masks = {"text": bundle.native["text"], "audio": bundle.native["text"] & bundle.native["audio"], "vision": bundle.native["text"] & bundle.native["vision"]}
    for code in combination if combination != "complete" else ():
        name = MODALITY[code]
        for row, sample_id in enumerate(bundle.ids):
            domain = coordinate_masks[name][row]
            expected = None
            if entries is not None:
                expected = int(entries[(sample_id, combination, position, float(fraction))]["synthetic_missing_counts"][name])
            selected = _positions(domain, fraction, position, expected)
            values[name][row, selected] = 0.0
            observed[name][row, selected] = False
            removed += len(selected); eligible += int(domain.sum())
    return Bundle(bundle.ids, values, observed, bundle.labels), removed, eligible


def raw_q2_manifest(bundle: Bundle) -> dict:
    """Record every deterministic raw-stream deletion interval for auditability."""
    entries = []
    for combination in COMBINATIONS:
        for position in POSITIONS:
            for fraction in FRACTIONS:
                for row, sample_id in enumerate(bundle.ids):
                    counts = {name: 0 for name in MODALITY.values()}
                    starts = {name: None for name in MODALITY.values()}
                    ends = {name: None for name in MODALITY.values()}
                    lengths = {name: int(bundle.native[name][row].sum()) for name in MODALITY.values()}
                    for code in combination if combination != "complete" else ():
                        name = MODALITY[code]
                        selected = _positions(bundle.native[name][row], fraction, position)
                        counts[name] = int(len(selected))
                        if len(selected):
                            starts[name] = int(selected[0])
                            ends[name] = int(selected[-1]) + 1
                    entries.append({
                        "sample_id": sample_id,
                        "combination": combination,
                        "position": position,
                        "requested_fraction": fraction,
                        "coordinate_lengths": lengths,
                        "synthetic_missing_counts": counts,
                        "synthetic_start_positions": starts,
                        "synthetic_end_positions": ends,
                    })
    return {
        "protocol_version": "raw-unaligned-continuous-local-v1",
        "input_view": "raw_unaligned",
        "coordinate_definition": "text_bert attention mask; audio_lengths; vision_lengths",
        "combinations": list(COMBINATIONS),
        "positions": list(POSITIONS),
        "fractions": list(FRACTIONS),
        "entries": entries,
    }
