"""Model-agnostic Problem 2 fair baseline execution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from e_emotion.contracts import Polarity
from e_emotion.evaluation import score_predictions, write_predictions_csv
from e_emotion.problem2_fair.catalog import is_frozen_output_path
from e_emotion.problem2_fair.core import Problem2Split, Q2_V2_CONDITIONS, finalize_prediction, validate_model_input_view
from e_emotion.problem2_fair.q2 import ensure_q2_v2_manifest, materialize_q2_condition, pool_unaligned_to_text_slots
from e_emotion.problem2_fair.views import Problem2Dataset


PROTOCOL_VERSION = "problem2-fair-v1"


@dataclass(frozen=True)
class RawMethodPrediction:
    """Method-owned polarity decision plus its unmodified intensity output."""

    raw_intensity: np.ndarray
    polarity: tuple[Polarity, ...]
    decision_source: str

    def __post_init__(self) -> None:
        raw = np.asarray(self.raw_intensity, dtype=np.float64)
        if raw.ndim != 1 or len(raw) != len(self.polarity) or not np.isfinite(raw).all():
            raise ValueError("raw method prediction requires finite matching vectors")
        object.__setattr__(self, "raw_intensity", raw)
        object.__setattr__(self, "polarity", tuple(Polarity.from_value(value) for value in self.polarity))
        if not self.decision_source:
            raise ValueError("decision_source is required")


class BaselineMethodAdapter(Protocol):
    """The only seam between the public runner and a method implementation."""

    method_id: str

    def train(self, dataset: Problem2Dataset, *, seed: int, run_dir: Path) -> Path: ...

    def predict(self, split: Problem2Split, checkpoint: Path) -> RawMethodPrediction: ...

    def reencode_text(self, split: Problem2Split) -> Problem2Split: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(split: Problem2Split, result) -> list[dict[str, object]]:
    return [
        {
            "id": str(sample_id),
            "raw_intensity": float(raw),
            "intensity": float(final),
            "polarity": label.value,
        }
        for sample_id, raw, final, label in zip(split.ids, result.raw_intensity, result.intensity, result.polarity)
    ]


class BaselineRun:
    """Run one method against one data view without changing its implementation."""

    def __init__(self, adapter: BaselineMethodAdapter) -> None:
        self.adapter = adapter

    def _prepare(self, split: Problem2Split) -> Problem2Split:
        layout = getattr(self.adapter, "input_layout", "native")
        if layout == "unaligned_windowed":
            prepared = pool_unaligned_to_text_slots(split) if split.view == "unaligned_po" else split
        elif layout == "native":
            prepared = split
        else:
            raise ValueError(f"unsupported adapter input layout: {layout!r}")
        reencoder = getattr(self.adapter, "reencode_text", None)
        if not callable(reencoder):
            raise ValueError("adapter must implement reencode_text for Problem 2 fair evaluation")
        refreshed = reencoder(prepared)
        if not isinstance(refreshed, Problem2Split):
            raise ValueError("adapter reencode_text must return Problem2Split")
        return refreshed

    def _predict(self, split: Problem2Split, checkpoint: Path):
        native = self.adapter.predict(self._prepare(split), checkpoint)
        return finalize_prediction(native.raw_intensity, native.polarity), native.decision_source

    def execute(
        self,
        dataset: Problem2Dataset,
        *,
        seed: int,
        run_dir: str | Path,
        manifest_path: str | Path | None = None,
    ) -> dict:
        target = Path(run_dir).resolve()
        if is_frozen_output_path(target):
            raise ValueError("run_dir is inside a frozen method directory")
        if manifest_path is not None and is_frozen_output_path(Path(manifest_path).resolve()):
            raise ValueError("manifest_path is inside a frozen method directory")
        if target.exists():
            raise FileExistsError(target)
        target.mkdir(parents=True)
        model_input_view = (
            "unaligned_windowed"
            if dataset.view == "unaligned_po" and getattr(self.adapter, "input_layout", "native") == "unaligned_windowed"
            else dataset.view
        )
        validate_model_input_view(dataset.view, model_input_view)
        checkpoint = self.adapter.train(dataset, seed=seed, run_dir=target)
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

        valid, valid_source = self._predict(dataset["valid"], checkpoint)
        clean, clean_source = self._predict(dataset["test"], checkpoint)
        clean_metrics = score_predictions(
            dataset["test"].regression,
            clean.intensity,
            clean.polarity,
            truth_polarity=dataset["test"].classification,
        )
        write_predictions_csv(target / "valid_predictions.csv", _rows(dataset["valid"], valid))
        write_predictions_csv(target / "test_predictions.csv", _rows(dataset["test"], clean))

        q2_dir = target / "q2_v2"
        q2_dir.mkdir()
        manifest = ensure_q2_v2_manifest(dataset, manifest_path or (q2_dir / "q2_mask_manifest.json"))
        (q2_dir / "q2_mask_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        conditions: list[dict] = []
        q2_rows: list[dict[str, object]] = []
        entries_by_condition: dict[tuple[str, str | None, float], list[dict]] = {}
        for entry in manifest["entries"]:
            if entry["split"] == "test":
                key = (entry["combination"], entry["position"], float(entry["requested_fraction"]))
                entries_by_condition.setdefault(key, []).append(entry)
        for combination, position, fraction in Q2_V2_CONDITIONS:
            condition = materialize_q2_condition(
                dataset["test"], combination, position, fraction,
                entries=entries_by_condition.get((combination, position, fraction), []),
            )
            final, source = self._predict(condition, checkpoint)
            metrics = score_predictions(
                condition.regression,
                final.intensity,
                final.polarity,
                truth_polarity=condition.classification,
            )
            conditions.append(
                {
                    "combination": combination,
                    "position": position,
                    "requested_fraction": fraction,
                    "status": "ok",
                    "decision_source": source,
                    **metrics,
                }
            )
            q2_rows.extend(
                {
                    **row,
                    "combination": combination,
                    "position": position,
                    "requested_fraction": fraction,
                }
                for row in _rows(condition, final)
            )
        if len(q2_rows) != dataset["test"].size * len(Q2_V2_CONDITIONS):
            raise RuntimeError("Q2-v2 prediction coverage is incomplete")
        with (q2_dir / "predictions.csv").open("x", encoding="utf-8", newline="") as stream:
            import csv

            writer = csv.DictWriter(stream, fieldnames=list(q2_rows[0]))
            writer.writeheader()
            writer.writerows(q2_rows)
        q2 = {
            "protocol_version": PROTOCOL_VERSION,
            "q2_protocol_version": manifest["protocol_version"],
            "method": self.adapter.method_id,
            "view": dataset.view,
            "model_input_view": model_input_view,
            "seed": seed,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "mask_sha256": manifest["mask_sha256"],
            "n_conditions": len(conditions),
            "n_predictions": len(q2_rows),
            "conditions": conditions,
        }
        (q2_dir / "metrics.json").write_text(
            json.dumps(q2, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        result = {
            "protocol_version": PROTOCOL_VERSION,
            "method": self.adapter.method_id,
            "view": dataset.view,
            "model_input_view": model_input_view,
            "seed": seed,
            "data_root": str(dataset.root) if all((dataset.root / f"{name}.npz").is_file() for name in ("train", "valid", "test")) else None,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "decision_sources": {"valid": valid_source, "clean_test": clean_source},
            "clean": clean_metrics,
            "q2": {"n_conditions": q2["n_conditions"], "n_predictions": q2["n_predictions"], "mask_sha256": q2["mask_sha256"]},
        }
        (target / "metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        (target / "protocol_manifest.json").write_text(
            json.dumps(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "method": self.adapter.method_id,
                    "view": dataset.view,
                    "model_input_view": model_input_view,
                    "seed": seed,
                    "data_root": result["data_root"],
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": _sha256(checkpoint),
                    "q2_mask_sha256": manifest["mask_sha256"],
                    "decision_sources": result["decision_sources"],
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return result


__all__ = ["BaselineMethodAdapter", "BaselineRun", "PROTOCOL_VERSION", "RawMethodPrediction"]
