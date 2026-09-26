"""Independent validation of one completed Problem 2 fair run."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from e_emotion.contracts import Polarity
from e_emotion.evaluation import read_records_csv, score_predictions
from e_emotion.problem2_fair.config import default_view_root
from e_emotion.problem2_fair.core import Q2_V2_CONDITIONS, project_intensity, validate_model_input_view
from e_emotion.problem2_fair.q2 import validate_q2_v2_manifest
from e_emotion.problem2_fair.views import load_problem2_dataset


_SPLITS = ("train", "valid", "test")
_METRICS = ("n", "accuracy", "macro_f1", "weighted_f1", "mae", "pearson")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_rows(rows: list[dict[str, str]], *, name: str) -> None:
    if not rows or not {"id", "raw_intensity", "intensity", "polarity"} <= set(rows[0]):
        raise ValueError(f"{name} predictions have missing required columns")
    for row in rows:
        raw = float(row["raw_intensity"])
        final = float(row["intensity"])
        polarity = Polarity.from_value(row["polarity"])
        expected = float(project_intensity([raw], (polarity,))[0])
        if not math.isfinite(raw) or not math.isfinite(final) or not math.isclose(
            final, expected, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"{name} intensity differs from project_intensity")


def _compare_metrics(actual: dict, recorded: dict, *, context: str) -> None:
    for name in _METRICS:
        left = actual[name]
        right = recorded.get(name)
        if left is None or right is None:
            if left is not None or right is not None:
                raise ValueError(f"{context} {name} metric differs from predictions")
        elif not math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f"{context} {name} metric differs from predictions")


def verify_run(path: str | Path, *, require_source: bool = False) -> dict:
    """Validate source, frozen decision, predictions, metrics and provenance."""
    run = Path(path).resolve()
    q2_root = run / "q2_v2"
    summary = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    q2 = json.loads((q2_root / "metrics.json").read_text(encoding="utf-8"))
    provenance = json.loads((run / "protocol_manifest.json").read_text(encoding="utf-8"))
    manifest = json.loads((q2_root / "q2_mask_manifest.json").read_text(encoding="utf-8"))
    validate_q2_v2_manifest(manifest)
    if summary.get("protocol_version") != "problem2-fair-v1":
        raise ValueError("unexpected run protocol version")
    for key in ("method", "view", "model_input_view", "seed"):
        if summary.get(key) != q2.get(key) or summary.get(key) != provenance.get(key):
            label = key.replace("_", " ")
            raise ValueError(f"run/Q2/protocol {label} identity differs")
    if summary["view"] != manifest.get("view"):
        raise ValueError("run and Q2-v2 manifest data views differ")
    validate_model_input_view(summary["view"], summary["model_input_view"])
    expected_conditions = {
        (combination, position, float(fraction)) for combination, position, fraction in Q2_V2_CONDITIONS
    }
    manifest_conditions = {
        (entry["combination"], entry["position"], float(entry["requested_fraction"]))
        for entry in manifest["conditions"]
    }
    if manifest_conditions != expected_conditions or len(manifest["conditions"]) != 64:
        raise ValueError("Q2-v2 manifest condition matrix is not canonical")
    mask_hash = manifest["mask_sha256"]
    if q2.get("mask_sha256") != mask_hash or summary.get("q2", {}).get("mask_sha256") != mask_hash:
        raise ValueError("Q2-v2 manifest mask hash differs from run metrics")
    if provenance.get("q2_mask_sha256") != mask_hash:
        raise ValueError("protocol manifest mask hash differs from Q2-v2 manifest")
    checkpoint = Path(summary["checkpoint"])
    expected_checkpoint_hash = summary["checkpoint_sha256"]
    if (
        q2.get("checkpoint_sha256") != expected_checkpoint_hash
        or provenance.get("checkpoint_sha256") != expected_checkpoint_hash
        or _sha256(checkpoint) != expected_checkpoint_hash
    ):
        raise ValueError("checkpoint file hash differs from recorded checkpoint hash")

    if provenance.get("data_root") != summary.get("data_root"):
        raise ValueError("protocol manifest data root differs from run metrics")
    recorded_root = summary.get("data_root")
    source_root = Path(recorded_root) if recorded_root else Path(default_view_root(summary["view"]))
    dataset = None
    if source_root.is_dir():
        for split in _SPLITS:
            if _sha256(source_root / f"{split}.npz") != manifest["data_hashes"][split]:
                raise ValueError(f"{split} data hash differs from Q2-v2 manifest")
        dataset = load_problem2_dataset(source_root, view=summary["view"])
    elif recorded_root:
        raise ValueError("recorded data root does not exist")
    elif require_source:
        raise ValueError("fair report requires reopenable source NPZ data")

    valid_rows = read_records_csv(run / "valid_predictions.csv")
    clean_rows = read_records_csv(run / "test_predictions.csv")
    _check_rows(valid_rows, name="valid")
    _check_rows(clean_rows, name="clean")
    for split_name, rows in (("valid", valid_rows), ("test", clean_rows)):
        expected_ids = {entry["sample_id"] for entry in manifest["entries"] if entry["split"] == split_name}
        if {row["id"] for row in rows} != expected_ids:
            raise ValueError(f"{split_name} prediction ID coverage differs from Q2-v2 manifest")

    with (q2_root / "predictions.csv").open(encoding="utf-8", newline="") as source:
        predictions = list(csv.DictReader(source))
    if not predictions or not {"combination", "position", "requested_fraction"} <= set(predictions[0]):
        raise ValueError("Q2-v2 predictions have missing condition columns")
    _check_rows(predictions, name="Q2-v2")
    indexed: dict[tuple[str, str | None, float], dict[str, dict[str, str]]] = {}
    seen: set[tuple[str, str, str | None, float]] = set()
    for row in predictions:
        condition = (row["combination"], row["position"] or None, float(row["requested_fraction"]))
        key = (row["id"], *condition)
        if key in seen:
            raise ValueError("duplicate Q2-v2 condition/sample prediction")
        seen.add(key)
        indexed.setdefault(condition, {})[row["id"]] = row
    expected = {
        (str(entry["sample_id"]), entry["combination"], entry["position"], float(entry["requested_fraction"]))
        for entry in manifest["entries"] if entry["split"] == "test"
    }
    if seen != expected or q2.get("n_conditions") != 64 or q2.get("n_predictions") != len(predictions):
        raise ValueError("Q2-v2 artifact coverage is incomplete")
    if len(q2.get("conditions", [])) != 64:
        raise ValueError("Q2-v2 metrics omit conditions")
    clean_by_id = {row["id"]: row for row in clean_rows}
    complete = indexed[("complete", None, 0.0)]
    for sample_id, row in complete.items():
        if any(row[field] != clean_by_id[sample_id][field] for field in ("raw_intensity", "intensity", "polarity")):
            raise ValueError("Q2-v2 complete predictions differ from clean test predictions")
    if dataset is not None:
        truth = {str(sid): float(value) for sid, value in zip(dataset["test"].ids, dataset["test"].regression)}
        ordered_clean = [clean_by_id[str(sid)] for sid in dataset["test"].ids]
        actual_clean = score_predictions(
            dataset["test"].regression,
            [float(row["intensity"]) for row in ordered_clean],
            [row["polarity"] for row in ordered_clean],
            truth_polarity=dataset["test"].classification,
        )
        _compare_metrics(actual_clean, summary["clean"], context="clean")
        if len(truth) != len(ordered_clean):
            raise ValueError("clean test ID coverage differs from source")
        for condition in q2["conditions"]:
            key = (condition["combination"], condition["position"], float(condition["requested_fraction"]))
            if key not in expected_conditions or condition.get("status") != "ok":
                raise ValueError("Q2-v2 condition is noncanonical or failed")
            rows = indexed[key]
            ordered = [rows[str(sid)] for sid in dataset["test"].ids]
            actual_metrics = score_predictions(
                dataset["test"].regression,
                [float(row["intensity"]) for row in ordered],
                [row["polarity"] for row in ordered],
                truth_polarity=dataset["test"].classification,
            )
            _compare_metrics(actual_metrics, condition, context=f"Q2-v2 {key}")
    return {"status": "ok", "run": str(run), "q2_predictions": len(predictions)}


__all__ = ["verify_run"]
