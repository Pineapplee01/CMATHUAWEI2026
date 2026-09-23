"""Model-independent prediction CSVs and exact-ID coverage checks."""
import csv
from pathlib import Path

from e_emotion.contracts import Polarity
from e_emotion.evaluation.metrics import vector
from e_emotion.evaluation.outputs import AdaptedOutput, classes, intensities
from e_emotion.evaluation.scoring import score_predictions


def _indexed(rows):
    indexed = {}
    for row in rows:
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id.strip() or sample_id != sample_id.strip():
            raise ValueError("id must be a nonempty string without surrounding whitespace")
        if sample_id in indexed:
            raise ValueError(f"duplicate id: {sample_id}")
        indexed[sample_id] = row
    if not indexed:
        raise ValueError("records must be nonempty")
    return indexed


def _column(rows, field):
    try:
        return [row[field] for row in rows]
    except KeyError as exc:
        raise ValueError(f"missing required column: {field}") from exc


def prediction_rows(ids, output: AdaptedOutput) -> list[dict]:
    ids = list(ids)
    if len(ids) != len(output.intensity):
        raise ValueError("ids and predictions must have the same length")
    rows = [{"id": sample_id, "intensity": float(final),
             "polarity": Polarity.from_value(int(label)).value, "raw_intensity": float(raw)}
            for sample_id, final, label, raw in zip(ids, output.intensity, output.polarity, output.raw_intensity)]
    _indexed(rows)
    return rows


def score_records(truth_rows, prediction_rows) -> dict:
    truth, predicted = _indexed(truth_rows), _indexed(prediction_rows)
    if truth.keys() != predicted.keys():
        raise ValueError(f"ID coverage mismatch: missing={sorted(truth.keys()-predicted.keys())}, "
                         f"extra={sorted(predicted.keys()-truth.keys())}")
    actual_rows = list(truth.values())
    final_rows = [predicted[sample_id] for sample_id in truth]
    has_classes = any("polarity" in row for row in actual_rows)
    return score_predictions(
        _column(actual_rows, "intensity"), _column(final_rows, "intensity"),
        _column(final_rows, "polarity"),
        truth_polarity=_column(actual_rows, "polarity") if has_classes else None,
    )


def write_predictions_csv(path, rows) -> Path:
    """Validate final values and write round-trip precision; never overwrite an artifact."""
    rows = list(rows)
    _indexed(rows)
    final_values = intensities(_column(rows, "intensity"))
    labels = classes(_column(rows, "polarity"))
    fields = ["id", "intensity", "polarity"]
    if any("raw_intensity" in row for row in rows):
        vector(_column(rows, "raw_intensity"), "raw_intensity")
        fields.append("raw_intensity")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, label, final in zip(rows, labels, final_values):
            record = {field: row[field] for field in fields}
            # NumPy float32.__str__ uses shortened text; promote before serialization
            # to preserve exactly the value that the float64 scorer receives.
            record["intensity"] = float(final)
            if "raw_intensity" in record:
                record["raw_intensity"] = float(record["raw_intensity"])
            record["polarity"] = Polarity.from_value(int(label)).value
            writer.writerow(record)
    return target


def read_records_csv(path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not {"id", "intensity"} <= set(fields):
            raise ValueError("CSV requires unique columns including id and intensity")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("CSV row width does not match header")
    _indexed(rows)
    return rows
