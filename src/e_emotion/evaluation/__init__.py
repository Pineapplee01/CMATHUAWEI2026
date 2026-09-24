"""Evaluation metrics."""

from e_emotion.evaluation.metrics import accuracy, f1_macro, mae, pearson
from e_emotion.evaluation.outputs import adapt_output, select_neutral_threshold
from e_emotion.evaluation.protocol_guard import (
    build_downstream_manifest,
    build_protocol_manifest,
    build_run_manifest,
    validate_npz_dataset,
    validate_processed_dataset,
    validate_downstream_manifest,
    validate_protocol_manifest,
)
from e_emotion.evaluation.records import prediction_rows, read_records_csv, score_records, write_predictions_csv
from e_emotion.evaluation.scoring import score_predictions

__all__ = ["accuracy", "f1_macro", "mae", "pearson", "adapt_output", "select_neutral_threshold",
           "prediction_rows", "read_records_csv", "score_records", "write_predictions_csv", "score_predictions",
           "build_downstream_manifest", "build_protocol_manifest", "validate_downstream_manifest",
           "validate_protocol_manifest", "build_run_manifest", "validate_processed_dataset",
           "validate_npz_dataset"]
