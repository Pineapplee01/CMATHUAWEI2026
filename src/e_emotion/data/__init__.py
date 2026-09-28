"""Public data loading and validation APIs."""

from e_emotion.data.paths import DataPathPolicy
from e_emotion.data.processed import (
    FEATURE_VERSION,
    MODALITIES,
    PROTOCOL_VERSION,
    ProcessedDataset,
    ProcessedSplit,
    build_manifest,
    build_processed_manifest,
    contiguous_missing_mask,
    load_processed_dataset,
    load_processed_split,
    native_coordinate_mask,
    sha256_file,
)
from e_emotion.data.repositories import (
    Attachment2Repository,
    ExplainabilityRepository,
    MissingModalityRepository,
)
from e_emotion.data.validation import (
    ValidationReport,
    inspect_data_root,
    validate_attachment2_payload,
    validate_data_root,
)

__all__ = [
    "Attachment2Repository",
    "DataPathPolicy",
    "MODALITIES",
    "PROTOCOL_VERSION",
    "FEATURE_VERSION",
    "ProcessedDataset",
    "ProcessedSplit",
    "ExplainabilityRepository",
    "MissingModalityRepository",
    "ValidationReport",
    "inspect_data_root",
    "validate_attachment2_payload",
    "validate_data_root",
    "contiguous_missing_mask",
    "native_coordinate_mask",
    "sha256_file",
    "build_processed_manifest",
    "build_manifest",
    "load_processed_dataset",
    "load_processed_split",
]
