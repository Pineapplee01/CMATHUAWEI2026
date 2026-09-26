"""Fair Problem 2 baseline contracts and execution modules."""

from e_emotion.problem2_fair.core import (
    FinalPrediction,
    Problem2Split,
    Q2_V2_CONDITIONS,
    finalize_prediction,
    project_intensity,
    validate_model_input_view,
)
from e_emotion.problem2_fair.q2 import (
    Q2_V2_PROTOCOL_VERSION,
    build_q2_v2_manifest,
    ensure_q2_v2_manifest,
    materialize_q2_condition,
    pool_unaligned_to_text_slots,
    validate_q2_v2_manifest,
)
from e_emotion.problem2_fair.views import Problem2Dataset, load_problem2_dataset
from e_emotion.problem2_fair.run import BaselineMethodAdapter, BaselineRun, RawMethodPrediction
from e_emotion.problem2_fair.catalog import MethodSpec, active_method_specs, all_method_specs, is_frozen_output_path, method_spec
from e_emotion.problem2_fair.artifacts import ArtifactStore
from e_emotion.problem2_fair.registry import MethodRecord, MethodRegistry, load_registry
from e_emotion.problem2_fair.result_index import result_index_path, update_result_index
from e_emotion.problem2_fair.test252 import TEST252_PROTOCOL_VERSION, evaluate_test252
from e_emotion.problem2_fair.report import summarize_runs
from e_emotion.problem2_fair.view_report import summarize_view
from e_emotion.problem2_fair.config import FAIR_SEEDS, default_manifest_path, default_view_root

__all__ = [
    "FinalPrediction",
    "BaselineMethodAdapter",
    "BaselineRun",
    "ArtifactStore",
    "FAIR_SEEDS",
    "MethodSpec",
    "MethodRecord",
    "MethodRegistry",
    "Problem2Split",
    "RawMethodPrediction",
    "active_method_specs",
    "all_method_specs",
    "Problem2Dataset",
    "Q2_V2_CONDITIONS",
    "Q2_V2_PROTOCOL_VERSION",
    "build_q2_v2_manifest",
    "default_manifest_path",
    "default_view_root",
    "ensure_q2_v2_manifest",
    "finalize_prediction",
    "materialize_q2_condition",
    "method_spec",
    "is_frozen_output_path",
    "project_intensity",
    "validate_q2_v2_manifest",
    "validate_model_input_view",
    "summarize_runs",
    "summarize_view",
    "load_problem2_dataset",
    "load_registry",
    "result_index_path",
    "update_result_index",
    "TEST252_PROTOCOL_VERSION",
    "evaluate_test252",
]
