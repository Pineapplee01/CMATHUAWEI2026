from e_emotion.problem2_fair.artifacts import ArtifactStore
from e_emotion.problem2_fair.registry import load_registry


def test_method_registry_is_the_source_of_method_directory_and_adapter():
    registry = load_registry()
    record = registry.get("aumdf")

    assert record.directory == "AUMDF"
    assert record.adapter == "e_emotion.baselines.aumdf.fair_adapter:create_adapter"
    assert record.is_runnable
    assert registry.get("cica").state == "paper_only"


def test_artifact_store_uses_method_directory_view_and_fixed_seed(tmp_path):
    store = ArtifactStore(tmp_path, registry=load_registry())

    assert store.run_dir("concat_mlp", "aligned_po", 1) == tmp_path / "Concat-MLP" / "aligned_po" / "seed-1"
    assert store.log_path("concat_mlp", "aligned_po", 1) == tmp_path / "Concat-MLP" / "aligned_po" / "seed-1.log"
    assert store.report_path("concat_mlp", "aligned_po") == tmp_path / "Concat-MLP" / "aligned_po" / "three-seed-report.json"
