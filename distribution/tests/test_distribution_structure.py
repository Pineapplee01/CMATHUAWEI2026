from __future__ import annotations

import ast
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_package_import_is_lightweight() -> None:
    import multimodal_emotion

    assert multimodal_emotion.__version__ == "0.1.0"


def test_project_metadata_and_source_package_use_the_canonical_name() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    source_packages = {
        path.name
        for path in (ROOT / "src").iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    }

    assert data["project"]["name"] == "multimodal-emotion"
    assert source_packages == {"multimodal_emotion"}


def test_no_console_script_interface_is_declared() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "scripts" not in data.get("project", {})


def test_external_asset_directories_are_placeholders_only() -> None:
    reference_root = ROOT / "reference"
    files = sorted(path.relative_to(reference_root).as_posix() for path in reference_root.rglob("*") if path.is_file())

    assert files == [
        "checkpoints/README.md",
        "models/README.md",
        "tools/README.md",
    ]


def test_contest_data_uses_a_valid_local_or_github_layout() -> None:
    data_root = ROOT / "data"
    attachment_directories = {path.name for path in data_root.iterdir() if path.is_dir()}

    if not attachment_directories:
        assert [path.name for path in data_root.iterdir() if path.is_file()] == ["README.md"]
        assert "## GitHub Publication" in (data_root / "README.md").read_text(encoding="utf-8")
        return

    assert attachment_directories == {
        "appendix_1",
        "appendix_2",
        "appendix_3",
        "appendix_4",
    }
    assert (data_root / "appendix_1" / "mosei_raw_videos_100" / "label-100.xlsx").is_file()
    assert (data_root / "appendix_2" / "aligned_50.pkl").is_file()
    assert (data_root / "appendix_2" / "unaligned_50.pkl").is_file()
    assert (data_root / "appendix_3" / "aligned").is_dir()
    assert (data_root / "appendix_3" / "unaligned").is_dir()
    assert (data_root / "appendix_4" / "aligned").is_dir()
    assert (data_root / "appendix_4" / "unaligned").is_dir()


def test_results_directory_starts_with_documentation_only() -> None:
    result_files = [path.name for path in (ROOT / "results").iterdir() if path.is_file()]

    assert result_files == ["README.md"]


def test_path_mapping_documents_legacy_submission_paths() -> None:
    text = (ROOT / "docs" / "path_mapping.md").read_text(encoding="utf-8")

    assert "AAA提交版代码及结果/问题二/代码/aligned.py" in text
    assert "src/multimodal_emotion/prediction/problem2_aligned.py" in text
    assert "preprocess/problem2/preprocess_aligned.py" in text
    assert "src/multimodal_emotion/preprocess/appendix2_aligned.py" in text


def test_distribution_sources_do_not_import_legacy_q_utils_by_bare_name() -> None:
    source_root = ROOT / "src" / "multimodal_emotion"
    violations: list[str] = []
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in {"q2_utils", "q3_utils"}:
                violations.append(path.relative_to(ROOT).as_posix())

    assert violations == []


def test_distribution_sources_do_not_mutate_python_import_path() -> None:
    source_root = ROOT / "src" / "multimodal_emotion"
    violations: list[str] = []
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "path"
                and isinstance(node.value, ast.Name)
                and node.value.id == "sys"
            ):
                violations.append(path.relative_to(ROOT).as_posix())

    assert violations == []
