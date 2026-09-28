from __future__ import annotations

import ast
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_package_import_is_lightweight() -> None:
    import cpmcm_huawei2026

    assert cpmcm_huawei2026.__version__ == "0.1.0"


def test_no_console_script_interface_is_declared() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "scripts" not in data.get("project", {})


def test_external_asset_directories_are_placeholders_only() -> None:
    asset_root = ROOT / "external_assets"
    files = sorted(path.relative_to(asset_root).as_posix() for path in asset_root.rglob("*") if path.is_file())

    assert files == [
        "AAAcheckpoints/README.md",
        "AAAdata/README.md",
        "AAAmodel/README.md",
    ]


def test_path_mapping_documents_legacy_submission_paths() -> None:
    text = (ROOT / "docs" / "path_mapping.md").read_text(encoding="utf-8")

    assert "AAA提交版代码及结果/问题二/代码/aligned.py" in text
    assert "src/cpmcm_huawei2026/prediction/problem2_aligned.py" in text
    assert "preprocess/problem2/preprocess_aligned.py" in text
    assert "src/cpmcm_huawei2026/preprocessing/appendix2_aligned.py" in text


def test_distribution_sources_do_not_import_legacy_q_utils_by_bare_name() -> None:
    source_root = ROOT / "src" / "cpmcm_huawei2026"
    violations: list[str] = []
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in {"q2_utils", "q3_utils"}:
                violations.append(path.relative_to(ROOT).as_posix())

    assert violations == []


def test_distribution_sources_do_not_mutate_python_import_path() -> None:
    source_root = ROOT / "src" / "cpmcm_huawei2026"
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
