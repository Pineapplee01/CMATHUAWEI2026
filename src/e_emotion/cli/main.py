"""Command-line entry points for data inspection and validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from e_emotion.config import ConfigError, load_config
from e_emotion.data import inspect_data_root, validate_data_root
from e_emotion.data.paths import DataPathPolicy
from e_emotion.evaluation import read_records_csv, score_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="e_emotion", description="2026华为杯 E题 project tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("inspect-data", "inspect the local data mirror without loading feature tensors"),
        ("validate-data", "validate Attachment 2 and special-test file contracts"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--config", default="configs/base.yaml", type=Path)
    run = subparsers.add_parser("run", help="reserved task-domain execution entry point")
    run.add_argument("task", choices=("alignment", "robustness", "explainability"))
    run.add_argument("--config", default="configs/base.yaml", type=Path)
    score = subparsers.add_parser("score", help="score final prediction CSV without modifying predictions")
    score.add_argument("--config", default="configs/base.yaml", type=Path)
    score.add_argument("--truth", required=True, type=Path, help="label CSV inside project data/")
    score.add_argument("--predictions", required=True, type=Path)
    score.add_argument("--output", type=Path, help="optional new JSON report inside artifacts/")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"configuration error: {exc}")
        return 2
    if args.command == "score":
        try:
            labels = DataPathPolicy(config.project_root / "data").require_file(
                config.project_root / args.truth)
            report = score_records(read_records_csv(labels),
                                   read_records_csv(config.project_root / args.predictions))
            rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
            if args.output:
                output = DataPathPolicy(config.output_root).resolve(config.project_root / args.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("x", encoding="utf-8") as handle:
                    handle.write(rendered)
            print(rendered)
            return 0
        except (ValueError, OSError) as exc:
            print(f"scoring error: {exc}")
            return 2
    if args.command == "inspect-data":
        print(json.dumps(inspect_data_root(config.data_root), ensure_ascii=False, indent=2))
        return 0
    if args.command == "validate-data":
        report = validate_data_root(config.data_root)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        errors = [
            error
            for value in report["reports"].values()
            if isinstance(value, dict)
            for error in value.get("errors", [])
        ]
        return 1 if errors else 0
    print(f"task entry point reserved for {args.task}; no model implementation is included yet")
    return 0
