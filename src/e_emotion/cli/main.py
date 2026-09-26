"""Command-line entry points for data inspection and validation."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Sequence

from e_emotion.config import ConfigError, load_config
from e_emotion.data import (
    inspect_data_root,
    inspect_processed_root,
    validate_data_root,
    validate_processed_root,
)
from e_emotion.data.paths import DataPathPolicy
from e_emotion.evaluation import read_records_csv, score_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="e_emotion", description="2026华为杯 E题 project tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("inspect-data", "inspect the configured processed NPZ root and optional special-test mirror"),
        ("validate-data", "validate processed Attachment 2 NPZs and optional special-test files"),
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
    baseline = subparsers.add_parser("baseline", help="run or audit the Problem 2 fair-baseline protocol")
    baseline.add_argument("--config", default="configs/base.yaml", type=Path)
    baseline_commands = baseline.add_subparsers(dest="baseline_command", required=True)
    validate = baseline_commands.add_parser("validate", help="validate one processed_po view and render Q2-v2 manifest")
    validate.add_argument("--view", choices=("aligned_po", "unaligned_po"), required=True)
    validate.add_argument("--data-root", type=Path, help="override the protocol-approved view root")
    baseline_run = baseline_commands.add_parser("run", help="run one registered method")
    baseline_run.add_argument("--method", required=True)
    baseline_run.add_argument("--view", choices=("aligned_po", "unaligned_po"), required=True)
    baseline_run.add_argument("--data-root", type=Path, help="override the protocol-approved view root")
    baseline_run.add_argument("--seed", default=1, type=int)
    baseline_run.add_argument("--run-dir", type=Path, help="must equal the canonical ArtifactStore run directory")
    baseline_run.add_argument("--artifact-root", type=Path, help="override the Baseline artifact root")
    verify = baseline_commands.add_parser("verify", help="validate a completed fair-baseline run")
    verify.add_argument("--run-dir", required=True, type=Path)
    test252 = baseline_commands.add_parser("test252", help="evaluate one verified aligned run on the 252-scenario protocol")
    test252.add_argument("--method", required=True)
    test252.add_argument("--seed", required=True, type=int)
    test252.add_argument("--data-root", type=Path, help="override the approved aligned_po root")
    test252.add_argument("--artifact-root", type=Path, help="override the Baseline artifact root")
    report = baseline_commands.add_parser("report", help="aggregate matching five-seed fair-baseline runs")
    report.add_argument("--run-dir", type=Path, action="append")
    report.add_argument("--view", choices=("aligned_po", "unaligned_po"))
    report.add_argument("--method", action="append", help="limit a view report to selected first-wave methods")
    report.add_argument("--artifact-root", type=Path, help="override the Baseline artifact root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "baseline":
        from e_emotion.problem2_fair import (
            BaselineRun,
            ArtifactStore,
            active_method_specs,
            build_q2_v2_manifest,
            default_view_root,
            load_problem2_dataset,
            method_spec,
            summarize_runs,
            summarize_view,
        )
        from e_emotion.problem2_fair.verification import verify_run
        from e_emotion.problem2_fair.test252 import evaluate_test252

        if args.baseline_command == "validate":
            try:
                dataset = load_problem2_dataset(args.data_root or default_view_root(args.view), view=args.view)
                print(json.dumps(build_q2_v2_manifest(dataset), ensure_ascii=False, indent=2, allow_nan=False))
                return 0
            except (OSError, ValueError) as exc:
                print(f"baseline validation error: {exc}")
                return 2
        if args.baseline_command == "run":
            try:
                spec = method_spec(args.method)
            except KeyError as exc:
                print(f"baseline run error: {exc}")
                return 2
            if spec.status == "paper_only":
                print(f"baseline run error: {args.method} is paper_only and has no adapter")
                return 2
            if spec.status != "ready":
                print(f"baseline run error: {args.method} is not ready ({spec.status})")
                return 2
            try:
                if spec.adapter is None:
                    raise ValueError("registered method has no adapter factory")
                module_name, attribute = spec.adapter.split(":", 1)
                factory = getattr(importlib.import_module(module_name), attribute)
                adapter = factory()
                if adapter.method_id != args.method:
                    raise ValueError("adapter method_id does not match --method")
                if args.method not in {item.method_id for item in active_method_specs()}:
                    raise ValueError("method is not active in the Baseline Workspace")
                store = ArtifactStore(args.artifact_root)
                target = store.run_dir(args.method, args.view, args.seed)
                if args.run_dir is not None and args.run_dir.resolve() != target:
                    raise ValueError("run-dir must equal the canonical ArtifactStore run directory")
                dataset = load_problem2_dataset(args.data_root or default_view_root(args.view), view=args.view)
                report = BaselineRun(adapter).execute(
                    dataset,
                    seed=args.seed,
                    run_dir=target,
                    manifest_path=store.root / "_manifests" / f"{args.view}_q2_v2.json",
                )
                print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
                return 0
            except (ImportError, AttributeError, OSError, ValueError) as exc:
                print(f"baseline run error: {exc}")
                return 2
        if args.baseline_command == "verify":
            try:
                print(json.dumps(verify_run(args.run_dir), ensure_ascii=False))
                return 0
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"baseline verify error: {exc}")
                return 2
        if args.baseline_command == "test252":
            try:
                spec = method_spec(args.method)
                if not spec.is_runnable:
                    raise ValueError(f"{args.method} is not runnable ({spec.status})")
                if "aligned_po" not in spec.views or spec.adapter is None:
                    raise ValueError(f"{args.method} cannot run the aligned test252 protocol")
                store = ArtifactStore(args.artifact_root)
                run_dir = store.run_dir(args.method, "aligned_po", args.seed)
                verify_run(run_dir, require_source=True)
                metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
                module_name, attribute = spec.adapter.split(":", 1)
                adapter = getattr(importlib.import_module(module_name), attribute)()
                dataset = load_problem2_dataset(args.data_root or default_view_root("aligned_po"), view="aligned_po")
                result = evaluate_test252(adapter, dataset, metrics["checkpoint"], run_dir / "test252")
                print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
                return 0
            except (ImportError, AttributeError, OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"baseline test252 error: {exc}")
                return 2
        if args.baseline_command == "report":
            try:
                if args.view and args.run_dir:
                    raise ValueError("choose either --view or --run-dir")
                if args.view:
                    options = {"artifact_root": args.artifact_root}
                    if args.method:
                        options["methods"] = args.method
                    result = summarize_view(args.view, verify=True, **options)
                elif args.run_dir:
                    for run_dir in args.run_dir:
                        verify_run(run_dir, require_source=True)
                    result = summarize_runs(args.run_dir)
                else:
                    raise ValueError("report requires --view or at least one --run-dir")
                print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
                return 0
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"baseline report error: {exc}")
                return 2
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
        report = inspect_processed_root(config.processed_root)
        report["configured_split_files"] = dict(config.split_files)
        if config.data_root.is_dir():
            report["raw_data_root"] = inspect_data_root(config.data_root)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "validate-data":
        expected_sizes = {"train": 3395, "valid": 728, "test": 727} if config.feature_version == "aligned_50" else None
        report = validate_processed_root(config.processed_root, expected_split_sizes=expected_sizes)
        if config.data_root.is_dir():
            report["raw_data_root"] = validate_data_root(config.data_root)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        errors = list(report.get("errors", []))
        raw_report = report.get("raw_data_root", {})
        errors.extend(
            error
            for value in raw_report.get("reports", {}).values()
            if isinstance(value, dict)
            for error in value.get("errors", [])
        )
        return 1 if errors else 0
    print(f"task entry point reserved for {args.task}; no model implementation is included yet")
    return 0
