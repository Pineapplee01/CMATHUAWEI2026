#!/usr/bin/env python3
"""Compare a legacy and mask-aware baseline run under the same Q2 grid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean


METRICS = ("accuracy", "macro_f1", "weighted_f1", "mae", "pearson")
EXPECTED_MASK_SHA256 = "bddfc02985e9528bcab58a252ebb9beaa8c6a9812b609b6b88c7a547fd623916"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def normalize_metrics(values: dict) -> dict[str, float]:
    result = dict(values)
    if "accuracy" not in result and "accuracy_3" in result:
        result["accuracy"] = result["accuracy_3"]
    missing = [name for name in METRICS if name not in result]
    if missing:
        raise ValueError(f"metrics are missing {missing}")
    return {name: float(result[name]) for name in METRICS}


def extract_test_metrics(payload: dict) -> dict[str, float]:
    if isinstance(payload.get("test"), dict):
        return normalize_metrics(payload["test"])
    if isinstance(payload.get("test_metrics"), dict):
        return normalize_metrics(payload["test_metrics"])
    raise ValueError("training metrics do not contain a clean test result")


def condition_key(row: dict) -> tuple[str, str, float]:
    return (str(row["combination"]), str(row["position"]), float(row["requested_fraction"]))


def checked_conditions(payload: dict) -> dict[tuple[str, str, float], dict]:
    if payload.get("mask_sha256") != EXPECTED_MASK_SHA256:
        raise ValueError("Q2 mask hash does not match the shared test manifest")
    rows = payload.get("conditions")
    if not isinstance(rows, list) or len(rows) != 63:
        raise ValueError("Q2 result must contain exactly 63 conditions")
    table = {condition_key(row): row for row in rows}
    if len(table) != 63:
        raise ValueError("Q2 result contains duplicate conditions")
    for row in table.values():
        if row.get("status") != "ok" or int(row.get("n", 0)) != 727:
            raise ValueError(f"invalid Q2 condition: {condition_key(row)}")
        normalize_metrics(row)
    return table


def average(rows: list[dict]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot average an empty condition set")
    return {name: fmean(float(row[name]) for row in rows) for name in METRICS}


def delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {name: after[name] - before[name] for name in METRICS}


def comparison(before_rows: dict, after_rows: dict) -> dict:
    if set(before_rows) != set(after_rows):
        raise ValueError("legacy and mask-aware Q2 grids differ")
    rows_before = list(before_rows.values())
    rows_after = list(after_rows.values())
    complete_before = [row for row in rows_before if row["combination"] == "complete"]
    complete_after = [row for row in rows_after if row["combination"] == "complete"]
    incomplete_before = [row for row in rows_before if row["combination"] != "complete"]
    incomplete_after = [row for row in rows_after if row["combination"] != "complete"]
    if len(complete_before) != 9 or len(incomplete_before) != 54:
        raise ValueError("unexpected Q2 complete/incomplete condition count")

    def summarized(old: list[dict], new: list[dict]) -> dict:
        old_mean, new_mean = average(old), average(new)
        return {"legacy": old_mean, "mask_aware": new_mean, "delta_mask_aware_minus_legacy": delta(old_mean, new_mean)}

    fractions = {}
    for fraction in (0.1, 0.3, 0.5):
        old = [row for row in incomplete_before if float(row["requested_fraction"]) == fraction]
        new = [row for row in incomplete_after if float(row["requested_fraction"]) == fraction]
        fractions[str(fraction)] = summarized(old, new)
    combinations = {}
    for combination in ("T", "A", "V", "TA", "TV", "AV"):
        old = [row for row in incomplete_before if row["combination"] == combination]
        new = [row for row in incomplete_after if row["combination"] == combination]
        combinations[combination] = summarized(old, new)

    worst_key = min(
        incomplete_after,
        key=lambda row: (float(row["macro_f1"]), -float(row["mae"])),
    )
    key = condition_key(worst_key)
    return {
        "shared_mask_sha256": EXPECTED_MASK_SHA256,
        "condition_count": {"complete": 9, "incomplete": 54, "total": 63},
        "clean_q2": summarized(complete_before, complete_after),
        "incomplete_q2_mean": summarized(incomplete_before, incomplete_after),
        "by_requested_missing_fraction": fractions,
        "by_missing_modality_combination": combinations,
        "worst_mask_aware_condition": {
            "combination": key[0],
            "position": key[1],
            "requested_fraction": key[2],
            "legacy": normalize_metrics(before_rows[key]),
            "mask_aware": normalize_metrics(after_rows[key]),
            "delta_mask_aware_minus_legacy": delta(
                normalize_metrics(before_rows[key]), normalize_metrics(after_rows[key])
            ),
        },
    }


def markdown(payload: dict) -> str:
    def table(title: str, values: dict) -> list[str]:
        lines = [f"## {title}", "", "| Metric | Legacy | Mask-aware | Delta |", "| --- | ---: | ---: | ---: |"]
        for name in METRICS:
            lines.append(
                f"| {name} | {values['legacy'][name]:.6f} | {values['mask_aware'][name]:.6f} | "
                f"{values['delta_mask_aware_minus_legacy'][name]:+.6f} |"
            )
        return lines + [""]

    lines = ["# Position-Mask Repair Comparison", ""]
    lines += table("Clean Test", payload["clean_test"])
    lines += table("Q2 Mean Across 54 Missing Conditions", payload["q2"]["incomplete_q2_mean"])
    lines += ["## Q2 By Requested Missing Fraction", "", "| Fraction | Legacy Macro-F1 | Mask-aware Macro-F1 | Delta | Legacy MAE | Mask-aware MAE | Delta |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for fraction, values in payload["q2"]["by_requested_missing_fraction"].items():
        old, new, changes = values["legacy"], values["mask_aware"], values["delta_mask_aware_minus_legacy"]
        lines.append(f"| {fraction} | {old['macro_f1']:.6f} | {new['macro_f1']:.6f} | {changes['macro_f1']:+.6f} | {old['mae']:.6f} | {new['mae']:.6f} | {changes['mae']:+.6f} |")
    lines += ["", "## Interpretation Guardrail", "", "This is one rerun with the same nominal seed, not a multi-seed significance test. Any change combines training variability with the intended behavior change: native-invalid and synthetic-missing positions no longer participate in attention, convolution leakage, or pooling.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--legacy-train", type=Path, required=True)
    parser.add_argument("--mask-aware-train", type=Path, required=True)
    parser.add_argument("--legacy-q2", type=Path, required=True)
    parser.add_argument("--mask-aware-q2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    legacy_train = extract_test_metrics(load_json(args.legacy_train))
    mask_aware_train = extract_test_metrics(load_json(args.mask_aware_train))
    legacy_q2 = checked_conditions(load_json(args.legacy_q2))
    mask_aware_q2 = checked_conditions(load_json(args.mask_aware_q2))
    payload = {
        "method": args.method,
        "comparison_version": "mask-aware-v1",
        "clean_test": {
            "legacy": legacy_train,
            "mask_aware": mask_aware_train,
            "delta_mask_aware_minus_legacy": delta(legacy_train, mask_aware_train),
        },
        "q2": comparison(legacy_q2, mask_aware_q2),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
