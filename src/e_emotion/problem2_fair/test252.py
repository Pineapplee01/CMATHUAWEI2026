"""Aligned-test 252-scenario local-missingness protocol."""

from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from e_emotion.evaluation import score_predictions, write_predictions_csv
from e_emotion.problem2_fair.core import MODALITIES, Problem2Split
from e_emotion.problem2_fair.run import BaselineRun
from e_emotion.problem2_fair.views import Problem2Dataset


TEST252_PROTOCOL_VERSION = "aligned-test-local-missingness-252-v1"
TEST252_COMBINATIONS = ("T", "A", "V", "TA", "TV", "AV", "TAV")
TEST252_RATES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7)
TEST252_MASK_SEEDS = (2026, 2027, 2028)
_CODES = {"T": "text", "A": "audio", "V": "vision"}


def scenarios() -> tuple[tuple[str, float, str, int], ...]:
    """Match problem2_retrain_v2: 7 combinations x 6 rates x 6 conditions."""
    rows = []
    for combination in TEST252_COMBINATIONS:
        for rate in TEST252_RATES:
            for position in ("start", "middle", "end"):
                rows.append((combination, rate, position, TEST252_MASK_SEEDS[0]))
            rows.extend((combination, rate, "random", seed) for seed in TEST252_MASK_SEEDS)
    return tuple(rows)


def scenario_key(combination: str, rate: float, position: str, seed: int) -> str:
    return f"{combination}_{rate:.1f}_{position}_{seed}"


def _interval(physical: np.ndarray, observed: np.ndarray, *, rate: float, position: str, seed: int, sample: int, modality: int) -> tuple[int, int]:
    slots = np.flatnonzero(physical)
    before = int(observed.sum())
    if not len(slots) or rate == 0 or before < 2:
        return 0, 0
    start_bound, end_bound = int(slots[0]), int(slots[-1]) + 1
    width = min(end_bound - start_bound, max(1, round(rate * (end_bound - start_bound))))
    if position == "random":
        rng = np.random.default_rng(np.random.SeedSequence([seed, sample, modality]))
        start = int(rng.integers(start_bound, end_bound - width + 1))
    elif position == "start":
        start = start_bound
    elif position == "middle":
        start = start_bound + (end_bound - start_bound - width) // 2
    elif position == "end":
        start = end_bound - width
    else:
        raise ValueError(f"unknown test252 position: {position}")
    end = start + width
    while end > start and int(observed[start:end].sum()) == before:
        if position == "end":
            start += 1
        else:
            end -= 1
    return start, end


def materialize_scenario(
    split: Problem2Split, combination: str, rate: float, position: str, seed: int
) -> tuple[Problem2Split, list[dict[str, object]]]:
    """Delete continuous intervals independently on each selected modality axis."""
    if split.view != "aligned_po":
        raise ValueError("test252 protocol requires the aligned_po test split")
    if combination not in TEST252_COMBINATIONS or rate not in TEST252_RATES:
        raise ValueError("test252 scenario is outside the canonical grid")
    selected = tuple(_CODES[code] for code in combination)
    observed = {name: np.array(mask, copy=True) for name, mask in split.observed.items()}
    values = {name: np.array(value, copy=True) for name, value in split.values.items()}
    records: list[dict[str, object]] = []
    for row, sample_id in enumerate(split.ids):
        for code, name in _CODES.items():
            if name not in selected:
                continue
            modality = "TAV".index(code)
            source = split.observed[name][row]
            start, end = _interval(
                split.physical_support[name][row],
                source,
                rate=rate,
                position=position,
                seed=seed,
                sample=row,
                modality=modality,
            )
            observed[name][row, start:end] = False
            before = int(source.sum())
            removed = int(source[start:end].sum())
            records.append(
                {
                    "sample_id": str(sample_id),
                    "modality": code,
                    "start_slot": start,
                    "end_slot_exclusive": end,
                    "interval_slots": end - start,
                    "removed_observed_slots": removed,
                    "observed_before": before,
                    "observed_after": before - removed,
                    "actual_removed_fraction": removed / max(before, 1),
                    "too_few_observations": bool(before < 2),
                }
            )
    for name in MODALITIES:
        values[name][~observed[name]] = 0.0
    tokens = np.array(split.input_ids, copy=True)
    tokens[~observed["text"]] = 0
    return replace(split, values=values, observed=observed, input_ids=tokens), records


def _mean_rows(rows: Iterable[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    metrics = ("accuracy", "macro_f1", "weighted_f1", "mae", "pearson", "actual_removed_fraction")
    result = []
    for group, values in sorted(groups.items()):
        item = dict(zip(keys, group))
        for metric in metrics:
            finite = [float(value[metric]) for value in values if value.get(metric) is not None]
            item[metric] = float(np.mean(finite)) if finite else None
        result.append(item)
    return result


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_test252(adapter, dataset: Problem2Dataset, checkpoint: str | Path, output: str | Path) -> dict:
    """Evaluate one frozen aligned checkpoint with the referenced 252 mask grid."""
    if dataset.view != "aligned_po":
        raise ValueError("test252 evaluation requires an aligned_po dataset")
    root = Path(output)
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    test = dataset["test"]
    runner = BaselineRun(adapter)
    summary_rows: list[dict[str, object]] = []
    scenario_root = root / "scenarios"
    for combination, rate, position, seed in scenarios():
        key = scenario_key(combination, rate, position, seed)
        partial, records = materialize_scenario(test, combination, rate, position, seed)
        final, source = runner._predict(partial, Path(checkpoint))
        metrics = score_predictions(
            partial.regression,
            final.intensity,
            final.polarity,
            truth_polarity=partial.classification,
        )
        prediction_rows = [
            {
                "id": str(sample_id),
                "raw_intensity": float(raw),
                "intensity": float(intensity),
                "polarity": polarity.value,
            }
            for sample_id, raw, intensity, polarity in zip(
                partial.ids, final.raw_intensity, final.intensity, final.polarity, strict=True
            )
        ]
        write_predictions_csv(scenario_root / "predictions" / f"{key}.csv", prediction_rows)
        _write_rows(scenario_root / "masks" / f"{key}.csv", records)
        summary_rows.append(
            {
                "combination": combination,
                "rate": rate,
                "position": position,
                "mask_seed": seed,
                "actual_removed_fraction": float(np.mean([row["actual_removed_fraction"] for row in records])),
                "decision_source": source,
                **{key: metrics[key] for key in ("accuracy", "macro_f1", "weighted_f1", "mae", "pearson")},
            }
        )
    _write_rows(root / "local_scenarios.csv", summary_rows)
    _write_rows(root / "position_effects.csv", _mean_rows(summary_rows, ("combination", "rate", "position")))
    duration = _mean_rows(_mean_rows(summary_rows, ("combination", "rate", "position")), ("combination", "rate"))
    _write_rows(root / "duration_effects.csv", duration)
    _write_rows(root / "overall_by_rate.csv", _mean_rows(duration, ("rate",)))
    protocol = {
        "protocol_version": TEST252_PROTOCOL_VERSION,
        "scenarios_expected": 252,
        "samples_per_scenario": test.size,
        "split": "test",
        "rates": list(TEST252_RATES),
        "mask_seeds": list(TEST252_MASK_SEEDS),
        "positions": ["start", "middle", "end", "random"],
        "joint_interval_policy": "per-modality physical axes; random starts sampled independently",
        "aggregation": "mean random repeats first; equal weight for four positions and seven modality combinations",
        "selection": "frozen checkpoint; no epoch selection from scenario scores",
    }
    (root / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return protocol | {"scenarios": len(summary_rows), "output": str(root)}


__all__ = [
    "TEST252_COMBINATIONS",
    "TEST252_MASK_SEEDS",
    "TEST252_PROTOCOL_VERSION",
    "TEST252_RATES",
    "evaluate_test252",
    "materialize_scenario",
    "scenario_key",
    "scenarios",
]
