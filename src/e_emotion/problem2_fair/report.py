"""Fixed-seed reporting for completed Problem 2 fair-baseline runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from e_emotion.problem2_fair.config import FAIR_SEEDS
from e_emotion.problem2_fair.core import Q2_V2_CONDITIONS, validate_model_input_view
from e_emotion.problem2_fair.run import PROTOCOL_VERSION


_METRICS = ("accuracy", "macro_f1", "weighted_f1", "mae", "pearson")


def _summary(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {"mean": float(data.mean()), "std": float(data.std(ddof=0))}


def summarize_runs(run_dirs: Iterable[str | Path]) -> dict:
    """Aggregate complete runs only when their fair-comparison identities match."""
    roots = [Path(path).resolve() for path in run_dirs]
    if not roots:
        raise ValueError("at least one run directory is required")
    summaries = [json.loads((root / "metrics.json").read_text(encoding="utf-8")) for root in roots]
    identity = {
        (item.get("protocol_version"), item.get("method"), item.get("view"), item.get("model_input_view"))
        for item in summaries
    }
    if len(identity) != 1 or next(iter(identity))[0] != PROTOCOL_VERSION:
        raise ValueError("runs must share one method, view and fair-comparison protocol")
    seeds = [item.get("seed") for item in summaries]
    if len(set(seeds)) != len(seeds):
        raise ValueError("runs must use distinct seeds")
    if tuple(sorted(seeds)) != FAIR_SEEDS:
        raise ValueError(f"runs must contain exactly the fair seed set {FAIR_SEEDS}")
    _, method, view, model_input_view = next(iter(identity))
    validate_model_input_view(view, model_input_view)
    clean = {
        metric: _summary([float(item["clean"][metric]) for item in summaries if item["clean"][metric] is not None])
        for metric in _METRICS
        if all(item["clean"].get(metric) is not None for item in summaries)
    }
    condition_values: dict[tuple[str, str | None, float], dict[str, list[float]]] = {}
    expected_mask_hash: str | None = None
    expected_conditions: set[tuple[str, str | None, float]] | None = None
    for root, summary in zip(roots, summaries, strict=True):
        q2 = json.loads((root / "q2_v2" / "metrics.json").read_text(encoding="utf-8"))
        if q2.get("n_conditions") != 64:
            raise ValueError(f"{root}: Q2-v2 is incomplete")
        if any(q2.get(key) != summary.get(key) for key in ("method", "view", "model_input_view", "seed")):
            raise ValueError(f"{root}: Q2-v2 identity disagrees with run summary")
        mask_hash = q2.get("mask_sha256")
        if not isinstance(mask_hash, str) or len(mask_hash) != 64:
            raise ValueError(f"{root}: Q2-v2 mask hash is missing")
        if expected_mask_hash is None:
            expected_mask_hash = mask_hash
        elif mask_hash != expected_mask_hash:
            raise ValueError("runs do not share one Q2-v2 mask hash")
        identities = {
            (condition["combination"], condition["position"], float(condition["requested_fraction"]))
            for condition in q2["conditions"]
        }
        if len(identities) != 64:
            raise ValueError(f"{root}: Q2-v2 condition identities are incomplete")
        canonical = {(combination, position, float(fraction)) for combination, position, fraction in Q2_V2_CONDITIONS}
        if identities != canonical:
            raise ValueError(f"{root}: Q2-v2 condition identities differ from the canonical matrix")
        if expected_conditions is None:
            expected_conditions = identities
        elif identities != expected_conditions:
            raise ValueError("runs do not share one Q2-v2 condition matrix")
        for condition in q2["conditions"]:
            key = (condition["combination"], condition["position"], float(condition["requested_fraction"]))
            bucket = condition_values.setdefault(key, {})
            for metric in _METRICS:
                value = condition.get(metric)
                if value is not None:
                    bucket.setdefault(metric, []).append(float(value))
    conditions = [
        {
            "combination": key[0],
            "position": key[1],
            "requested_fraction": key[2],
            "metrics": {metric: _summary(values) for metric, values in bucket.items() if len(values) == len(roots)},
        }
        for key, bucket in sorted(condition_values.items(), key=lambda item: (item[0][0], item[0][1] or "", item[0][2]))
    ]
    if len(conditions) != 64:
        raise ValueError("runs do not share the complete Q2-v2 condition matrix")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "method": method,
        "view": view,
        "source_view": view,
        "model_input_view": model_input_view,
        "seeds": sorted(seeds),
        "seed_count": len(seeds),
        "mask_sha256": expected_mask_hash,
        "clean": clean,
        "q2_conditions": conditions,
    }


__all__ = ["summarize_runs"]
