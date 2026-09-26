"""Q2-v2 physical-window materialization for Problem 2 input views."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from e_emotion.problem2_fair.core import (
    MODALITIES,
    Problem2Split,
    Q2_V2_CONDITIONS,
    _raw_window,
    _selected,
    _window_start,
)
from e_emotion.problem2_fair.views import Problem2Dataset


Q2_V2_PROTOCOL_VERSION = "q2-continuous-local-v2"


def _manifest_hash(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("mask_sha256", None)
    canonical = json.dumps(unsigned, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _support_on_text_axis(split: Problem2Split, row: int, modalities: tuple[str, ...]) -> np.ndarray:
    support = np.zeros(50, dtype=bool)
    for modality in modalities:
        value = split.physical_support[modality][row]
        if value.shape[0] == 50:
            support |= value
        else:
            for index in np.flatnonzero(value):
                support[min(49, int(index * 50 // value.shape[0]))] = True
    return support


def _window_for_row(
    split: Problem2Split,
    row: int,
    modalities: tuple[str, ...],
    fraction: float,
    position: str,
) -> tuple[int, int]:
    # The physical interval is shared across T/A/V conditions.  The selected
    # combination controls deletion only; it must not move the interval.
    support = _support_on_text_axis(split, row, MODALITIES)
    coordinates = np.flatnonzero(support)
    if not len(coordinates):
        return 0, 0
    start, end = int(coordinates[0]), int(coordinates[-1]) + 1
    local_start, local_end = _window_start(end - start, fraction, position)
    return start + local_start, start + local_end


def _would_remove_all_observations(
    split: Problem2Split,
    row: int,
    modalities: tuple[str, ...],
    text_start: int,
    text_end: int,
) -> bool:
    for modality in modalities:
        observed = split.observed[modality][row]
        if not observed.any():
            continue
        start, end = (text_start, text_end) if observed.shape[0] == 50 else _raw_window(observed.shape[0], text_start, text_end)
        if observed[start:end].sum() == observed.sum():
            return True
    return False


def _safe_window_for_row(
    split: Problem2Split,
    row: int,
    modalities: tuple[str, ...],
    fraction: float,
    position: str,
) -> tuple[int, int]:
    start, end = _window_for_row(split, row, modalities, fraction, position)
    # Keep one globally shared physical interval.  Safety must inspect every
    # modality, otherwise the same condition position moves with the selected
    # combination (for example T versus TA).
    while end > start and _would_remove_all_observations(split, row, MODALITIES, start, end):
        if position == "end":
            start += 1
        else:
            end -= 1
    return start, end


def condition_windows(
    split: Problem2Split,
    combination: str,
    position: str | None,
    fraction: float,
) -> list[dict[str, tuple[int | None, int | None]]]:
    """Return physical interval coordinates for each sample and selected modality."""
    selected = _selected(combination)
    if not selected:
        if position is not None or fraction != 0.0:
            raise ValueError("complete Q2 condition must use None and 0.0")
        return [{name: (None, None) for name in MODALITIES} for _ in range(split.size)]
    if position is None:
        raise ValueError("missing Q2 condition requires a position")
    result = []
    for row in range(split.size):
        text_start, text_end = _safe_window_for_row(split, row, selected, fraction, position)
        item: dict[str, tuple[int | None, int | None]] = {}
        for modality in MODALITIES:
            if modality not in selected or text_start == text_end:
                item[modality] = (None, None)
                continue
            length = split.observed[modality].shape[1]
            item[modality] = (text_start, text_end) if length == 50 else _raw_window(length, text_start, text_end)
        result.append(item)
    return result


def materialize_q2_condition(
    split: Problem2Split,
    combination: str,
    position: str | None,
    fraction: float,
    *,
    entries: list[Mapping[str, Any]] | None = None,
) -> Problem2Split:
    """Apply one Q2-v2 condition while retaining immutable physical support P."""
    selected = _selected(combination)
    if entries is None:
        windows = condition_windows(split, combination, position, fraction)
    else:
        expected_ids = {str(value) for value in split.ids.tolist()}
        by_id: dict[str, Mapping[str, Any]] = {}
        for entry in entries:
            sample_id = str(entry["sample_id"])
            if (
                entry["split"] != split.split
                or entry["combination"] != combination
                or entry["position"] != position
                or float(entry["requested_fraction"]) != float(fraction)
                or sample_id in by_id
            ):
                raise ValueError("Q2-v2 manifest condition identity is invalid")
            by_id[sample_id] = entry
        if set(by_id) != expected_ids:
            raise ValueError("Q2-v2 manifest does not cover the split sample IDs")
        windows = []
        for sample_id in split.ids.tolist():
            entry = by_id[str(sample_id)]
            window = {}
            for modality in MODALITIES:
                saved = entry["window_by_modality"][modality]
                start, end = saved["start"], saved["end"]
                if start is None or end is None:
                    if start is not None or end is not None:
                        raise ValueError("Q2-v2 manifest has a half-open missing window")
                    window[modality] = (None, None)
                else:
                    if modality not in selected or not 0 <= int(start) < int(end) <= split.observed[modality].shape[1]:
                        raise ValueError("Q2-v2 manifest contains an invalid modality window")
                    window[modality] = (int(start), int(end))
            windows.append(window)
    observed = {name: np.array(mask, copy=True) for name, mask in split.observed.items()}
    for row, window in enumerate(windows):
        for modality in selected:
            start, end = window[modality]
            if start is not None and end is not None:
                observed[modality][row, start:end] = False
            if entries is not None:
                sample_id = str(split.ids[row])
                removed = int(split.observed[modality][row].sum() - observed[modality][row].sum())
                if removed != int(by_id[sample_id]["synthetic_missing_counts"][modality]):
                    raise ValueError("Q2-v2 manifest deleted-count differs from applied window")
    values = {name: np.array(value, copy=True) for name, value in split.values.items()}
    for name in MODALITIES:
        values[name][~observed[name]] = 0.0
    input_ids = np.array(split.input_ids, copy=True)
    input_ids[~observed["text"]] = 0
    return Problem2Split(
        ids=split.ids,
        values=values,
        physical_support=split.physical_support,
        observed=observed,
        input_ids=input_ids,
        regression=split.regression,
        classification=split.classification,
        view=split.view,
        split=split.split,
    )


def pool_unaligned_to_text_slots(split: Problem2Split) -> Problem2Split:
    """Pool 500-position audio and vision after any Q2 observation deletion."""
    if split.view != "unaligned_po":
        raise ValueError("only unaligned_po can be pooled into unaligned_windowed")
    values = {"text": np.array(split.values["text"], copy=True)}
    support = {"text": np.array(split.physical_support["text"], copy=True)}
    observed = {"text": np.array(split.observed["text"], copy=True)}
    for modality in ("audio", "vision"):
        source = split.values[modality]
        source_support = split.physical_support[modality]
        source_observed = split.observed[modality]
        pooled = np.zeros((split.size, 50, source.shape[2]), dtype=source.dtype)
        pooled_support = np.zeros((split.size, 50), dtype=bool)
        pooled_observed = np.zeros((split.size, 50), dtype=bool)
        for slot in range(50):
            start, end = _raw_window(source.shape[1], slot, slot + 1)
            local_support = source_support[:, start:end]
            local_observed = source_observed[:, start:end]
            pooled_support[:, slot] = local_support.any(axis=1)
            pooled_observed[:, slot] = local_observed.any(axis=1)
            weighted = source[:, start:end] * local_observed[..., None]
            counts = local_observed.sum(axis=1, keepdims=True)
            pooled[:, slot] = np.divide(
                weighted.sum(axis=1),
                counts,
                out=np.zeros((split.size, source.shape[2]), dtype=source.dtype),
                where=counts > 0,
            )
        values[modality] = pooled
        support[modality] = pooled_support
        observed[modality] = pooled_observed
    return Problem2Split(
        ids=split.ids,
        values=values,
        physical_support=support,
        observed=observed,
        input_ids=split.input_ids,
        regression=split.regression,
        classification=split.classification,
        view="unaligned_windowed",
        split=split.split,
    )


def _dataset_digest(split: Problem2Split) -> str:
    digest = hashlib.sha256()
    for value in (
        split.ids,
        split.input_ids,
        split.regression,
        split.classification,
        *(split.physical_support[name] for name in MODALITIES),
        *(split.observed[name] for name in MODALITIES),
    ):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_hashes(dataset: Problem2Dataset) -> dict[str, str]:
    hashes = {}
    for name in ("train", "valid", "test"):
        source = dataset.root / f"{name}.npz"
        hashes[name] = _file_hash(source) if source.is_file() else _dataset_digest(dataset[name])
    return hashes


def _entry(
    split: Problem2Split,
    row: int,
    condition: tuple[str, str | None, float],
    windows: list[dict[str, tuple[int | None, int | None]]],
) -> dict[str, Any]:
    combination, position, fraction = condition
    window = windows[row]
    selected = set(_selected(combination))
    missing = {}
    for modality in MODALITIES:
        start, end = window[modality]
        count = 0 if start is None else int(split.observed[modality][row, start:end].sum())
        missing[modality] = count if modality in selected else 0
    return {
        "split": split.split,
        "sample_id": str(split.ids[row]),
        "combination": combination,
        "position": position,
        "requested_fraction": fraction,
        "window_by_modality": {
            name: {"start": window[name][0], "end": window[name][1]}
            for name in MODALITIES
        },
        "physical_support_counts": {name: int(split.physical_support[name][row].sum()) for name in MODALITIES},
        "native_observed_counts": {name: int(split.observed[name][row].sum()) for name in MODALITIES},
        "synthetic_missing_counts": missing,
    }


def build_q2_v2_manifest(dataset: Problem2Dataset) -> dict[str, Any]:
    """Build the exact 64-condition Q2-v2 manifest for one declared data view."""
    entries: list[dict[str, Any]] = []
    for split_name in ("train", "valid", "test"):
        split = dataset[split_name]
        for condition in Q2_V2_CONDITIONS:
            windows = condition_windows(split, *condition)
            entries.extend(_entry(split, row, condition, windows) for row in range(split.size))
    result: dict[str, Any] = {
        "protocol_version": Q2_V2_PROTOCOL_VERSION,
        "view": dataset.view,
        "condition_count": len(Q2_V2_CONDITIONS),
        "conditions": [
            {"combination": combination, "position": position, "requested_fraction": fraction}
            for combination, position, fraction in Q2_V2_CONDITIONS
        ],
        "data_hashes": _data_hashes(dataset),
        "entries": entries,
    }
    result["mask_sha256"] = _manifest_hash(result)
    return result


def ensure_q2_v2_manifest(dataset: Problem2Dataset, path: str | Path) -> dict[str, Any]:
    """Reuse a matching view-level manifest or build it exactly once."""
    target = Path(path)
    expected_hashes = _data_hashes(dataset)
    if target.exists():
        cached = json.loads(target.read_text(encoding="utf-8"))
        if cached.get("mask_sha256") != _manifest_hash(cached):
            raise ValueError(f"cached Q2-v2 manifest hash is invalid: {target}")
        if (
            cached.get("protocol_version") != Q2_V2_PROTOCOL_VERSION
            or cached.get("view") != dataset.view
            or cached.get("condition_count") != len(Q2_V2_CONDITIONS)
            or cached.get("data_hashes") != expected_hashes
        ):
            raise ValueError(f"cached Q2-v2 manifest does not match {dataset.view}: {target}")
        return cached
    manifest = build_q2_v2_manifest(dataset)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest


def validate_q2_v2_manifest(manifest: Mapping[str, Any]) -> bool:
    """Verify the self-contained hash and basic identity of a stored manifest."""
    if manifest.get("protocol_version") != Q2_V2_PROTOCOL_VERSION:
        raise ValueError("unexpected Q2-v2 protocol version")
    if manifest.get("condition_count") != len(Q2_V2_CONDITIONS):
        raise ValueError("unexpected Q2-v2 condition count")
    if manifest.get("mask_sha256") != _manifest_hash(manifest):
        raise ValueError("Q2-v2 manifest hash is invalid")
    return True


__all__ = [
    "Q2_V2_PROTOCOL_VERSION",
    "build_q2_v2_manifest",
    "condition_windows",
    "ensure_q2_v2_manifest",
    "materialize_q2_condition",
    "pool_unaligned_to_text_slots",
    "validate_q2_v2_manifest",
]
