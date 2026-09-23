"""Adapters for the competition-provided feature and special-test files."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from e_emotion.contracts import DatasetVariant, SplitName
from e_emotion.data.paths import DataPathPolicy


def _variant(value: DatasetVariant | str) -> DatasetVariant:
    return value if isinstance(value, DatasetVariant) else DatasetVariant(str(value).lower())


class Attachment2Repository:
    def __init__(self, data_root: str | Path) -> None:
        self.policy = DataPathPolicy(data_root)

    def path_for(self, variant: DatasetVariant | str) -> Path:
        selected = _variant(variant)
        return self.policy.resolve(Path("附件2-数据集特征文件") / f"{selected.value}_50.pkl")

    def load(self, variant: DatasetVariant | str) -> dict[str, Any]:
        with self.policy.require_file(self.path_for(variant)).open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("附件2 pickle must contain a dictionary")
        return payload

    def load_split(self, variant: DatasetVariant | str, split: SplitName | str) -> dict[str, Any]:
        split_name = split.value if isinstance(split, SplitName) else str(split)
        payload = self.load(variant)
        if split_name not in payload:
            raise KeyError(f"missing split: {split_name}")
        return payload[split_name]


class MissingModalityRepository:
    def __init__(self, data_root: str | Path) -> None:
        self.policy = DataPathPolicy(data_root)

    def path_for(self, index: int, variant: DatasetVariant | str) -> Path:
        selected = _variant(variant)
        if not 1 <= index <= 30:
            raise ValueError("附件3 index must be between 1 and 30")
        if selected is DatasetVariant.ALIGNED:
            relative = Path("附件3-模态缺失特征样本") / "对齐版本" / f"附件3_{index:02d}.pkl"
        else:
            relative = (
                Path("附件3-模态缺失特征样本")
                / "未对齐版本"
                / f"附件3_未对齐版本_{index:02d}.pkl"
            )
        return self.policy.resolve(relative)

    def load(self, index: int, variant: DatasetVariant | str) -> dict[str, Any]:
        with self.policy.require_file(self.path_for(index, variant)).open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("附件3 pickle must contain a dictionary")
        return payload


class ExplainabilityRepository:
    def __init__(self, data_root: str | Path) -> None:
        self.policy = DataPathPolicy(data_root)

    def path_for(self, index: int, variant: DatasetVariant | str) -> Path:
        selected = _variant(variant)
        if not 1 <= index <= 20:
            raise ValueError("附件4 index must be between 1 and 20")
        relative = (
            Path("附件4-可解释专项视频样本与特征文件")
            / "附件4-可解释专项视频样本与特征文件"
            / ("对齐版本" if selected is DatasetVariant.ALIGNED else "未对齐版本")
            / f"{index:02d}.pkl"
        )
        return self.policy.resolve(relative)

    def load(self, index: int, variant: DatasetVariant | str) -> dict[str, Any]:
        with self.policy.require_file(self.path_for(index, variant)).open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("附件4 pickle must contain a dictionary")
        return payload
