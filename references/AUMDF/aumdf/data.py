"""Competition-only input boundary and train-fitted normalization."""
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from aumdf.model import MODALITIES

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_data_path(project_root, path):
    root = Path(project_root).resolve()
    value = Path(path)
    value = (root / value).resolve() if not value.is_absolute() else value.resolve()
    if not value.is_relative_to((root / "data").resolve()):
        raise ValueError("input must resolve inside project data/")
    return value


class FeatureDataset(Dataset):
    def __init__(self, section, scaler):
        self.section, self.scaler = section, scaler

    def __len__(self):
        return len(self.section["id"])

    def __getitem__(self, i):
        valid = self.section["valid"][i]
        features, masks, observed = {}, {}, {}
        for m in MODALITIES:
            raw = self.section[m][i]
            is_observed = valid & np.any(raw != 0, axis=-1)
            mean = np.asarray(self.scaler[m]["mean"], dtype=np.float32)
            std = np.asarray(self.scaler[m]["std"], dtype=np.float32)
            scaled = ((raw - mean) / std).astype(np.float32)
            scaled[~is_observed] = 0
            features[m] = torch.from_numpy(scaled)
            masks[m] = torch.from_numpy(valid.copy())
            observed[m] = torch.from_numpy(is_observed.copy())
        return {"id": str(self.section["id"][i]), "features": features, "valid": masks,
                "observed": observed, "target": torch.tensor(self.section["regression_labels"][i], dtype=torch.float32)}


def load_attachment2(path, project_root=PROJECT_ROOT, expected_dims=(768,74,35), scaler=None):
    path = resolve_data_path(project_root, path)
    # Only locally trusted organizer-provided pickle files; pickle is executable.
    with path.open("rb") as stream:
        raw = pickle.load(stream)
    sections, nonfinite, all_ids = {}, {}, []
    for split in ("train", "valid", "test"):
        section = raw[split]
        n = len(section["id"])
        labels = np.asarray(section["regression_labels"], dtype=np.float32)
        if labels.shape != (n,) or not np.isfinite(labels).all() or (np.abs(labels)>3).any():
            raise ValueError(f"invalid labels in {split}")
        bert = np.asarray(section["text_bert"])
        if bert.ndim != 3 or bert.shape[:2] != (n,3):
            raise ValueError("text_bert attention mask required for padding provenance")
        valid = bert[:,1,:].astype(bool)
        if (np.diff(valid.astype(int), axis=1) > 0).any():
            raise ValueError("padding mask must be a valid prefix")
        target_section = dict(id=list(section["id"]), regression_labels=labels, valid=valid)
        nonfinite[split] = {}
        for m, dimension in zip(MODALITIES, expected_dims):
            array = np.asarray(section[m], dtype=np.float32)
            if array.shape != (*valid.shape,dimension):
                raise ValueError(f"{split}.{m}: requires aligned lengths and configured dimensions")
            nonfinite[split][m] = int((~np.isfinite(array)).sum())
            target_section[m] = np.nan_to_num(array, nan=0, posinf=0, neginf=0)
        classes = np.asarray(section.get("classification_labels", np.sign(labels)+1))
        if not np.array_equal(classes, np.sign(labels)+1):
            raise ValueError("classification labels disagree with negative/neutral/positive mapping")
        sections[split] = target_section
        all_ids.extend(str(s) for s in section["id"])
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("duplicate sample IDs across provided splits")
    if scaler is None:
        scaler = {}
        for m in MODALITIES:
            array = sections["train"][m]
            observed = sections["train"]["valid"] & np.any(array != 0, axis=-1)
            values = array[observed]
            if len(values):
                mean, std = values.mean(0, dtype=np.float64), values.std(0, dtype=np.float64)
            else:
                mean, std = np.zeros(array.shape[-1]), np.ones(array.shape[-1])
            scaler[m] = {"mean":mean.tolist(), "std":np.maximum(std,1e-5).tolist()}
    datasets = {s:FeatureDataset(v,scaler) for s,v in sections.items()}
    audit = {"split_sizes":{s:len(v) for s,v in datasets.items()}, "nonfinite_replaced":nonfinite,
             "padding_source":"text_bert attention-mask valid prefix",
             "zero_rows":"conservatively treated as unavailable; padding recorded separately",
             "scaler_fit_split":"train", "data_file":str(path)}
    return datasets, scaler, audit
