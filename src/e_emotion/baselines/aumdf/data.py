"""Competition-only input boundary and train-fitted normalization."""
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from e_emotion.data import load_processed_dataset

from e_emotion.baselines.aumdf.model import MODALITIES

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CANONICAL_PROCESSED_ROOT = "/user_home/gaojianan/CPMCM/AAAdata/Appendix_2/标准化/对齐版本/processed"
_ALLOWED_CANONICAL_METADATA = {
    "scaler_params.npz",
    "preprocess_report.json",
    "model_input_contract.json",
    "bert_encode_report.json",
    "XT_raw_before_zscore_train.npz",
}


def resolve_data_path(project_root, path):
    root = Path(project_root).resolve()
    value = Path(path)
    value = (root / value).resolve() if not value.is_absolute() else value.resolve()
    if not value.is_relative_to((root / "data").resolve()):
        raise ValueError("input must resolve inside project data/")
    return value


def resolve_source_path(project_root, path, *, strict=True, canonical_root=CANONICAL_PROCESSED_ROOT):
    """Resolve a data source, optionally enforcing the public NPZ boundary.

    ``strict=False`` remains available only for explicit historical unit-test
    fixtures. The default accepts only the exact canonical Appendix 2 root.
    """
    candidate = Path(path).expanduser()
    if candidate.suffix.lower() == ".pkl":
        if strict:
            raise ValueError("legacy .pkl inputs are rejected; use the processed NPZ directory")
        return resolve_data_path(project_root, path)
    if candidate.suffix.lower() == ".npz" or candidate.is_dir():
        resolved = candidate.resolve()
        if resolved.suffix.lower() == ".npz":
            if strict:
                raise ValueError("dataset input must be the processed NPZ directory, not a single split")
            root = resolved.parent
        else:
            root = resolved
        if strict:
            canonical = Path(canonical_root).expanduser().resolve()
            if root != canonical:
                raise ValueError(f"dataset root must be the canonical processed root {canonical}")
            if not root.is_dir():
                raise ValueError(f"canonical processed root must exist: {canonical}")
        expected = [root / f"{split}.npz" for split in ("train", "valid", "test")]
        if not all(item.is_file() for item in expected):
            raise ValueError(f"processed data root must contain train.npz, valid.npz and test.npz: {root}")
        if strict:
            names = {item.name for item in root.iterdir() if item.is_file()}
            required = {item.name for item in expected}
            unknown = names - required - _ALLOWED_CANONICAL_METADATA
            if unknown:
                raise ValueError(f"canonical processed root contains unsupported files: {sorted(unknown)}")
        return resolved
    if strict:
        raise ValueError("dataset input must be the canonical processed NPZ directory")
    return resolve_data_path(project_root, path)


class FeatureDataset(Dataset):
    def __init__(self, section, scaler):
        self.section, self.scaler = section, scaler

    def __len__(self):
        return len(self.section["id"])

    def __getitem__(self, i):
        features, masks, observed = {}, {}, {}
        for m in MODALITIES:
            raw_valid = self.section["valid"]
            valid = raw_valid[m][i] if isinstance(raw_valid, dict) else raw_valid[i]
            raw = self.section[m][i]
            # Processed NPZ files carry native availability explicitly. Never
            # infer missingness from standardized feature values.
            raw_observed = self.section.get("observed", {}).get(m)
            is_observed = valid if raw_observed is None else raw_observed[i]
            mean = np.asarray(self.scaler[m]["mean"], dtype=np.float32)
            std = np.asarray(self.scaler[m]["std"], dtype=np.float32)
            scaled = ((raw - mean) / std).astype(np.float32)
            scaled[~is_observed] = 0
            features[m] = torch.from_numpy(scaled)
            masks[m] = torch.from_numpy(valid.copy())
            observed[m] = torch.from_numpy(is_observed.copy())
        return {"id": str(self.section["id"][i]), "features": features, "valid": masks,
                "observed": observed, "target": torch.tensor(self.section["regression_labels"][i], dtype=torch.float32)}


def load_attachment2(path, project_root=PROJECT_ROOT, expected_dims=(768,74,35), scaler=None,
                     *, strict=True, canonical_root=CANONICAL_PROCESSED_ROOT):
    candidate = Path(path).expanduser()
    if strict:
        source = resolve_source_path(project_root, candidate, strict=True, canonical_root=canonical_root)
        return load_processed_attachment2(source, expected_dims=expected_dims)
    if candidate.suffix.lower() == ".npz" or candidate.is_dir():
        return load_processed_attachment2(candidate, expected_dims=expected_dims)
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


def load_processed_attachment2(path, expected_dims=(768, 74, 35)):
    """Adapt the shared processed NPZ contract to AUMDF's dataset interface."""
    root = Path(path).expanduser()
    root = root.parent if root.suffix == ".npz" else root
    dataset = load_processed_dataset(root)
    sections = {}
    for split, section in dataset.splits.items():
        for modality, expected in zip(MODALITIES, expected_dims):
            values = section.features[modality]
            if values.shape != (section.size, 50, expected):
                raise ValueError(f"{split}.{modality}: expected {(section.size, 50, expected)}, got {values.shape}")
        sections[split] = {
            "id": [str(item) for item in section.ids.tolist()],
            "regression_labels": section.regression,
            "classification_labels": section.classification,
            "valid": {m: section.native_valid_mask[m] for m in MODALITIES},
            "observed": {m: section.observed_mask[m] for m in MODALITIES},
            **{m: section.features[m] for m in MODALITIES},
        }
    # The NPZ values are already organizer-standardized. Applying the saved
    # train statistics again would double-standardize the input, so the AUMDF
    # adapter uses an identity transform and records the source explicitly.
    scaler = {m: {"mean": np.zeros(d, dtype=np.float32).tolist(),
                  "std": np.ones(d, dtype=np.float32).tolist()}
              for m, d in zip(MODALITIES, expected_dims)}
    datasets = {split: FeatureDataset(section, scaler) for split, section in sections.items()}
    audit = {
        "split_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "data_format": "processed_npz",
        "feature_version": "aligned_processed",
        "data_root": str(root.resolve()),
        "native_mask_keys": {"text": "mT", "audio": "mA", "vision": "mV"},
        "observed_mask_definition": "native_valid_mask & synthetic_keep",
        "normalization_source": "organizer_processed_train_fitted; identity adapter",
        "scaler_fit_split": "organizer_train",
    }
    return datasets, scaler, audit
