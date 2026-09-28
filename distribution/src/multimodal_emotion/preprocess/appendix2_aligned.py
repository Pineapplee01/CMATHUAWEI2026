#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""附件2对齐版 → 标准化/对齐版本/processed_po

读原始对齐 pkl（文本/音视均 50 槽）→ 冻结 BERT 逐槽编码 → 打 P/O →
仅在 train 的 O 位拟合分模态 z-score → 写出 NPZ + scaler_params。

掩码：P=union_extent(attn|oA|oV)；文本再剔除 PAD/CLS/SEP；O=观测∩P；¬O 置零。
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

import numpy.core as _np_core

sys.modules.setdefault("numpy._core", _np_core)
sys.modules.setdefault("numpy._core.numeric", _np_core.numeric)
sys.modules.setdefault("numpy._core.multiarray", _np_core.multiarray)
sys.modules.setdefault("numpy._core.fromnumeric", _np_core.fromnumeric)

from multimodal_emotion.shared.paths import (
    data_root,
    distribution_root,
    reference_root,
    results_root,
    relative_to_distribution_root,
    resolve_distribution_path,
)

ROOT = distribution_root()
RAW = data_root() / "appendix_2" / "aligned_50.pkl"
OUT = results_root() / "appendix_2" / "aligned"
BERT = reference_root() / "models" / "bert-base-uncased"
EPS = 1e-8
SPECIAL = {0, 101, 102}
NOTE = "P=union_extent; O=观测∩P; 特征仅 O=1 保留并 z-score; ¬O 置零; 无补全预处理"


def project_path(value: Path) -> Path:
    value = Path(value)
    return resolve_distribution_path(value)


def rel_display(value: Path) -> str:
    value = Path(value)
    return relative_to_distribution_root(value)


def load_source(raw: Path) -> dict:
    """Load the contest's one-file Appendix 2 bundle without rewriting it."""
    if raw.is_dir():
        raise ValueError(f"--raw must point to aligned_50.pkl, not a directory: {raw}")
    with raw.open("rb") as f:
        source = pickle.load(f)
    required = {"train", "valid", "test"}
    if not isinstance(source, dict) or not required.issubset(source):
        raise ValueError(f"{raw} must contain top-level train/valid/test splits")
    return source


def load_split(source: dict, split: str) -> dict:
    d = source[split]
    tb = np.asarray(d["text_bert"])
    audio = np.asarray(d["audio"], dtype=np.float64)
    vision = np.asarray(d["vision"], dtype=np.float64)
    assert tb.ndim == 3 and tb.shape[1:] == (3, 50), tb.shape
    assert audio.shape[1:] == (50, 74), audio.shape
    assert vision.shape[1:] == (50, 35), vision.shape
    assert np.isfinite(audio).all() and np.isfinite(vision).all()
    return {
        "I": tb[:, 0, :].astype(np.int64),
        "Q": tb[:, 1, :].astype(bool),
        "G": tb[:, 2, :].astype(np.int64),
        "audio": audio,
        "vision": vision,
        "id": np.asarray([str(x) for x in d["id"]], dtype=object),
        "classification_labels": np.asarray(d["classification_labels"], dtype=np.float64).reshape(-1),
        "regression_labels": np.asarray(d["regression_labels"], dtype=np.float64).reshape(-1),
    }


def union_extent(support: np.ndarray) -> np.ndarray:
    n, length = support.shape
    out = np.zeros((n, length), dtype=bool)
    for i in range(n):
        pos = np.flatnonzero(support[i])
        if len(pos):
            out[i, : int(pos[-1]) + 1] = True
    return out


def build_po(pack: dict):
    ids, attn = pack["I"], pack["Q"]
    o_t_raw = attn & ~np.isin(ids, list(SPECIAL))
    o_a_raw = np.any(pack["audio"] != 0, axis=-1)
    o_v_raw = np.any(pack["vision"] != 0, axis=-1)
    extent = union_extent(attn | o_a_raw | o_v_raw)
    p_t = extent & ~np.isin(ids, list(SPECIAL))
    p_a = extent.copy()
    p_v = extent.copy()
    o_t = o_t_raw & p_t
    o_a = o_a_raw & p_a
    o_v = o_v_raw & p_v
    p = np.stack([p_t, p_a, p_v], axis=1)
    o = np.stack([o_t, o_a, o_v], axis=1)
    return p, o


@torch.no_grad()
def encode_bert(ids: np.ndarray, model, device: str, batch_size: int) -> np.ndarray:
    n, length = ids.shape
    out = np.zeros((n, length, model.config.hidden_size), dtype=np.float32)
    model.eval()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        input_ids = torch.as_tensor(ids[start:end], dtype=torch.long, device=device)
        attn = (input_ids != 0).long()
        hidden = model(input_ids=input_ids, attention_mask=attn).last_hidden_state
        hidden = hidden * attn.unsqueeze(-1).to(hidden.dtype)
        out[start:end] = hidden.detach().cpu().numpy().astype(np.float32)
        if (start // batch_size) % 20 == 0:
            print(f"  BERT {end}/{n}", flush=True)
    return out


def fit_scaler(x: np.ndarray, usable: np.ndarray):
    obs = x[usable]
    if obs.size == 0:
        raise RuntimeError("训练集无有效位置，无法拟合标准化")
    mu = obs.mean(axis=0, dtype=np.float64)
    sigma = np.maximum(obs.std(axis=0, dtype=np.float64), EPS)
    return mu, sigma, int(obs.shape[0])


def apply_scaler(x: np.ndarray, usable: np.ndarray, mu, sigma) -> np.ndarray:
    z = ((x.astype(np.float64) - mu) / sigma).astype(np.float32)
    z[~usable] = 0.0
    return z


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    raw = project_path(args.raw)
    out = project_path(args.output)
    if out.is_relative_to(data_root()):
        parser.error("--output must not be inside data/; use results/ for derived files")
    out.mkdir(parents=True, exist_ok=True)

    source = load_source(raw)
    packs = {}
    for split in ("train", "valid", "test"):
        pack = load_split(source, split)
        p, o = build_po(pack)
        pack["P"], pack["O"] = p, o
        pack["audio"] = pack["audio"].astype(np.float32)
        pack["vision"] = pack["vision"].astype(np.float32)
        pack["audio"][~o[:, 1]] = 0.0
        pack["vision"][~o[:, 2]] = 0.0
        print(
            f"[{split}] N={len(pack['id'])}  "
            f"O均值 T/A/V={o[:,0].mean():.4f}/{o[:,1].mean():.4f}/{o[:,2].mean():.4f}  "
            f"P均值 T/A/V={p[:,0].mean():.4f}/{p[:,1].mean():.4f}/{p[:,2].mean():.4f}  "
            f"holes A/V={(p[:,1] & ~o[:,1]).sum()}/{(p[:,2] & ~o[:,2]).sum()}",
            flush=True,
        )
        packs[split] = pack

    print(f"加载冻结 BERT: {BERT} on {device}", flush=True)
    model = AutoModel.from_pretrained(str(BERT), local_files_only=True).to(device)
    model.requires_grad_(False)
    try:
        for split, pack in packs.items():
            print(f"编码 {split} ...", flush=True)
            xt = encode_bert(pack["I"], model, device, args.batch_size)
            xt[~pack["O"][:, 0]] = 0.0
            pack["XT_raw"] = xt
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    train = packs["train"]
    mu_t, sig_t, n_t = fit_scaler(train["XT_raw"], train["O"][:, 0])
    mu_a, sig_a, n_a = fit_scaler(train["audio"], train["O"][:, 1])
    mu_v, sig_v, n_v = fit_scaler(train["vision"], train["O"][:, 2])
    print(f"[scaler] T/A/V 有效行数={n_t}/{n_a}/{n_v}", flush=True)
    np.savez(
        out / "scaler_params.npz",
        mu_T=mu_t, sigma_T=sig_t,
        mu_A=mu_a, sigma_A=sig_a,
        mu_V=mu_v, sigma_V=sig_v,
    )

    report = {
        "source": rel_display(raw),
        "output": rel_display(out),
        "bert": rel_display(BERT),
        "mask_rule": (
            "P=union_extent(attn|oA|oV); O=观测∩P; "
            "特征在 ¬O 处置零（含 P 外与 P∧¬O）；分模态 train-O z-score；无补全"
        ),
        "note": NOTE,
        "splits": {},
    }

    for split, pack in packs.items():
        xt = apply_scaler(pack["XT_raw"], pack["O"][:, 0], mu_t, sig_t)
        xa = apply_scaler(pack["audio"], pack["O"][:, 1], mu_a, sig_a)
        xv = apply_scaler(pack["vision"], pack["O"][:, 2], mu_v, sig_v)
        payload = {
            "XT": xt, "XA": xa, "XV": xv,
            "P": pack["P"], "O": pack["O"],
            "I": pack["I"], "Q": pack["Q"], "G": pack["G"],
            "id": pack["id"], "ids": pack["id"],
            "classification_labels": pack["classification_labels"],
            "regression_labels": pack["regression_labels"],
            "note": NOTE,
        }
        path = out / f"{split}.npz"
        np.savez_compressed(path, **payload)
        report["splits"][split] = {
            "n": int(len(pack["id"])),
            "O_mean": [float(pack["O"][:, m].mean()) for m in range(3)],
            "P_mean": [float(pack["P"][:, m].mean()) for m in range(3)],
            "holes_P_not_O": [int((pack["P"][:, m] & ~pack["O"][:, m]).sum()) for m in range(3)],
            "XT_dtype": str(xt.dtype),
        }
        print(f"[save] {rel_display(path)} XT={xt.shape}", flush=True)

    (out / "preprocess_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[done]", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
