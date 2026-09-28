#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""附件2未对齐版 → 标准化/未对齐版本/processed_po

与对齐版 processed_po 同契约（BERT 逐槽、P/O、float32、仅 train-O 上 z-score、¬O 置零），
差别仅在于音视保持原生 500 槽、各模态自有有效范围（不要求跨模态时间对齐）。

文本：L=50；音视：L=500。P/O 分模态存储（P_T/O_T 与 P_A/O_A、P_V/O_V 长度不同）。
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

# 兼容 NumPy 2.x 协议写入的 pkl
import numpy.core as _np_core

sys.modules.setdefault("numpy._core", _np_core)
sys.modules.setdefault("numpy._core.numeric", _np_core.numeric)
sys.modules.setdefault("numpy._core.multiarray", _np_core.multiarray)
sys.modules.setdefault("numpy._core.fromnumeric", _np_core.fromnumeric)

from cpmcm_huawei2026.shared.paths import asset_root, relative_to_asset_root

ASSET_ROOT = asset_root()
RAW = ASSET_ROOT / "AAAdata" / "Appendix_2_raw" / "未对齐版本"
OUT = ASSET_ROOT / "AAAdata" / "Appendix_2_regenerated" / "未对齐版本"
BERT = ASSET_ROOT / "AAAmodel" / "bert-base-uncased"
EPS = 1e-8
SPECIAL = {0, 101, 102}
NOTE = (
    "未对齐: 文本50、音视500；各模态独立有效范围（长度/attention），"
    "不要求跨模态对齐；O=观测∩P；¬O置零；train-O分模态z-score；无池化"
)


def project_path(value: Path) -> Path:
    value = Path(value)
    return value if value.is_absolute() else ASSET_ROOT / value


def rel_display(value: Path) -> str:
    value = Path(value)
    return relative_to_asset_root(value)


def load_split(split: str, raw: Path) -> dict:
    with open(raw / f"{split}.pkl", "rb") as f:
        d = pickle.load(f)
    tb = np.asarray(d["text_bert"])
    audio = np.asarray(d["audio"], dtype=np.float64)
    vision = np.asarray(d["vision"], dtype=np.float64)
    assert tb.ndim == 3 and tb.shape[1:] == (3, 50), tb.shape
    assert audio.shape[1:] == (500, 74), audio.shape
    assert vision.shape[1:] == (500, 35), vision.shape
    assert np.isfinite(audio).all() and np.isfinite(vision).all()
    al = np.asarray(d["audio_lengths"], dtype=np.int64).reshape(-1)
    vl = np.asarray(d["vision_lengths"], dtype=np.int64).reshape(-1)
    n = tb.shape[0]
    assert al.shape == (n,) and vl.shape == (n,)
    assert (al >= 0).all() and (al <= 500).all()
    assert (vl >= 0).all() and (vl <= 500).all()
    return {
        "I": tb[:, 0, :].astype(np.int64),
        "Q": tb[:, 1, :].astype(bool),
        "G": tb[:, 2, :].astype(np.int64),
        "audio": audio,
        "vision": vision,
        "audio_lengths": al,
        "vision_lengths": vl,
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


def length_extent(lengths: np.ndarray, L: int) -> np.ndarray:
    t = np.arange(L)[None, :]
    return t < lengths[:, None]


def build_po(pack: dict):
    """各模态独立 P/O：文本用 attention 前缀；音视用官方 lengths 前缀。"""
    ids, attn = pack["I"], pack["Q"]
    o_t_raw = attn & ~np.isin(ids, list(SPECIAL))
    o_a_raw = np.any(pack["audio"] != 0, axis=-1)
    o_v_raw = np.any(pack["vision"] != 0, axis=-1)

    p_t = union_extent(attn) & ~np.isin(ids, list(SPECIAL))
    p_a = length_extent(pack["audio_lengths"], 500)
    p_v = length_extent(pack["vision_lengths"], 500)

    o_t = o_t_raw & p_t
    o_a = o_a_raw & p_a
    o_v = o_v_raw & p_v
    return p_t, o_t, p_a, o_a, p_v, o_v


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
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    raw = project_path(args.raw)
    out = project_path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    packs = {}
    for split in ("train", "valid", "test"):
        pack = load_split(split, raw)
        p_t, o_t, p_a, o_a, p_v, o_v = build_po(pack)
        pack.update(P_T=p_t, O_T=o_t, P_A=p_a, O_A=o_a, P_V=p_v, O_V=o_v)
        # ¬O 处置零（标准化前）
        pack["audio"] = pack["audio"].astype(np.float32)
        pack["vision"] = pack["vision"].astype(np.float32)
        pack["audio"][~o_a] = 0.0
        pack["vision"][~o_v] = 0.0
        print(
            f"[{split}] N={len(pack['id'])}  "
            f"O均值 T/A/V={o_t.mean():.4f}/{o_a.mean():.4f}/{o_v.mean():.4f}  "
            f"P均值 T/A/V={p_t.mean():.4f}/{p_a.mean():.4f}/{p_v.mean():.4f}  "
            f"holes A/V={(p_a & ~o_a).sum()}/{(p_v & ~o_v).sum()}",
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
            xt[~pack["O_T"]] = 0.0
            # CLS/SEP 虽不在 O_T，编码时仍由 attention 保留；内容位外再清零一次
            pack["XT_raw"] = xt
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    train = packs["train"]
    mu_t, sig_t, n_t = fit_scaler(train["XT_raw"], train["O_T"])
    mu_a, sig_a, n_a = fit_scaler(train["audio"], train["O_A"])
    mu_v, sig_v, n_v = fit_scaler(train["vision"], train["O_V"])
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
        "bert": "AAAmodel/bert-base-uncased",
        "variant": "unaligned_native500",
        "lengths": {"text": 50, "audio": 500, "vision": 500},
        "mask_rule": (
            "文本P=attention前缀且剔除PAD/CLS/SEP；音视P=t<official_length；"
            "O=观测∩P；各模态独立，无跨模态对齐；¬O置零；train-O z-score"
        ),
        "note": NOTE,
        "splits": {},
    }

    for split, pack in packs.items():
        xt = apply_scaler(pack["XT_raw"], pack["O_T"], mu_t, sig_t)
        xa = apply_scaler(pack["audio"], pack["O_A"], mu_a, sig_a)
        xv = apply_scaler(pack["vision"], pack["O_V"], mu_v, sig_v)
        payload = {
            "XT": xt,
            "XA": xa,
            "XV": xv,
            "P_T": pack["P_T"],
            "O_T": pack["O_T"],
            "P_A": pack["P_A"],
            "O_A": pack["O_A"],
            "P_V": pack["P_V"],
            "O_V": pack["O_V"],
            "I": pack["I"],
            "Q": pack["Q"],
            "G": pack["G"],
            "audio_lengths": pack["audio_lengths"],
            "vision_lengths": pack["vision_lengths"],
            "id": pack["id"],
            "ids": pack["id"],
            "classification_labels": pack["classification_labels"],
            "regression_labels": pack["regression_labels"],
            "note": NOTE,
        }
        path = out / f"{split}.npz"
        np.savez_compressed(path, **payload)
        report["splits"][split] = {
            "n": int(len(pack["id"])),
            "O_mean": [float(pack["O_T"].mean()), float(pack["O_A"].mean()), float(pack["O_V"].mean())],
            "P_mean": [float(pack["P_T"].mean()), float(pack["P_A"].mean()), float(pack["P_V"].mean())],
            "holes_P_not_O": [
                int((pack["P_T"] & ~pack["O_T"]).sum()),
                int((pack["P_A"] & ~pack["O_A"]).sum()),
                int((pack["P_V"] & ~pack["O_V"]).sum()),
            ],
            "XT_shape": list(xt.shape),
            "XA_shape": list(xa.shape),
            "XV_shape": list(xv.shape),
            "XT_dtype": str(xt.dtype),
        }
        print(f"[save] {path} XT={xt.shape} XA={xa.shape} XV={xv.shape}", flush=True)

    (out / "preprocess_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[done]", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
