"""核验共同播放时间轴掩码、support 与三模态特征契约。"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def verify(directory):
    directory = Path(directory)
    with np.load(directory / "masks.npz", allow_pickle=False) as masks:
        ids = masks["sample_id"]
        n = len(ids)
        if n == 0 or len(set(ids.tolist())) != n:
            raise ValueError("样本为空或ID重复")
        for name in ("text_mask", "audio_mask", "vision_mask"):
            x = masks[name]
            if x.shape != (n, 50) or not np.isin(x, [0, 1]).all():
                raise ValueError(f"{name}: 形状或二值约束错误")
        for name in ("audio_support", "vision_support"):
            if name not in masks.files:
                raise ValueError(f"缺少 {name}")
            x = masks[name]
            if x.shape != (n, 50) or not np.isin(x, [0, 1]).all():
                raise ValueError(f"{name}: 形状或二值约束错误")
        if np.any((masks["audio_support"] == 0) & (masks["audio_mask"] == 1)):
            raise ValueError("音频无支持窗被标为有效")
        if np.any((masks["vision_support"] == 0) & (masks["vision_mask"] == 1)):
            raise ValueError("视觉无支持窗被标为有效")

        text_m = masks["text_mask"]
        audio_m = masks["audio_mask"]
        vision_m = masks["vision_mask"]
        audio_s = masks["audio_support"]
        vision_s = masks["vision_support"]

    with np.load(directory / "text_features.npz", allow_pickle=False) as tf:
        if not np.array_equal(tf["sample_id"], ids):
            raise ValueError("文本特征样本顺序不同")
        text = tf["text"]
        if text.shape != (n, 50, 768) or not np.isfinite(text).all():
            raise ValueError("BERT形状或有限性错误")
        if not np.array_equal(tf["text_mask"], text_m):
            raise ValueError("两份文本掩码不一致")
        if np.any(text[text_m == 0] != 0):
            raise ValueError("无效文本词元未归零")

    with np.load(directory / "audio_features.npz", allow_pickle=False) as af:
        if not np.array_equal(af["sample_id"], ids):
            raise ValueError("音频与掩码样本顺序不同")
        audio = af["audio"]
        if audio.shape != (n, 50, 74) or not np.isfinite(audio).all():
            raise ValueError("COVAREP形状或有限性错误")
        if len(af["feature_names"]) != 74:
            raise ValueError("音频特征名数量错误")
        if not np.array_equal(af["audio_mask"], audio_m):
            raise ValueError("两份音频掩码不一致")
        if np.any(audio[audio_m == 0] != 0):
            raise ValueError("无效音频窗口未归零")
        if "audio_support" in af.files and np.any(audio[af["audio_support"] == 0] != 0):
            raise ValueError("无支持音频窗口未归零")
        if np.any(audio[audio_s == 0] != 0):
            raise ValueError("无支持音频窗口未归零(masks)")

    with np.load(directory / "vision_features.npz", allow_pickle=False) as vf:
        if not np.array_equal(vf["sample_id"], ids):
            raise ValueError("视觉特征样本顺序不同")
        vision = vf["vision"]
        if vision.shape != (n, 50, 35) or not np.isfinite(vision).all():
            raise ValueError("AU35形状或有限性错误")
        if len(vf["feature_names"]) != 35:
            raise ValueError("视觉特征名数量错误")
        if not np.array_equal(vf["vision_mask"], vision_m):
            raise ValueError("两份视觉掩码不一致")
        if np.any(vision[vision_m == 0] != 0):
            raise ValueError("无效视觉窗口未归零")
        if "vision_support" in vf.files and np.any(vision[vf["vision_support"] == 0] != 0):
            raise ValueError("无支持视觉窗口未归零")
        if np.any(vision[vision_s == 0] != 0):
            raise ValueError("无支持视觉窗口未归零(masks)")

    for modality in ("text", "audio", "vision"):
        table = pd.read_csv(directory / f"{modality}_quality.csv")
        if table.sample_id.drop_duplicates().tolist() != ids.tolist():
            raise ValueError(f"{modality}质量表覆盖不一致")
        if len(table) != n * 50:
            raise ValueError(f"{modality}质量记录数错误")
        if "support_mask" not in table.columns:
            raise ValueError(f"{modality}质量表缺少 support_mask")

    report = {
        "samples": n,
        "alignment": "mfa_word",
        "text_shape": list(text.shape),
        "audio_shape": list(audio.shape),
        "vision_shape": list(vision.shape),
        "passed": True,
        "scope": "MFA词槽结构核验；不等于人工边界准确率",
        "valid_fraction": {
            m: float(masks_mean)
            for m, masks_mean in (
                ("text", text_m.mean()),
                ("audio", audio_m.mean()),
                ("vision", vision_m.mean()),
            )
        },
        "support_fraction": {
            "audio": float(audio_s.mean()),
            "vision": float(vision_s.mean()),
        },
    }
    (directory / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="problem1_v3/results")
    print(verify(parser.parse_args().directory))
