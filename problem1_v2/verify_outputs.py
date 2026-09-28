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
        for name in ("audio_support", "vision_support", "text_support"):
            if name not in masks.files:
                # text_support 为新字段；旧结果可缺，缺则跳过严格检查
                if name == "text_support":
                    continue
                raise ValueError(f"缺少 {name}")
            x = masks[name]
            if x.shape != (n, 50) or not np.isin(x, [0, 1]).all():
                raise ValueError(f"{name}: 形状或二值约束错误")
        if np.any((masks["audio_support"] == 0) & (masks["audio_mask"] == 1)):
            raise ValueError("音频无支持窗被标为有效")
        if np.any((masks["vision_support"] == 0) & (masks["vision_mask"] == 1)):
            raise ValueError("视觉无支持窗被标为有效")
        if "text_support" in masks.files and np.any(
            (masks["text_support"] == 0) & (masks["text_mask"] == 1)
        ):
            raise ValueError("文本无支持窗被标为有效")

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
        if "input_ids" not in tf.files:
            raise ValueError("缺少 input_ids（应含 [CLS]/[SEP]，对齐附件2 I）")
        input_ids = tf["input_ids"]
        if input_ids.shape != (n, 50):
            raise ValueError("input_ids 形状错误")
        # BERT：[CLS]=101，[SEP]=102
        if not np.all(input_ids[:, 0] == 101):
            raise ValueError("槽0必须为[CLS]=101")
        for i in range(n):
            row = input_ids[i]
            valid = text_m[i] == 1
            if 102 not in set(row[valid].tolist()):
                raise ValueError(f"样本{ids[i]} 有效序列缺少[SEP]=102")
        if "text_content_mask" in tf.files:
            cm = tf["text_content_mask"]
            if cm.shape != (n, 50) or not np.isin(cm, [0, 1]).all():
                raise ValueError("text_content_mask 形状/二值错误")
            # content 排除 CLS/SEP/PAD
            if np.any(cm[:, 0] == 1):
                raise ValueError("text_content_mask 不应覆盖 [CLS]")
            if np.any((cm == 1) & (input_ids == 102)):
                raise ValueError("text_content_mask 不应覆盖 [SEP]")
            if np.any((cm == 1) & (text_m == 0)):
                raise ValueError("text_content_mask 不能落在 padding 上")
        if "text_intervals" in tf.files and "text_time_mask" in tf.files:
            intervals = tf["text_intervals"]
            tt = tf["text_time_mask"]
            if intervals.shape != (n, 50, 2):
                raise ValueError("text_intervals 形状错误")
            if np.any((tt == 1) & (intervals[:, :, 1] <= intervals[:, :, 0])):
                raise ValueError("有效词元时间窗非法")
            if np.any((tt == 1) & (text_m == 0)):
                raise ValueError("time_mask 不能落在 padding 上")
            # CLS/SEP 通常无时间窗
            if np.any(tt[:, 0] == 1):
                raise ValueError("[CLS] 不应有 time_mask")
            if np.any((tt == 1) & (input_ids == 102)):
                raise ValueError("[SEP] 不应有 time_mask")

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

    # 同词子词共享时间窗 → 同 mask 下 AV 应一致
    with np.load(directory / "text_features.npz", allow_pickle=False) as tf2:
        if "text_intervals" in tf2.files and "text_time_mask" in tf2.files:
            intervals = tf2["text_intervals"]
            tt = tf2["text_time_mask"]
            with np.load(directory / "audio_features.npz", allow_pickle=False) as af2:
                audio2 = af2["audio"]
                am2 = af2["audio_mask"]
            with np.load(directory / "vision_features.npz", allow_pickle=False) as vf2:
                vision2 = vf2["vision"]
                vm2 = vf2["vision_mask"]
            for i in range(n):
                for k in range(49):
                    if tt[i, k] != 1 or tt[i, k + 1] != 1:
                        continue
                    if not np.allclose(intervals[i, k], intervals[i, k + 1]):
                        continue
                    if am2[i, k] == 1 and am2[i, k + 1] == 1:
                        if not np.allclose(audio2[i, k], audio2[i, k + 1], atol=1e-5):
                            raise ValueError(
                                f"样本{ids[i]} 槽{k}/{k+1} 共享时间窗但音频不同"
                            )
                    if vm2[i, k] == 1 and vm2[i, k + 1] == 1:
                        if not np.allclose(vision2[i, k], vision2[i, k + 1], atol=1e-5):
                            raise ValueError(
                                f"样本{ids[i]} 槽{k}/{k+1} 共享时间窗但视觉不同"
                            )

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
        if len(table) != n * (1 if modality == "text" else 50):
            raise ValueError(f"{modality}质量记录数错误")
        if modality in ("audio", "vision") and "support_mask" not in table.columns:
            raise ValueError(f"{modality}质量表缺少 support_mask")

    report = {
        "samples": n,
        "text_shape": list(text.shape),
        "audio_shape": list(audio.shape),
        "vision_shape": list(vision.shape),
        "scheme": "wordpiece50_mfa_shared_word_time_av_weighted_mean",
        "passed": True,
        "scope": "结构与输出一致性；同词子词共享时间窗则AV应相同；不等于人工质量判定准确率",
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
    parser.add_argument("--directory", default="problem1_v2/results")
    print(verify(parser.parse_args().directory))
