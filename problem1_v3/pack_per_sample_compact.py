#!/usr/bin/env python3
"""从 problem1_v3 批结果整理题目 4.8 逐样本包（小体积、无本机绝对路径）。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _split_id(sample_id: str) -> tuple[str, str, int]:
    if "$_$" in sample_id:
        video_id, clip = sample_id.split("$_$", 1)
    elif "__" in sample_id:
        video_id, clip = sample_id.split("__", 1)
    else:
        video_id, clip = sample_id, "0"
    return sample_id, video_id, int(clip)


def _sanitize_text(text: str) -> str:
    s = str(text or "")
    if not s or s.lower() in {"nan", "none"}:
        return ""
    s = re.sub(r"/user_home/[^/\s]+/", "", s)
    s = re.sub(r"[A-Za-z]:\\Users\\[^\\\s]+\\", "", s)
    s = re.sub(r"(?:^|[\s'\"(=])(?:\./|/)?(?:[\w.-]+/)*CPMCM/", " ", s)
    s = re.sub(r"日志\s+\S+/([\w.-]+\.log)", r"日志 \1", s)
    s = re.sub(r"命令失败\s+mfa[，,]\s*日志\s+\S+", "命令失败 mfa", s)
    s = re.sub(r"\s+", " ", s).strip(" ,;|")
    return s[:240]


def _anomaly_from_failure(fail: str) -> str:
    raw = str(fail or "")
    f = raw.lower()
    if not raw or f in {"nan", "none"}:
        return ""
    if "silent" in f or "静音" in raw:
        return "静音轨，无法强制对齐"
    if "oov" in f or "noalignments" in f.replace("_", "").replace(" ", ""):
        return "词典OOV或无对齐结果"
    if "命令失败" in raw or "failed" in f or "error" in f:
        return "MFA对齐失败"
    return _sanitize_text(raw)[:200]


def _copy_if(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def pack_v3(src: Path, labels: Path, out: Path, deliver_root: Path, make_zip: bool = True) -> Path:
    out = Path(out)
    if out.exists():
        # 仅清理本包内容，避免误删父目录
        for child in list(out.iterdir()):
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    samples_dir = out / "samples"
    support_dir = out / "supporting"
    figures_dir = out / "figures"
    samples_dir.mkdir(parents=True, exist_ok=True)
    support_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_excel(labels)
    labels_df["clip_id"] = labels_df["clip_id"].astype(int)
    labels_df["sample_id"] = (
        labels_df["video_id"].astype(str) + "$_$" + labels_df["clip_id"].astype(str)
    )
    text_map = {
        str(r.sample_id): ("" if pd.isna(r.text) else str(r.text))
        for r in labels_df.itertuples()
    }

    man = pd.read_csv(src / "alignment_manifest.csv")
    # 写出脱敏台账
    man_pub = man.copy()
    if "alignment_failure" in man_pub.columns:
        man_pub["alignment_failure"] = man_pub["alignment_failure"].map(
            lambda x: _anomaly_from_failure(x) if pd.notna(x) and str(x).strip() else ""
        )
    man_pub.to_csv(support_dir / "alignment_manifest.csv", index=False, encoding="utf-8-sig")

    with np.load(src / "text_features.npz", allow_pickle=False) as z:
        ids = [str(x) for x in z["sample_id"]]
        text = z["text"].astype(np.float32)
        text_mask = z["text_mask"].astype(np.int8)
    with np.load(src / "audio_features.npz", allow_pickle=False) as z:
        audio = z["audio"].astype(np.float32)
        audio_mask = z["audio_mask"].astype(np.int8)
    with np.load(src / "vision_features.npz", allow_pickle=False) as z:
        vision = z["vision"].astype(np.float32)
        vision_mask = z["vision_mask"].astype(np.int8)
    with np.load(src / "intervals.npz", allow_pickle=False) as z:
        intervals = z["intervals"].astype(np.float32)
        word_support = z["word_support"].astype(np.int8)
    shutil.copy2(src / "intervals.npz", support_dir / "intervals.npz")

    man_by = {str(r.sample_id): r for r in man.itertuples()}
    rows = []
    for i, sid in enumerate(ids):
        full_id, video_id, clip_id = _split_id(sid)
        m = man_by.get(sid)
        duration = float(getattr(m, "duration", 0.0) or 0.0) if m is not None else 0.0
        align = str(getattr(m, "alignment", "unknown")) if m is not None else "unknown"
        conf = str(getattr(m, "timing_confidence", "")) if m is not None else ""
        fail = str(getattr(m, "alignment_failure", "") or "") if m is not None else ""
        n_words = int(getattr(m, "n_words", 0) or 0) if m is not None else 0
        n_aligned = int(getattr(m, "n_aligned_words", 0) or 0) if m is not None else 0

        vt, va, vv = int(text_mask[i].sum()), int(audio_mask[i].sum()), int(vision_mask[i].sum())
        valid_length = int(word_support[i].sum())
        anomaly = _anomaly_from_failure(fail)
        if fail and fail.lower() not in ("nan", "none"):
            extraction_log = f"mfa_failed:{anomaly or 'unknown'}"
        elif align != "mfa":
            extraction_log = f"alignment={align}"
            anomaly = anomaly or extraction_log
        else:
            extraction_log = f"mfa_ok words={n_aligned}/{n_words}"

        alignment_quality = conf if conf and conf.lower() not in ("nan", "none") else align
        payload = {
            "id": full_id,
            "video_id": video_id,
            "clip_id": np.int32(clip_id),
            "raw_text": text_map.get(sid, ""),
            "duration": np.float32(duration),
            "text": text[i],
            "audio": audio[i],
            "vision": vision[i],
            "text_mask": text_mask[i],
            "audio_mask": audio_mask[i],
            "vision_mask": vision_mask[i],
            "start_time": intervals[i, :, 0],
            "end_time": intervals[i, :, 1],
            "valid_length": np.int32(valid_length),
            "valid_length_tav": np.asarray([vt, va, vv], dtype=np.int32),
            "alignment_quality": str(alignment_quality),
            "extraction_log": extraction_log,
            "alignment_method": "mfa_word",
        }
        safe = sid.replace("$_$", "__").replace("/", "_")
        path = samples_dir / f"{safe}.npz"
        np.savez_compressed(path, **payload)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(
            {
                "样本编号": sid,
                "视频时长": round(duration, 4),
                "有效位置数": valid_length,
                "三模态维度": "50×768 / 50×74 / 50×35",
                "有效比例_T": round(vt / 50.0, 4),
                "有效比例_A": round(va / 50.0, 4),
                "有效比例_V": round(vv / 50.0, 4),
                "对齐方式": "MFA词级",
                "异常说明": anomaly,
                "输出文件": path.name,
                "输出文件校验值": digest,
                "文件字节": path.stat().st_size,
            }
        )

    summary_path = out / "全量100样本_题目结构汇总_方法B.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8-sig")

    # 附属材料：仅保留非冗余项（全量汇总已在包根目录）
    _copy_if(deliver_root / "示例_词级时间对应_-9y-fZ3swSY__0.csv", support_dir / "示例_词级时间对应_-9y-fZ3swSY__0.csv")
    typ = deliver_root / "典型样本帧_-9y-fZ3swSY_0"
    if typ.is_dir():
        for name in ("audio_alignment_check.txt", "audio_energy_vs_mfa_words.png"):
            _copy_if(typ / name, support_dir / "典型样本_-9y-fZ3swSY_0" / name)

    fig_src = src.parent / "figures" / "results"
    _copy_if(fig_src / "valid_fraction.pdf", figures_dir / "valid_fraction.pdf")
    _copy_if(fig_src / "quality_masks.pdf", figures_dir / "quality_masks.pdf")

    readme = out / "README.txt"
    readme.write_text(
        "问题一 v3（MFA 词级）交付包（压缩、无本机绝对路径）\n\n"
        "目录说明（均为相对本文件夹）：\n"
        "  samples/*.npz\n"
        "    每样本：id, video_id, clip_id, raw_text, duration,\n"
        "    text[50,768], audio[50,74], vision[50,35],\n"
        "    text/audio/vision_mask[50], start_time/end_time[50],\n"
        "    valid_length, alignment_quality, extraction_log\n"
        "  全量100样本_题目结构汇总_方法B.csv\n"
        "    样本编号、时长、有效位置、维度、有效比例、对齐方式、异常、校验值\n"
        "  supporting/\n"
        "    alignment_manifest.csv, intervals.npz,\n"
        "    示例_词级时间对应_-9y-fZ3swSY__0.csv, 典型样本能量对照\n"
        "  figures/\n"
        "    valid_fraction.pdf, quality_masks.pdf\n"
        "  samples_v3_compact.zip\n"
        "    上述主要内容打包（便于提交）\n\n"
        "读取示例：\n"
        "  import numpy as np\n"
        "  z = np.load('samples/-9y-fZ3swSY__0.npz')\n"
        "  print(z['id'], z['start_time'][:5], z['text'].shape)\n",
        encoding="utf-8",
    )

    meta = {
        "scheme": "mfa_word_per_sample_compact",
        "source_relative": "problem1_v3/results",
        "n_samples": len(ids),
        "bytes_samples": int(sum(p.stat().st_size for p in samples_dir.glob("*.npz"))),
        "dtype": {"features": "float32", "masks": "int8", "times": "float32"},
        "note": "路径均为交付包内相对路径；异常说明已脱敏，不含本机用户目录",
    }

    if make_zip:
        zip_path = out / "samples_v3_compact.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for p in sorted(samples_dir.glob("*.npz")):
                zf.write(p, arcname=f"samples/{p.name}")
            zf.write(summary_path, arcname=summary_path.name)
            zf.write(readme, arcname=readme.name)
            for p in sorted(support_dir.rglob("*")):
                if p.is_file():
                    zf.write(p, arcname=str(p.relative_to(out)).replace("\\", "/"))
            for p in sorted(figures_dir.glob("*")):
                if p.is_file():
                    zf.write(p, arcname=f"figures/{p.name}")
        meta["bytes_zip"] = int(zip_path.stat().st_size)

    (out / "pack_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("done", out, "samples_bytes", meta["bytes_samples"], "zip", meta.get("bytes_zip"))
    return out


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=root / "problem1_v3/results")
    parser.add_argument("--labels", type=Path, default=root / "AAAdata/Appendix_1/label-100.xlsx")
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "deliverables/问题一_特征提取与时序对齐/per_sample_v3_compact",
    )
    parser.add_argument(
        "--deliver-root",
        type=Path,
        default=root / "deliverables/问题一_特征提取与时序对齐",
    )
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()
    pack_v3(args.src, args.labels, args.out, args.deliver_root, make_zip=not args.no_zip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
