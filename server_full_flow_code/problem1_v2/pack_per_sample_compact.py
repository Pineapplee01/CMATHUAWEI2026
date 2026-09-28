#!/usr/bin/env python3
"""从 problem1_v2 批结果整理逐样本包（与 v3 compact 同结构；含 CLS/SEP）。"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from problem1_v3.pack_per_sample_compact import (
    _anomaly_from_failure,
    _copy_if,
    _sanitize_text,
    _split_id,
)

TYPICAL_SID = "-9y-fZ3swSY$_$0"
TYPICAL_SAFE = "-9y-fZ3swSY__0"


def _ensure_typical_v2(src: Path, support_dir: Path, deliver_root: Path) -> None:
    """典型样本：完整 WordPiece 词元序列 + 对应时间（无时间为空；同词可重复）。"""
    from transformers import BertTokenizerFast

    tok = BertTokenizerFast.from_pretrained(
        str(Path(__file__).resolve().parents[1] / "AAAmodel" / "bert-base-uncased")
    )
    with np.load(src / "text_features.npz", allow_pickle=False) as z:
        ids = [str(x) for x in z["sample_id"]]
        input_ids_all = z["input_ids"]
        tmask_all = z["text_mask"]
        ttime_all = z["text_time_mask"]
        intervals_all = z["text_intervals"]

    def rows_for(sid: str, only_timed: bool = False) -> list[dict]:
        """完整词元展示：保留全部有效槽（含 CLS/SEP/标点）；PAD 不写。"""
        i = ids.index(sid)
        out_rows: list[dict] = []
        for k in range(50):
            if int(tmask_all[i, k]) != 1:
                continue
            timed = int(ttime_all[i, k]) == 1
            if only_timed and not timed:
                continue
            tid = int(input_ids_all[i, k])
            t0 = float(intervals_all[i, k, 0])
            t1 = float(intervals_all[i, k, 1])
            out_rows.append(
                {
                    "token_id": k,
                    "wordpiece": tok.convert_ids_to_tokens(tid),
                    "t0": round(t0, 4) if timed else "",
                    "t1": round(t1, 4) if timed else "",
                    "has_time": int(timed),
                }
            )
        return out_rows

    # 主表：完整 WordPiece 词元序列（与附件2槽位一致）
    csv_name = f"示例_词元时间对应_{TYPICAL_SAFE}.csv"
    df = pd.DataFrame(rows_for(TYPICAL_SID, only_timed=False))
    df.to_csv(support_dir / csv_name, index=False, encoding="utf-8-sig")
    df.to_csv(
        support_dir / f"示例_WordPiece时间对应_{TYPICAL_SAFE}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    alt_sid, alt_safe = "-3g5yACwYnA$_$3", "-3g5yACwYnA__3"
    if alt_sid in ids:
        pd.DataFrame(rows_for(alt_sid, only_timed=False)).to_csv(
            support_dir / f"示例_词元时间对应_{alt_safe}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(rows_for(alt_sid, only_timed=False)).to_csv(
            support_dir / f"示例_WordPiece时间对应_{alt_safe}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    typ = support_dir / "典型样本_-9y-fZ3swSY_0"
    typ.mkdir(parents=True, exist_ok=True)
    src_typ = deliver_root / "典型样本帧_-9y-fZ3swSY_0"
    for name in ("audio_alignment_check.txt", "audio_energy_vs_mfa_words.png"):
        _copy_if(src_typ / name, typ / name)
    (typ / "README.txt").write_text(
        f"典型样本 {TYPICAL_SID}（v2 WordPiece–MFA 词时共享）\n"
        f"- {csv_name}：完整 WordPiece 词元序列（含 [CLS]/[SEP]/标点）\n"
        f"  列 = token_id, wordpiece, t0, t1, has_time\n"
        f"  has_time=1 时 [t0,t1) 为 MFA 整词窗；同词多子词时间可相同\n"
        f"  has_time=0（CLS/SEP/无对齐标点）时间列为空\n"
        f"- 示例_词元时间对应_-3g5yACwYnA__3.csv：含 ## 子词，时间重复更直观\n"
        f"- audio_energy_vs_mfa_words.png / audio_alignment_check.txt："
        f"MFA 词区间与能量对照（与 v3 共用 TextGrid）\n",
        encoding="utf-8",
    )


def pack_v2(src: Path, labels: Path, out: Path, deliver_root: Path, make_zip: bool = True) -> Path:
    out = Path(out)
    if out.exists():
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

    tq = pd.read_csv(src / "text_quality.csv")
    tq_by = {str(r.sample_id): r for r in tq.itertuples()}

    man_path = src / "token_time_manifest.json"
    manifests = json.loads(man_path.read_text(encoding="utf-8")) if man_path.is_file() else []
    man_by = {str(m["sample_id"]): m for m in manifests}
    # 脱敏台账（去掉大列表字段）
    man_pub = []
    for m in manifests:
        fail = _anomaly_from_failure(m.get("alignment_failure", ""))
        man_pub.append(
            {
                "sample_id": m["sample_id"],
                "alignment": m.get("alignment", ""),
                "alignment_confidence": m.get("alignment_confidence", ""),
                "alignment_failure": fail,
                "n_mfa_words": m.get("n_mfa_words", 0),
                "n_timed_tokens": m.get("n_timed_tokens", 0),
                "note": _sanitize_text(m.get("note", "")),
            }
        )
    pd.DataFrame(man_pub).to_csv(
        support_dir / "token_time_manifest.csv", index=False, encoding="utf-8-sig"
    )

    with np.load(src / "text_features.npz", allow_pickle=False) as z:
        ids = [str(x) for x in z["sample_id"]]
        text = z["text"].astype(np.float32)
        text_mask = z["text_mask"].astype(np.int8)
        text_time_mask = z["text_time_mask"].astype(np.int8)
        text_content_mask = z["text_content_mask"].astype(np.int8)
        intervals = z["text_intervals"].astype(np.float32)
        input_ids = z["input_ids"].astype(np.int32)
    with np.load(src / "audio_features.npz", allow_pickle=False) as z:
        audio = z["audio"].astype(np.float32)
        audio_mask = z["audio_mask"].astype(np.int8)
        audio_support = z["audio_support"].astype(np.int8) if "audio_support" in z.files else audio_mask
    with np.load(src / "vision_features.npz", allow_pickle=False) as z:
        vision = z["vision"].astype(np.float32)
        vision_mask = z["vision_mask"].astype(np.int8)
        vision_support = (
            z["vision_support"].astype(np.int8) if "vision_support" in z.files else vision_mask
        )

    np.savez_compressed(
        support_dir / "text_intervals.npz",
        sample_id=np.array(ids),
        text_intervals=intervals,
        text_time_mask=text_time_mask,
        text_content_mask=text_content_mask,
    )

    rows = []
    for i, sid in enumerate(ids):
        full_id, video_id, clip_id = _split_id(sid)
        tqr = tq_by.get(sid)
        m = man_by.get(sid, {})
        duration = float(getattr(tqr, "duration_sec", 0.0) or 0.0) if tqr is not None else 0.0
        align = str(m.get("alignment", getattr(tqr, "alignment", "unknown") if tqr else "unknown"))
        fail = str(m.get("alignment_failure", getattr(tqr, "alignment_failure", "") if tqr else "") or "")
        n_words = int(m.get("n_mfa_words", getattr(tqr, "n_mfa_words", 0) if tqr else 0) or 0)
        n_timed = int(m.get("n_timed_tokens", getattr(tqr, "n_timed_tokens", 0) if tqr else 0) or 0)

        vt, va, vv = int(text_mask[i].sum()), int(audio_mask[i].sum()), int(vision_mask[i].sum())
        valid_length = int(text_time_mask[i].sum())
        anomaly = _anomaly_from_failure(fail)
        if fail and fail.lower() not in ("nan", "none"):
            extraction_log = f"mfa_failed:{anomaly or 'unknown'}"
        elif align != "mfa":
            extraction_log = f"alignment={align}"
            anomaly = anomaly or extraction_log
        else:
            extraction_log = f"mfa_ok timed_tokens={n_timed} words={n_words}"

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
            "audio_support": audio_support[i],
            "vision_support": vision_support[i],
            "text_time_mask": text_time_mask[i],
            "text_content_mask": text_content_mask[i],
            "input_ids": input_ids[i],
            "start_time": intervals[i, :, 0],
            "end_time": intervals[i, :, 1],
            "valid_length": np.int32(valid_length),
            "valid_length_tav": np.asarray([vt, va, vv], dtype=np.int32),
            "alignment_quality": align,
            "extraction_log": extraction_log,
            "alignment_method": "wordpiece_mfa_shared",
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
                "对齐方式": "WordPiece–MFA词时共享",
                "异常说明": anomaly,
                "输出文件": path.name,
                "输出文件校验值": digest,
                "文件字节": path.stat().st_size,
            }
        )

    summary_path = out / "全量100样本_题目结构汇总_方法A.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8-sig")

    fig_src = src.parent / "figures" / "results"
    _copy_if(fig_src / "valid_fraction.pdf", figures_dir / "valid_fraction.pdf")
    _copy_if(fig_src / "quality_masks.pdf", figures_dir / "quality_masks.pdf")

    # 典型样本（与 v3 supporting 同构）：词元时间表 + MFA 能量对照
    _ensure_typical_v2(src, support_dir, deliver_root)

    readme = out / "README.txt"
    readme.write_text(
        "问题一 v2（WordPiece–MFA 词时共享）交付包（压缩、无本机绝对路径）\n\n"
        "目录说明（均为相对本文件夹）：\n"
        "  samples/*.npz\n"
        "    每样本：id, video_id, clip_id, raw_text, duration,\n"
        "    text[50,768], audio[50,74], vision[50,35],\n"
        "    text/audio/vision_mask[50], audio/vision_support[50],\n"
        "    text_time_mask/text_content_mask[50], input_ids[50]（含CLS/SEP）,\n"
        "    start_time/end_time[50]（同词子词共享 MFA 整词窗）,\n"
        "    valid_length(=timed), alignment_quality, extraction_log\n"
        "  全量100样本_题目结构汇总_方法A.csv\n"
        "  supporting/\n"
        "    token_time_manifest.csv, text_intervals.npz,\n"
        "    示例_WordPiece时间对应_-9y-fZ3swSY__0.csv（及 _-3g5yACwYnA__3 示重复时间）,\n"
        "    典型样本_-9y-fZ3swSY_0/（能量对照图与核验说明）\n"
        "  figures/\n"
        "    valid_fraction.pdf, quality_masks.pdf\n"
        "  samples_v2_compact.zip\n\n"
        "读取示例：\n"
        "  import numpy as np\n"
        "  z = np.load('samples/-9y-fZ3swSY__0.npz')\n"
        "  print(z['id'], z['input_ids'][:3], z['text'].shape)\n",
        encoding="utf-8",
    )

    meta = {
        "scheme": "wordpiece_mfa_shared_per_sample_compact",
        "source_relative": "problem1_v2/results",
        "n_samples": len(ids),
        "bytes_samples": int(sum(p.stat().st_size for p in samples_dir.glob("*.npz"))),
        "dtype": {"features": "float32", "masks": "int8", "times": "float32", "input_ids": "int32"},
        "note": "含 [CLS]/[SEP] input_ids；路径均为交付包内相对路径；异常说明已脱敏",
    }

    if make_zip:
        zip_path = out / "samples_v2_compact.zip"
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
    parser.add_argument("--src", type=Path, default=root / "problem1_v2/results")
    parser.add_argument("--labels", type=Path, default=root / "AAAdata/Appendix_1/label-100.xlsx")
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "deliverables/问题一_特征提取与时序对齐/per_sample_v2_compact",
    )
    parser.add_argument(
        "--deliver-root",
        type=Path,
        default=root / "deliverables/问题一_特征提取与时序对齐",
    )
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()
    pack_v2(args.src, args.labels, args.out, args.deliver_root, make_zip=not args.no_zip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
