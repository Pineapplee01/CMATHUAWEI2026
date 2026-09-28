#!/usr/bin/env python3
"""问题1 v2：WordPiece 50 槽 + MFA 词时间（同词子词共享）+ 音视按词元窗重叠加权。

流程：
1. MFA 得到口语词区间 I_m=[s,e)；
2. BERT WordPiece 槽：内容子词经字符重叠继承所属整词时间（同词多子词共享同一窗）；
3. 音/视源帧 → common_hop → 对每个有时间的词元窗做交叠时长×质量加权均值。
因此同一口语词拆成的相邻子词槽，音（及视觉）特征相同，属预期而非随机撞车。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG.parent))

from problem1_v2.common import ensure_dir, load_config, load_samples  # noqa: E402
from problem1_v2.audio_quality import run_audio_on_token_intervals
from problem1_v2.text_quality import run_text_with_mfa
from problem1_v2.vision_quality import run_vision_on_token_intervals
from problem1_v2.normalize import standardize_features


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", nargs="?", default="extract", choices=["extract"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--keep-tmp", action="store_true")
    parser.add_argument(
        "--standardize",
        "--standardize-audio",
        dest="standardize",
        action="store_true",
        help="对文本BERT/音频/视觉：有效位估参→标准化→无效/无支持置零",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit 必须为正数")
        cfg["paths"]["output"] += f"_smoke_{args.limit}"
    out = ensure_dir(Path(cfg["paths"]["output"]))
    samples = load_samples(cfg)
    if args.limit is not None:
        samples = samples.head(int(args.limit))

    (
        tdf,
        tmask,
        tsupport,
        t_time_mask,
        t_content_mask,
        tids,
        tfeat,
        input_ids,
        token_intervals,
        prepared,
        manifests,
    ) = run_text_with_mfa(cfg, samples)

    adf, amask, asupport, aids, afeat, anames = run_audio_on_token_intervals(
        cfg, samples, tids, token_intervals, t_time_mask, prepared
    )
    vdf, vmask, vsupport, vids, vfeat, vnames = run_vision_on_token_intervals(
        cfg, tids, token_intervals, t_time_mask, prepared
    )
    assert tids == aids == vids

    # 同词子词：时间窗相同 → 聚合后 AV 应相同（在同 mask 下）
    # 由 MFA 桥接保证；此处不强制改写特征。

    if args.standardize:
        tfeat, tstats = standardize_features(tfeat, tmask, tsupport)
        afeat, astats = standardize_features(afeat, amask, asupport)
        vfeat, vstats = standardize_features(vfeat, vmask, vsupport)
        np.savez_compressed(
            out / "norm_stats.npz",
            text_mean=tstats["mean"],
            text_std=tstats["std"],
            audio_mean=astats["mean"],
            audio_std=astats["std"],
            vision_mean=vstats["mean"],
            vision_std=vstats["std"],
            audio_feature_names=np.array(anames),
            vision_feature_names=np.array(vnames),
        )

    tdf.to_csv(out / "text_quality.csv", index=False)
    adf.to_csv(out / "audio_quality.csv", index=False)
    vdf.to_csv(out / "vision_quality.csv", index=False)
    (out / "token_time_manifest.json").write_text(
        json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    intervals_arr = np.stack([np.asarray(x, dtype=np.float32) for x in token_intervals])

    np.savez_compressed(
        out / "masks.npz",
        sample_id=np.array(tids),
        text_mask=tmask.astype(np.int8),
        audio_mask=amask.astype(np.int8),
        vision_mask=vmask.astype(np.int8),
        text_support=tsupport.astype(np.int8),
        audio_support=asupport.astype(np.int8),
        vision_support=vsupport.astype(np.int8),
        text_time_mask=t_time_mask.astype(np.int8),
        text_content_mask=t_content_mask.astype(np.int8),  # 排除 CLS/SEP/PAD，对齐附件2 mT 语义
    )
    np.savez_compressed(
        out / "text_features.npz",
        sample_id=np.array(tids),
        text=tfeat.astype(np.float32),
        input_ids=input_ids.astype(np.int32),  # 与附件2 I 一致：含 [CLS]/[SEP]
        text_mask=tmask.astype(np.int8),
        text_support=tsupport.astype(np.int8),
        text_time_mask=t_time_mask.astype(np.int8),
        text_content_mask=t_content_mask.astype(np.int8),
        text_intervals=intervals_arr,
    )
    np.savez_compressed(
        out / "audio_features.npz",
        sample_id=np.array(aids),
        audio=afeat.astype(np.float32),
        feature_names=np.array(anames),
        audio_mask=amask.astype(np.int8),
        audio_support=asupport.astype(np.int8),
    )
    np.savez_compressed(
        out / "vision_features.npz",
        sample_id=np.array(vids),
        vision=vfeat.astype(np.float32),
        feature_names=np.array(vnames),
        vision_mask=vmask.astype(np.int8),
        vision_support=vsupport.astype(np.int8),
    )
    n_mfa = int(sum(1 for m in manifests if m["alignment"] == "mfa"))
    (out / "extraction_meta.json").write_text(
        json.dumps(
            {
                "scheme": "wordpiece50_mfa_shared_word_time_av_weighted_mean",
                "L": int(cfg["L"]),
                "common_hop_sec": float((cfg.get("av_sync") or {}).get("common_hop_sec", 0.04)),
                "n": len(tids),
                "n_mfa_ok": n_mfa,
                "note": (
                    "MFA整词时间→同词WordPiece子词共享时间窗→"
                    "音/视对每词元窗交叠时长×质量加权均值；"
                    "同词相邻子词AV相同为预期"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if not args.keep_tmp:
        for name in ("_tmp_wav", "_tmp_openface", "_tmp_mfa_text"):
            p = out / name
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)

    from problem1_v2.verify_outputs import verify
    from problem1_v2.visualization import quality_plots

    verify(out)
    try:
        quality_plots(out, PKG / "figures" / out.name)
    except Exception as exc:
        print(f"plots skipped: {exc}")

    print(
        f"done n={len(tids)} mfa_ok={n_mfa}/{len(tids)} "
        f"text={tfeat.shape} audio={afeat.shape} vision={vfeat.shape}"
        + (" [standardized TAV]" if args.standardize else "")
        + " [WP+MFA shared word time]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
