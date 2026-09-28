#!/usr/bin/env python3
"""问题1 v3：MFA 词级对齐 + 三模态特征（词槽，非等分 50 窗）。

音视：原生帧 → common_hop → 聚到 MFA 词区间 I_k。
文本：词级 BERT（词内 WordPiece 均值）；view50 前 50 词截断。
产出：masks / features / intervals / alignment_manifest / *_quality。
对齐不含 [CLS]/[SEP]；槽位 = 转写内容词。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG.parent))

from problem1_v3.common import ensure_dir, load_config, load_samples  # noqa: E402
from problem1_v3.covarep_audio import COVAREP_NAMES, run_covarep_dir
from problem1_v3.extractors import (
    VISION_NAMES,
    audio_source_covarep,
    bert_words,
    vision_source_openface,
    word_audio_quality,
)
from problem1_v3.mfa_alignment import aggregate, view50, view50_metadata, word_alignment
from problem1_v3.mfa_alignment.media import audio_track, media_info
from problem1_v3.normalize import standardize_features
from problem1_v3.shared_cache import install_covarep_mat, publish_covarep_mat


def _mfa_cfg(cfg: dict) -> dict:
    mfa = cfg.get("mfa", {})
    return {
        "mfa_executable": mfa.get("mfa_executable", "mfa"),
        "dictionary": cfg["paths"]["dictionary"],
        "acoustic": cfg["paths"]["acoustic"],
        "fallback_uniform": False,  # 已废弃：失败只记录，不均分
        "batch_alignment": False,
    }


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
        help="音视频/文本BERT：有效位估参→标准化→无效/无词槽再置零",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    L = int(cfg["L"])
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit 必须为正数")
        cfg["paths"]["output"] += f"_smoke_{args.limit}"
    out = ensure_dir(Path(cfg["paths"]["output"]))
    samples = load_samples(cfg)
    if args.limit is not None:
        samples = samples.head(int(args.limit))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(cfg["paths"]["bert"], use_fast=True, local_files_only=True)
    enc = AutoModel.from_pretrained(cfg["paths"]["bert"], local_files_only=True).to(device)
    enc.eval()

    root = Path(cfg["paths"]["video_root"])
    work_root = ensure_dir(out / "_tmp_mfa")
    wav_batch = ensure_dir(work_root / "wav_batch")
    mfa_cfg = _mfa_cfg(cfg)
    hop = float(cfg["audio"].get("covarep_hop_sec", 0.01))

    prepared = []
    all_rms = []
    for _, row in samples.iterrows():
        sid = str(row["sample_id"])
        video = root / str(row["rel_path"])
        work = ensure_dir(work_root / sid.replace("/", "_").replace("$_$", "__"))
        info = media_info(video)
        try:
            audio, status = audio_track(video, work, info)
        except (ValueError, RuntimeError) as exc:
            audio, status = None, {
                "audio_status": "decode_failed",
                "audio_feature_available": False,
                "audio_offset": 0.0,
                "error": str(exc),
            }
        text = "" if pd.isna(row["text"]) else str(row["text"])
        words, intervals, method, confidence, failure = word_alignment(
            text, audio, status, info["duration"], work, mfa_cfg
        )
        # 供批量 COVAREP：与 speech.wav 同内容
        if audio is not None and (work / "speech.wav").exists():
            shutil.copyfile(work / "speech.wav", wav_batch / f"{work.name}.wav")
            # 共享缓存命中则预装 .mat，避免重复 MATLAB
            install_covarep_mat(sid, wav_batch / f"{work.name}.mat")
            sig = audio[0]
            all_rms.append(float(np.sqrt(np.mean(np.square(sig.astype(np.float64))) + 1e-20)))
        prepared.append(
            {
                "sid": sid,
                "video": video,
                "work": work,
                "info": info,
                "audio": audio,
                "status": status,
                "words": words,
                "intervals": np.asarray(intervals, dtype=np.float64),
                "method": method,
                "confidence": confidence,
                "failure": failure,
                "text": text,
            }
        )

    # 仅对缺少 .mat 的 wav 跑 COVAREP
    need_wavs = [
        p for p in wav_batch.glob("*.wav") if not p.with_suffix(".mat").is_file()
    ]
    if need_wavs:
        # 临时子目录只含待处理 wav，避免 COVAREP 重跑已有
        pending = ensure_dir(work_root / "wav_pending")
        for p in list(pending.glob("*")):
            p.unlink()
        for p in need_wavs:
            shutil.copy2(p, pending / p.name)
        run_covarep_dir(cfg, pending)
        for p in pending.glob("*.mat"):
            shutil.copy2(p, wav_batch / p.name)
            # work.name == stem with __
            sid_guess = p.stem.replace("__", "$_$")
            publish_covarep_mat(sid_guess, wav_batch / p.name)
    tau_a = (
        float(np.percentile(all_rms, float(cfg["audio"]["tau_a_percentile"])))
        if all_rms
        else 0.0
    )

    ids, t_list, a_list, v_list = [], [], [], []
    tm_list, am_list, vm_list = [], [], []
    support_list, interval_list = [], []
    manifest_rows, tq_rows, aq_rows, vq_rows = [], [], [], []

    for item in prepared:
        sid = item["sid"]
        words = item["words"]
        intervals = np.asarray(item["intervals"], dtype=np.float64)
        n_transcript = len(words)
        aligned_ok = item["method"] == "mfa" and len(intervals) > 0 and len(intervals) == n_transcript

        if not aligned_ok:
            # MFA 失败：特征全零，台账记录词数与 failure
            view = {
                "text": np.zeros((L, 768), dtype=np.float32),
                "audio": np.zeros((L, 74), dtype=np.float32),
                "vision": np.zeros((L, 35), dtype=np.float32),
            }
            viewmask = np.zeros((3, L), dtype=bool)
            times = np.zeros((L, 2), dtype=np.float32)
            support = np.zeros(L, dtype=np.int8)
            a_q = np.zeros(0, dtype=np.float64)
            rho = np.zeros(0, dtype=np.float64)
            K = 0
        else:
            K = n_transcript
            # 文本：词级 BERT（无 CLS/SEP 槽）
            tw, tmap = bert_words(words, tok, enc, device)
            assert len(tw) == K

            # 音频源 COVAREP → 聚到词区间
            mat = wav_batch / f"{item['work'].name}.mat"
            ai, av, aq, _ = audio_source_covarep(
                mat,
                hop,
                float(item["status"].get("audio_offset", 0.0)),
                duration=float(item["info"]["duration"]),
                cfg=cfg,
            )
            a_feat, a_obs, _ = aggregate(intervals, ai, av, aq if len(aq) else np.zeros(0))
            sig = None if item["audio"] is None else item["audio"][0]
            sr = int(cfg["audio"]["sample_rate"]) if item["audio"] is None else int(item["audio"][1])
            a_q = word_audio_quality(
                sig,
                sr,
                intervals,
                tau_a,
                int(cfg["audio"]["k_min"]),
                float(cfg["audio"]["frame_ms"]),
                float(cfg["audio"]["hop_ms"]),
                float(cfg["audio"]["frame_energy_floor"]),
            )
            a_mask = (a_obs.astype(np.float64) * a_q) > 0
            a_feat = a_feat * a_mask[:, None]

            # 视觉 AU → 词区间
            vi, vv, vq, _ = vision_source_openface(
                item["video"], item["work"], float(item["info"]["duration"]), cfg, sample_id=sid
            )
            v_feat, v_obs, _ = aggregate(intervals, vi, vv, vq if len(vq) else np.zeros(0))
            rho = np.zeros(K, dtype=np.float64)
            if len(vi):
                for k, (ws, we) in enumerate(intervals):
                    ov = np.maximum(0.0, np.minimum(we, vi[:, 1]) - np.maximum(ws, vi[:, 0]))
                    wsum = float(ov.sum())
                    if wsum <= 0:
                        continue
                    rho[k] = float((ov * vq).sum() / wsum)
            v_mask = v_obs & (rho >= float(cfg["vision"]["tau_v"]))
            v_feat = v_feat * v_mask[:, None]

            t_mask = np.ones(K, dtype=bool)
            features = {"text": tw, "audio": a_feat, "vision": v_feat}
            masks = np.stack([t_mask, a_mask, v_mask])
            view, viewmask, times, _ = view50(intervals, features, masks)
            support = np.zeros(L, dtype=np.int8)
            support[: min(K, L)] = 1

        ids.append(sid)
        t_list.append(view["text"])
        a_list.append(view["audio"])
        v_list.append(view["vision"])
        tm_list.append(viewmask[0].astype(np.int8))
        am_list.append(viewmask[1].astype(np.int8))
        vm_list.append(viewmask[2].astype(np.int8))
        support_list.append(support)
        interval_list.append(times)

        meta = view50_metadata(K)
        manifest_rows.append(
            {
                "sample_id": sid,
                "n_words": n_transcript,
                "n_aligned_words": K,
                "alignment": item["method"],
                "timing_confidence": item["confidence"],
                "alignment_failure": item["failure"],
                "duration": float(item["info"]["duration"]),
                "audio_status": item["status"].get("audio_status"),
                **{k: meta[k] for k in ("view50_truncated", "view50_retained_units", "view50_dropped_units")},
            }
        )
        for k in range(L):
            has = k < K
            tq_rows.append(
                {
                    "sample_id": sid,
                    "word_id": k,
                    "word": words[k] if has else "",
                    "t0": float(intervals[k, 0]) if has else 0.0,
                    "t1": float(intervals[k, 1]) if has else 0.0,
                    "valid_mask": int(viewmask[0, k]),
                    "support_mask": int(has),
                }
            )
            aq_rows.append(
                {
                    "sample_id": sid,
                    "word_id": k,
                    "t0": float(intervals[k, 0]) if has else 0.0,
                    "t1": float(intervals[k, 1]) if has else 0.0,
                    "valid_mask": int(viewmask[1, k]),
                    "support_mask": int(has),
                    "audio_quality_ok": int(a_q[k]) if has else 0,
                }
            )
            vq_rows.append(
                {
                    "sample_id": sid,
                    "word_id": k,
                    "t0": float(intervals[k, 0]) if has else 0.0,
                    "t1": float(intervals[k, 1]) if has else 0.0,
                    "valid_mask": int(viewmask[2, k]),
                    "support_mask": int(has),
                    "valid_ratio": float(rho[k]) if has else 0.0,
                }
            )

    text_f = np.stack(t_list).astype(np.float32)
    audio_f = np.stack(a_list).astype(np.float32)
    vision_f = np.stack(v_list).astype(np.float32)
    tmask = np.stack(tm_list)
    amask = np.stack(am_list)
    vmask = np.stack(vm_list)
    support = np.stack(support_list)
    # 无词槽 / 无效处置零
    text_f *= tmask[..., None]
    text_f *= support[..., None]
    audio_f *= amask[..., None]
    audio_f *= support[..., None]
    vision_f *= vmask[..., None]
    vision_f *= support[..., None]

    if args.standardize:
        text_f, tstats = standardize_features(text_f, tmask, support)
        audio_f, astats = standardize_features(audio_f, amask, support)
        vision_f, vstats = standardize_features(vision_f, vmask, support)
        np.savez_compressed(
            out / "norm_stats.npz",
            text_mean=tstats["mean"],
            text_std=tstats["std"],
            audio_mean=astats["mean"],
            audio_std=astats["std"],
            vision_mean=vstats["mean"],
            vision_std=vstats["std"],
            audio_feature_names=np.array(COVAREP_NAMES),
            vision_feature_names=np.array(VISION_NAMES),
        )

    pd.DataFrame(manifest_rows).to_csv(out / "alignment_manifest.csv", index=False)
    pd.DataFrame(tq_rows).to_csv(out / "text_quality.csv", index=False)
    pd.DataFrame(aq_rows).to_csv(out / "audio_quality.csv", index=False)
    pd.DataFrame(vq_rows).to_csv(out / "vision_quality.csv", index=False)

    np.savez_compressed(
        out / "masks.npz",
        sample_id=np.array(ids),
        text_mask=tmask,
        audio_mask=amask,
        vision_mask=vmask,
        audio_support=support,
        vision_support=support,
        word_support=support,
    )
    np.savez_compressed(
        out / "intervals.npz",
        sample_id=np.array(ids),
        intervals=np.stack(interval_list).astype(np.float32),
        word_support=support,
    )
    np.savez_compressed(
        out / "text_features.npz",
        sample_id=np.array(ids),
        text=text_f,
        text_mask=tmask,
        word_support=support,
    )
    np.savez_compressed(
        out / "audio_features.npz",
        sample_id=np.array(ids),
        audio=audio_f,
        feature_names=np.array(COVAREP_NAMES),
        audio_mask=amask,
        audio_support=support,
    )
    np.savez_compressed(
        out / "vision_features.npz",
        sample_id=np.array(ids),
        vision=vision_f,
        feature_names=np.array(VISION_NAMES),
        vision_mask=vmask,
        vision_support=support,
    )
    (out / "extraction_meta.json").write_text(
        json.dumps(
            {
                "alignment": "mfa_word",
                "L": L,
                "special_tokens_in_alignment": False,
                "text_dim": 768,
                "audio_dim": 74,
                "vision_dim": 35,
                "tau_a": tau_a,
                "n": len(ids),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if not args.keep_tmp:
        shutil.rmtree(work_root, ignore_errors=True)

    from problem1_v3.verify_outputs import verify
    from problem1_v3.visualization import quality_plots

    verify(out)
    try:
        quality_plots(out, PKG / "figures" / out.name)
    except Exception as exc:
        print(f"plots skipped: {exc}")

    print(
        f"done MFA n={len(ids)} text={text_f.shape} audio={audio_f.shape} vision={vision_f.shape}"
        + (" [standardized TAV]" if args.standardize else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
