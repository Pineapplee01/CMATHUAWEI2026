#!/usr/bin/env python3
# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：GPT-6 Astra
# 版本/型号：GPT-6 Astra（gpt-6-astra）
# 开发机构/公司：OpenAI
# 版本颁布日期：2026年9月3日
# 发布信息来源：https://openai.com/index/safety-overview-gpt-6-astra/

from __future__ import annotations
import shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

PKG = Path(__file__).resolve().parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))
from common import (  # noqa: E402
    COVAREP_NAMES, VISION_NAMES, aggregate, audio_source_covarep, audio_track,
    batch_covarep_mats, bert_words, cleanup_tmps, ensure_dir, install_covarep_mat,
    standardize_tav, media_info, mfa_cfg, parse_extract_args, setup_run, sid_stem,
    view50, vision_source_openface, word_alignment, word_audio_quality,
)

def main() -> int:
    args = parse_extract_args()
    cfg, out, samples = setup_run(args, "config_b.json")
    L, device = int(cfg["L"]), torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(cfg["paths"]["bert"], use_fast=True, local_files_only=True)
    enc = AutoModel.from_pretrained(cfg["paths"]["bert"], local_files_only=True).to(device)
    enc.eval()
    root, work_root = Path(cfg["paths"]["video_root"]), ensure_dir(out / "_tmp_mfa")
    wav_batch, align_cfg = ensure_dir(work_root / "wav_batch"), mfa_cfg(cfg, with_reuse=False)
    hop = float(cfg["audio"].get("covarep_hop_sec", 0.01))
    prepared, all_rms = [], []
    for _, row in samples.iterrows():
        sid, video = str(row["sample_id"]), root / str(row["rel_path"])
        work = ensure_dir(work_root / sid_stem(sid))
        try:
            info = media_info(video)
            audio, status = audio_track(video, work, info)
        except Exception:
            info = {"duration": float(row.get("container_duration_sec") or 0.0)}
            audio, status = None, {"audio_status": "decode_failed", "audio_feature_available": False, "audio_offset": 0.0}
        text = "" if pd.isna(row["text"]) else str(row["text"])
        words, intervals, method, _, failure = word_alignment(text, audio, status, info["duration"], work, align_cfg)
        if audio is not None and (work / "speech.wav").exists():
            shutil.copyfile(work / "speech.wav", wav_batch / f"{work.name}.wav")
            install_covarep_mat(sid, wav_batch / f"{work.name}.mat")
            all_rms.append(float(np.sqrt(np.mean(np.square(audio[0].astype(np.float64))) + 1e-20)))
        prepared.append({"sid": sid, "video": video, "work": work, "info": info, "audio": audio, "status": status,
                         "words": words, "intervals": np.asarray(intervals, dtype=np.float64),
                         "method": method, "failure": failure, "text": text})
    need_wavs = [p for p in wav_batch.glob("*.wav") if not p.with_suffix(".mat").is_file()]
    batch_covarep_mats(cfg, need_wavs, work_root / "wav_pending", dest_dir=wav_batch)
    tau_a = float(np.percentile(all_rms, float(cfg["audio"]["tau_a_percentile"]))) if all_rms else 0.0
    ids, t_list, a_list, v_list = [], [], [], []
    tm_list, am_list, vm_list, support_list, interval_list = [], [], [], [], []
    zero_view = {"text": np.zeros((L, 768), dtype=np.float32), "audio": np.zeros((L, 74), dtype=np.float32),
                 "vision": np.zeros((L, 35), dtype=np.float32)}
    for item in prepared:
        sid, words = item["sid"], item["words"]
        intervals = np.asarray(item["intervals"], dtype=np.float64)
        K = len(words)
        aligned_ok = item["method"] == "mfa" and len(intervals) > 0 and len(intervals) == K
        if not aligned_ok:
            view, viewmask, times = zero_view, np.zeros((3, L), dtype=bool), np.zeros((L, 2), dtype=np.float32)
            support = np.zeros(L, dtype=np.int8)
        else:
            tw = bert_words(words, tok, enc, device)
            mat = wav_batch / f"{item['work'].name}.mat"
            ai, av, aq, _ = audio_source_covarep(mat, hop, float(item["status"].get("audio_offset", 0.0)),
                                                duration=float(item["info"]["duration"]), cfg=cfg)
            a_feat, a_obs = aggregate(intervals, ai, av, aq if len(aq) else np.zeros(0))
            sig = None if item["audio"] is None else item["audio"][0]
            sr = int(cfg["audio"]["sample_rate"]) if item["audio"] is None else int(item["audio"][1])
            a_q = word_audio_quality(sig, sr, intervals, tau_a, int(cfg["audio"]["k_min"]),
                                    float(cfg["audio"]["frame_ms"]), float(cfg["audio"]["hop_ms"]),
                                    float(cfg["audio"]["frame_energy_floor"]))
            a_mask = (a_obs.astype(np.float64) * a_q) > 0
            a_feat = a_feat * a_mask[:, None]
            vi, vv, vq, _ = vision_source_openface(item["video"], item["work"], float(item["info"]["duration"]), cfg, sample_id=sid)
            v_feat, v_obs = aggregate(intervals, vi, vv, vq if len(vq) else np.zeros(0))
            rho = np.zeros(K, dtype=np.float64)
            if len(vi):
                for k, (ws, we) in enumerate(intervals):
                    ov = np.maximum(0.0, np.minimum(we, vi[:, 1]) - np.maximum(ws, vi[:, 0]))
                    wsum = float(ov.sum())
                    if wsum > 0:
                        rho[k] = float((ov * vq).sum() / wsum)
            v_mask = v_obs & (rho >= float(cfg["vision"]["tau_v"]))
            v_feat = v_feat * v_mask[:, None]
            features = {"text": tw, "audio": a_feat, "vision": v_feat}
            masks = np.stack([np.ones(K, dtype=bool), a_mask, v_mask])
            view, viewmask, times = view50(intervals, features, masks)
            support = np.zeros(L, dtype=np.int8)
            support[: min(K, L)] = 1
        ids.append(sid); t_list.append(view["text"]); a_list.append(view["audio"]); v_list.append(view["vision"])
        tm_list.append(viewmask[0].astype(np.int8)); am_list.append(viewmask[1].astype(np.int8))
        vm_list.append(viewmask[2].astype(np.int8)); support_list.append(support); interval_list.append(times)
    text_f, audio_f, vision_f = np.stack(t_list).astype(np.float32), np.stack(a_list).astype(np.float32), np.stack(v_list).astype(np.float32)
    tmask, amask, vmask = np.stack(tm_list), np.stack(am_list), np.stack(vm_list)
    support = np.stack(support_list)
    text_f *= tmask[..., None] * support[..., None]
    audio_f *= amask[..., None] * support[..., None]
    vision_f *= vmask[..., None] * support[..., None]
    text_f, audio_f, vision_f = standardize_tav(text_f, tmask, audio_f, amask, vision_f, vmask,
                                                t_support=support, a_support=support, v_support=support)
    # 与 A/C 一致：词时间写入 text_intervals，不再单独 intervals.npz
    np.savez_compressed(out / "masks.npz", sample_id=np.array(ids), text_mask=tmask, audio_mask=amask, vision_mask=vmask,
                        audio_support=support, vision_support=support, word_support=support)
    np.savez_compressed(out / "text_features.npz", sample_id=np.array(ids), text=text_f, text_mask=tmask,
                        word_support=support, text_intervals=np.stack(interval_list).astype(np.float32))
    np.savez_compressed(out / "audio_features.npz", sample_id=np.array(ids), audio=audio_f,
                        feature_names=np.array(COVAREP_NAMES), audio_mask=amask, audio_support=support)
    np.savez_compressed(out / "vision_features.npz", sample_id=np.array(ids), vision=vision_f,
                        feature_names=np.array(VISION_NAMES), vision_mask=vmask, vision_support=support)
    cleanup_tmps(out, [work_root], args.keep_tmp)
    print(f"done method_b MFA n={len(ids)} text={text_f.shape} audio={audio_f.shape} vision={vision_f.shape}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
