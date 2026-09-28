#!/usr/bin/env python3
# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：GPT-6 Astra
# 版本/型号：GPT-6 Astra（gpt-6-astra）
# 开发机构/公司：OpenAI
# 版本颁布日期：2026年9月3日
# 发布信息来源：https://openai.com/index/safety-overview-gpt-6-astra/

from __future__ import annotations
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import BertModel, BertTokenizerFast

from multimodal_emotion.preprocess.problem1.common import (
    VISION_NAMES, COVAREP_NAMES, aggregate, audio_source_covarep, batch_covarep_mats,
    bridge_wordpiece_to_mfa, cleanup_tmps, encode_bert_offsets, ensure_dir, install_covarep_mat,
    standardize_tav, mfa_cfg, parse_extract_args, prepare_mfa_clip, setup_run, sid_stem,
    support_overlap, vision_source_openface, word_audio_quality,
)

def run_text_with_mfa(cfg, samples: pd.DataFrame):
    L, device = int(cfg["L"]), torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = BertTokenizerFast.from_pretrained(cfg["paths"]["bert"])
    enc = BertModel.from_pretrained(cfg["paths"]["bert"]).to(device)
    enc.eval()
    root, work_root, align_cfg = Path(cfg["paths"]["video_root"]), ensure_dir(Path(cfg["paths"]["output"]) / "_tmp_mfa_text"), mfa_cfg(cfg)
    masks, supports, time_masks, ids, feats = [], [], [], [], []
    intervals_list, prepared, input_ids_list, content_masks, n_mfa_ok = [], [], [], [], 0
    with torch.no_grad():
        for _, row in samples.iterrows():
            sid, text = str(row["sample_id"]), "" if pd.isna(row.get("text")) else str(row["text"])
            video, work = root / str(row["rel_path"]), ensure_dir(work_root / sid_stem(sid))
            ids_np, mask, content, offsets, hidden = encode_bert_offsets(
                tok, enc, text, L, device, padding=cfg["text"].get("padding", "max_length"),
                truncation=cfg["text"].get("truncation", True))
            clip = prepare_mfa_clip(text, video, work, align_cfg, fallback_duration=float(row.get("container_duration_sec") or 0.0))
            aligned_ok = clip["method"] == "mfa" and len(clip["words"]) > 0 and len(clip["intervals"]) == len(clip["words"])
            use_w = clip["words"] if aligned_ok else []
            use_iv = clip["intervals"] if aligned_ok else np.zeros((0, 2))
            token_intervals, time_mask = bridge_wordpiece_to_mfa(text, use_w, use_iv, offsets, ids_np, mask, tok)
            if aligned_ok and time_mask.any():
                n_mfa_ok += 1
            else:
                aligned_ok = False
            masks.append(mask); supports.append(mask.copy()); time_masks.append(time_mask.astype(np.int8))
            content_masks.append(content); input_ids_list.append(ids_np)
            intervals_list.append(token_intervals.astype(np.float32)); feats.append(hidden); ids.append(sid)
            prepared.append({"sid": sid, "video": video, "work": work, "info": clip["info"], "audio": clip["audio"],
                             "status": clip["status"], "duration": clip["duration"], "audio_offset": clip["audio_offset"],
                             "alignment": "mfa" if aligned_ok else "mfa_failed"})
    return (np.stack(masks), np.stack(supports), np.stack(time_masks), np.stack(content_masks), ids,
            np.stack(feats).astype(np.float32), np.stack(input_ids_list).astype(np.int32), intervals_list, prepared, n_mfa_ok)

def run_audio_on_token_intervals(cfg, sample_ids, token_intervals, time_masks, prepared):
    L, sr = int(cfg["L"]), int(cfg["audio"]["sample_rate"])
    k_min, frame_ms = int(cfg["audio"]["k_min"]), float(cfg["audio"]["frame_ms"])
    hop_ms, floor = float(cfg["audio"]["hop_ms"]), float(cfg["audio"]["frame_energy_floor"])
    hop_sec = float(cfg["audio"].get("covarep_hop_sec", 0.01))
    work, need = ensure_dir(Path(cfg["paths"]["output"]) / "_tmp_wav"), []
    wav_batch = ensure_dir(work / "batch")
    for item in prepared:
        speech, dst = item["work"] / "speech.wav", wav_batch / f"{item['work'].name}.wav"
        mat = dst.with_suffix(".mat")
        if item["audio"] is not None and speech.exists():
            shutil.copyfile(speech, dst)
            if not install_covarep_mat(item["sid"], mat):
                need.append(dst)
    batch_covarep_mats(cfg, need, work / "_pending_covarep", dest_dir=wav_batch)
    all_rms = [float(np.sqrt(np.mean(np.square(item["audio"][0].astype(np.float64))) + 1e-20))
               for item in prepared if item["audio"] is not None and item["audio"][0] is not None and len(item["audio"][0])]
    tau_a = float(np.percentile(all_rms, float(cfg["audio"]["tau_a_percentile"]))) if all_rms else 0.0
    masks, supports, ids, feats, by_sid = [], [], [], [], {p["sid"]: p for p in prepared}
    for i, sid in enumerate(sample_ids):
        item, intervals, tmask = by_sid[sid], np.asarray(token_intervals[i], dtype=np.float64), time_masks[i].astype(bool)
        feat, mask, support = np.zeros((L, 74), dtype=np.float32), np.zeros(L, dtype=np.int8), np.zeros(L, dtype=np.int8)
        mat = wav_batch / f"{item['work'].name}.mat"
        if item["alignment"] == "mfa" and tmask.any() and mat.is_file():
            src_i, src_v, src_q, _ = audio_source_covarep(mat, hop_sec, float(item.get("audio_offset", 0.0)),
                                                         duration=float(item["duration"]), cfg=cfg)
            if len(src_v) > 0:
                tgt = intervals[tmask]
                a_feat, a_obs = aggregate(tgt, src_i, src_v, src_q if len(src_q) else np.zeros(0))
                sig = None if item["audio"] is None else item["audio"][0]
                use_sr = sr if item["audio"] is None else int(item["audio"][1])
                a_q = word_audio_quality(sig, use_sr, tgt, tau_a, k_min, frame_ms, hop_ms, floor)
                audio_end = float(len(sig) / max(use_sr, 1)) if sig is not None else 0.0
                a_sup = support_overlap(tgt, audio_end)
                valid = a_obs.astype(np.float64) * a_q * a_sup
                feat[tmask] = a_feat * valid[:, None]
                mask[tmask] = (valid > 0).astype(np.int8)
                support[tmask] = a_sup.astype(np.int8)
        masks.append(mask); supports.append(support); feats.append(feat); ids.append(sid)
    return np.stack(masks), np.stack(supports), ids, np.stack(feats), list(COVAREP_NAMES)

def run_vision_on_token_intervals(cfg, sample_ids, token_intervals, time_masks, prepared):
    L, of_work = int(cfg["L"]), ensure_dir(Path(cfg["paths"]["output"]) / "_tmp_openface")
    by_sid, masks, supports, ids, feats = {p["sid"]: p for p in prepared}, [], [], [], []
    for i, sid in enumerate(sample_ids):
        item, intervals, tmask = by_sid[sid], np.asarray(token_intervals[i], dtype=np.float64), time_masks[i].astype(bool)
        feat, mask, support = np.zeros((L, 35), dtype=np.float32), np.zeros(L, dtype=np.int8), np.zeros(L, dtype=np.int8)
        if item["alignment"] == "mfa" and tmask.any() and item["video"].exists():
            work = ensure_dir(of_work / sid_stem(sid))
            try:
                src_i, src_v, src_q, _ = vision_source_openface(item["video"], work, float(item["duration"]), cfg, sample_id=sid)
            except Exception:
                src_i, src_v, src_q = np.zeros((0, 2)), np.zeros((0, 35), dtype=np.float32), np.zeros(0)
            if len(src_v) > 0:
                tgt = intervals[tmask]
                v_feat, v_obs = aggregate(tgt, src_i, src_v, src_q if len(src_q) else np.zeros(0))
                v_sup = support_overlap(tgt, float(item["duration"]), dtype=np.int8)
                valid = v_obs.astype(np.int8) * v_sup
                feat[tmask] = v_feat * valid[:, None]
                mask[tmask], support[tmask] = valid, v_sup
        masks.append(mask); supports.append(support); feats.append(feat); ids.append(sid)
    return np.stack(masks), np.stack(supports), ids, np.stack(feats), list(VISION_NAMES)

def main() -> int:
    args = parse_extract_args()
    cfg, out, samples = setup_run(args, "config_a.json")
    (tmask, tsupport, t_time_mask, t_content_mask, tids, tfeat, input_ids,
     token_intervals, prepared, n_mfa) = run_text_with_mfa(cfg, samples)
    amask, asupport, aids, afeat, anames = run_audio_on_token_intervals(cfg, tids, token_intervals, t_time_mask, prepared)
    vmask, vsupport, vids, vfeat, vnames = run_vision_on_token_intervals(cfg, tids, token_intervals, t_time_mask, prepared)
    tfeat, afeat, vfeat = standardize_tav(tfeat, tmask, afeat, amask, vfeat, vmask,
                                          t_support=tsupport, a_support=asupport, v_support=vsupport)
    intervals_arr = np.stack([np.asarray(x, dtype=np.float32) for x in token_intervals])
    np.savez_compressed(out / "masks.npz", sample_id=np.array(tids), text_mask=tmask.astype(np.int8),
                        audio_mask=amask.astype(np.int8), vision_mask=vmask.astype(np.int8),
                        text_support=tsupport.astype(np.int8), audio_support=asupport.astype(np.int8),
                        vision_support=vsupport.astype(np.int8), text_time_mask=t_time_mask.astype(np.int8),
                        text_content_mask=t_content_mask.astype(np.int8))
    np.savez_compressed(out / "text_features.npz", sample_id=np.array(tids), text=tfeat.astype(np.float32),
                        input_ids=input_ids.astype(np.int32), text_mask=tmask.astype(np.int8),
                        text_support=tsupport.astype(np.int8), text_time_mask=t_time_mask.astype(np.int8),
                        text_content_mask=t_content_mask.astype(np.int8), text_intervals=intervals_arr)
    np.savez_compressed(out / "audio_features.npz", sample_id=np.array(aids), audio=afeat.astype(np.float32),
                        feature_names=np.array(anames), audio_mask=amask.astype(np.int8), audio_support=asupport.astype(np.int8))
    np.savez_compressed(out / "vision_features.npz", sample_id=np.array(vids), vision=vfeat.astype(np.float32),
                        feature_names=np.array(vnames), vision_mask=vmask.astype(np.int8), vision_support=vsupport.astype(np.int8))
    cleanup_tmps(out, ("_tmp_wav", "_tmp_openface", "_tmp_mfa_text"), args.keep_tmp)
    print(f"done method_a n={len(tids)} mfa_ok={n_mfa}/{len(tids)} text={tfeat.shape} audio={afeat.shape} vision={vfeat.shape}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
