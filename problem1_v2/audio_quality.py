"""语音：按 WordPiece 词元时间窗（MFA 整词区间，同词子词共享）重叠加权聚合。"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from problem1_v2.common import ensure_dir
from problem1_v2.covarep_audio import COVAREP_NAMES, run_covarep_dir
from problem1_v2.shared_cache import install_covarep_mat, publish_covarep_mat
from problem1_v3.extractors import audio_source_covarep, word_audio_quality
from problem1_v3.mfa_alignment import aggregate


def run_audio_on_token_intervals(
    cfg: dict[str, Any],
    samples: pd.DataFrame,
    sample_ids: list[str],
    token_intervals: list[np.ndarray],
    time_masks: np.ndarray,
    prepared: list[dict[str, Any]],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], np.ndarray, list[str]]:
    L = int(cfg["L"])
    sr = int(cfg["audio"]["sample_rate"])
    k_min = int(cfg["audio"]["k_min"])
    frame_ms = float(cfg["audio"]["frame_ms"])
    hop_ms = float(cfg["audio"]["hop_ms"])
    floor = float(cfg["audio"]["frame_energy_floor"])
    hop_sec = float(cfg["audio"].get("covarep_hop_sec", 0.01))
    names = list(COVAREP_NAMES)

    work = ensure_dir(Path(cfg["paths"]["output"]) / "_tmp_wav")
    wav_batch = ensure_dir(work / "batch")
    need = []
    for item in prepared:
        sid = item["sid"]
        speech = item["work"] / "speech.wav"
        dst = wav_batch / f"{item['work'].name}.wav"
        mat = dst.with_suffix(".mat")
        if item["audio"] is not None and speech.exists():
            shutil.copyfile(speech, dst)
            if not install_covarep_mat(sid, mat):
                need.append(dst)

    if need:
        pending = ensure_dir(work / "_pending_covarep")
        for p in list(pending.glob("*")):
            p.unlink()
        for p in need:
            shutil.copy2(p, pending / p.name)
        run_covarep_dir(cfg, pending)
        for p in need:
            src = pending / f"{p.stem}.mat"
            if src.is_file():
                shutil.copy2(src, wav_batch / src.name)
                publish_covarep_mat(p.stem.replace("__", "$_$"), wav_batch / src.name)
        shutil.rmtree(pending, ignore_errors=True)

    # 批内 τ_A：有信号样本的全局 RMS 分位
    all_rms = []
    for item in prepared:
        if item["audio"] is None:
            continue
        sig = item["audio"][0]
        if sig is None or len(sig) == 0:
            continue
        all_rms.append(float(np.sqrt(np.mean(np.square(sig.astype(np.float64))) + 1e-20)))
    tau_a = (
        float(np.percentile(all_rms, float(cfg["audio"]["tau_a_percentile"])))
        if all_rms
        else 0.0
    )

    rows, masks, supports, ids, feats = [], [], [], [], []
    by_sid = {p["sid"]: p for p in prepared}

    for i, sid in enumerate(sample_ids):
        item = by_sid[sid]
        intervals = np.asarray(token_intervals[i], dtype=np.float64)
        tmask = time_masks[i].astype(bool)
        feat = np.zeros((L, 74), dtype=np.float32)
        mask = np.zeros(L, dtype=np.int8)
        support = np.zeros(L, dtype=np.int8)

        mat = wav_batch / f"{item['work'].name}.mat"
        covarep_ok = 0
        n_timed = int(tmask.sum())
        n_valid = 0

        if item["alignment"] == "mfa" and n_timed > 0 and mat.is_file():
            src_i, src_v, src_q, _ = audio_source_covarep(
                mat,
                hop_sec,
                float(item.get("audio_offset", 0.0)),
                duration=float(item["duration"]),
                cfg=cfg,
            )
            covarep_ok = int(len(src_v) > 0)
            if covarep_ok:
                tgt = intervals[tmask]
                a_feat, a_obs, _ = aggregate(tgt, src_i, src_v, src_q if len(src_q) else np.zeros(0))
                sig = None if item["audio"] is None else item["audio"][0]
                use_sr = sr if item["audio"] is None else int(item["audio"][1])
                a_q = word_audio_quality(
                    sig, use_sr, tgt, tau_a, k_min, frame_ms, hop_ms, floor
                )
                # 片尾支持：区间与 [0, audio_end) 相交
                audio_end = float(len(sig) / max(use_sr, 1)) if sig is not None else 0.0
                a_sup = np.array(
                    [
                        1 if min(float(e), audio_end) > max(float(s), 0.0) else 0
                        for s, e in tgt
                    ],
                    dtype=np.float64,
                )
                valid = a_obs.astype(np.float64) * a_q * a_sup
                scatter_feat = a_feat * valid[:, None]
                scatter_mask = (valid > 0).astype(np.int8)
                scatter_sup = a_sup.astype(np.int8)
                feat[tmask] = scatter_feat
                mask[tmask] = scatter_mask
                support[tmask] = scatter_sup
                n_valid = int(scatter_mask.sum())

        rows.append(
            {
                "sample_id": sid,
                "tau_a": tau_a,
                "covarep_ok": covarep_ok,
                "n_timed_tokens": n_timed,
                "n_valid_audio_slots": n_valid,
                "alignment": item["alignment"],
            }
        )
        masks.append(mask)
        supports.append(support)
        feats.append(feat)
        ids.append(sid)

    detail_rows = []
    for i, sid in enumerate(ids):
        for k in range(L):
            detail_rows.append(
                {
                    "sample_id": sid,
                    "window_id": k,
                    "slot": k,
                    "mask": int(masks[i][k]),
                    "support_mask": int(supports[i][k]),
                    "timed": int(time_masks[i][k]),
                    "t0": float(token_intervals[i][k, 0]),
                    "t1": float(token_intervals[i][k, 1]),
                    "tau_a": float(rows[i]["tau_a"]),
                    "covarep_ok": int(rows[i]["covarep_ok"]),
                    "alignment": rows[i]["alignment"],
                }
            )
    return pd.DataFrame(detail_rows), np.stack(masks), np.stack(supports), ids, np.stack(feats), names
