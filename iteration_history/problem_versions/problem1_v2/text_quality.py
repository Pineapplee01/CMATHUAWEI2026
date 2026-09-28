"""文本：BERT WordPiece 50 + MFA 整词时间；同词子词共享 [s,e)。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import BertModel, BertTokenizerFast

from problem1_v3.mfa_alignment import word_alignment
from problem1_v3.mfa_alignment.media import audio_track, media_info
from problem1_v2.common import ensure_dir
from problem1_v2.token_time import bridge_wordpiece_to_mfa
from shared.common import path as resolve_path


def _mfa_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    mfa = cfg.get("mfa") or {}
    reuse = mfa.get("reuse_mfa_root", "problem1_v3/results/_tmp_mfa")
    return {
        "mfa_executable": mfa.get("mfa_executable", "mfa"),
        "dictionary": cfg["paths"]["dictionary"],
        "acoustic": cfg["paths"]["acoustic"],
        "fallback_uniform": False,
        "batch_alignment": False,
        # 复用 v3 已有 TextGrid，避免重复 MFA（勿用 view50 的 intervals.npz）
        "reuse_mfa_root": str(resolve_path(reuse)),
    }


def run_text_with_mfa(cfg: dict[str, Any], samples: pd.DataFrame):
    """WordPiece 特征 + 词元时间窗；并返回 MFA 媒体准备信息供音视聚合。"""
    L = int(cfg["L"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = BertTokenizerFast.from_pretrained(cfg["paths"]["bert"])
    enc = BertModel.from_pretrained(cfg["paths"]["bert"]).to(device)
    enc.eval()

    root = Path(cfg["paths"]["video_root"])
    work_root = ensure_dir(Path(cfg["paths"]["output"]) / "_tmp_mfa_text")
    mfa_cfg = _mfa_cfg(cfg)

    rows, masks, supports, time_masks, ids, feats = [], [], [], [], [], []
    intervals_list, prepared, manifests = [], [], []
    input_ids_list, content_masks = [], []

    with torch.no_grad():
        for _, row in samples.iterrows():
            sid = str(row["sample_id"])
            text = "" if pd.isna(row.get("text")) else str(row["text"])
            video = root / str(row["rel_path"])
            work = ensure_dir(work_root / sid.replace("/", "_").replace("$_$", "__"))

            batch = tok(
                text,
                padding=cfg["text"].get("padding", "max_length"),
                truncation=cfg["text"].get("truncation", True),
                max_length=L,
                add_special_tokens=True,  # 与附件2一致：槽0=[CLS]，序列含[SEP]
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            offsets = [(int(a), int(b)) for a, b in batch.pop("offset_mapping")[0].tolist()]
            ids_np = batch["input_ids"][0].numpy()
            mask = batch["attention_mask"][0].numpy().astype(np.int8)
            if int(ids_np[0]) != int(tok.cls_token_id):
                raise ValueError(f"{sid}: 槽0不是[CLS]，got={int(ids_np[0])}")
            if int(tok.sep_token_id) not in set(ids_np.tolist()):
                raise ValueError(f"{sid}: 序列缺少[SEP]")
            content = (
                (ids_np != tok.cls_token_id)
                & (ids_np != tok.sep_token_id)
                & (ids_np != tok.pad_token_id)
                & (mask > 0)
            ).astype(np.int8)
            out = enc(**{k: v.to(device) for k, v in batch.items()})
            hidden = out.last_hidden_state[0].detach().cpu().numpy().astype(np.float32)
            hidden[mask == 0] = 0.0

            method, confidence, failure = "mfa_failed", "none", ""
            words: list[str] = []
            word_intervals = np.zeros((0, 2), dtype=np.float64)
            duration = float(row.get("container_duration_sec") or 0.0)
            audio_offset = 0.0
            audio = None
            status: dict[str, Any] = {"audio_offset": 0.0}
            info: dict[str, Any] = {"duration": duration}

            try:
                info = media_info(video)
                audio, status = audio_track(video, work, info)
                words, word_intervals, method, confidence, failure = word_alignment(
                    text, audio, status, info["duration"], work, mfa_cfg
                )
                duration = float(info["duration"])
                audio_offset = float(status.get("audio_offset", 0.0))
            except Exception as exc:  # noqa: BLE001
                failure = repr(exc)

            aligned_ok = method == "mfa" and len(words) > 0 and len(word_intervals) == len(words)
            if aligned_ok:
                try:
                    bridged = bridge_wordpiece_to_mfa(
                        text, words, word_intervals, offsets, ids_np, mask, tok
                    )
                except ValueError as exc:
                    aligned_ok = False
                    failure = str(exc)
                    bridged = bridge_wordpiece_to_mfa(
                        text, [], np.zeros((0, 2)), offsets, ids_np, mask, tok
                    )
            else:
                bridged = bridge_wordpiece_to_mfa(
                    text, [], np.zeros((0, 2)), offsets, ids_np, mask, tok
                )

            time_mask = bridged["time_mask"].astype(np.int8)
            token_intervals = bridged["token_intervals"].astype(np.float64)
            # text_support 与 attention 一致（结构符也有表示）；播放时间另见 time_mask
            support = mask.copy()

            rows.append(
                {
                    "sample_id": sid,
                    "token_length": int(mask.sum()),
                    "valid_token_num": int(mask.sum()),
                    "padding_num": int((mask == 0).sum()),
                    "n_timed_tokens": int(time_mask.sum()),
                    "n_content_tokens": int(content.sum()),
                    "n_mfa_words": int(len(words)),
                    "has_cls": 1,
                    "has_sep": 1,
                    "alignment": "mfa" if aligned_ok else "mfa_failed",
                    "alignment_failure": failure,
                    "tokenizer": "wordpiece",
                    "max_length": L,
                    "text_scheme": "wordpiece_mfa_shared_word_time",
                    "mask_consistent": int(
                        int(mask.sum()) == int((ids_np != tok.pad_token_id).sum())
                    ),
                    "feat_dim": 768,
                    "duration_sec": duration,
                }
            )
            masks.append(mask)
            supports.append(support)
            time_masks.append(time_mask)
            content_masks.append(content)
            input_ids_list.append(ids_np.astype(np.int32))
            intervals_list.append(token_intervals.astype(np.float32))
            feats.append(hidden)
            ids.append(sid)
            prepared.append(
                {
                    "sid": sid,
                    "video": video,
                    "work": work,
                    "info": info,
                    "audio": audio,
                    "status": status,
                    "duration": duration,
                    "audio_offset": audio_offset,
                    "alignment": "mfa" if aligned_ok else "mfa_failed",
                    "failure": failure,
                }
            )
            manifests.append(
                {
                    "sample_id": sid,
                    "alignment": "mfa" if aligned_ok else "mfa_failed",
                    "alignment_confidence": confidence,
                    "alignment_failure": failure,
                    "n_mfa_words": int(len(words)),
                    "n_timed_tokens": int(time_mask.sum()),
                    "word_axis": bridged["word_axis"],
                    "token_to_word_indices": bridged["bridges"],
                    "token_time_status": bridged["statuses"],
                    "note": "同词 WordPiece 子词共享同一 MFA 词时间窗；音视按该窗重叠加权",
                }
            )

    return (
        pd.DataFrame(rows),
        np.stack(masks),
        np.stack(supports),
        np.stack(time_masks),
        np.stack(content_masks),
        ids,
        np.stack(feats).astype(np.float32),
        np.stack(input_ids_list).astype(np.int32),
        intervals_list,
        prepared,
        manifests,
    )


# 兼容旧入口名（若外部仍调用）
def run_text(cfg, samples):
    tdf, tmask, tsupport, _tt, _cm, ids, tfeat, _ii, _iv, _prep, _man = run_text_with_mfa(
        cfg, samples
    )
    return tdf, tmask, tsupport, ids, tfeat
