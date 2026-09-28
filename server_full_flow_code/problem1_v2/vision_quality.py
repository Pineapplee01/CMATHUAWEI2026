"""视觉：按 WordPiece 词元时间窗（MFA 整词区间，同词子词共享）重叠加权聚合。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from problem1_v2.common import ensure_dir
from problem1_v3.extractors import VISION_NAMES, vision_source_openface
from problem1_v3.mfa_alignment import aggregate


def run_vision_on_token_intervals(
    cfg: dict[str, Any],
    sample_ids: list[str],
    token_intervals: list[np.ndarray],
    time_masks: np.ndarray,
    prepared: list[dict[str, Any]],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], np.ndarray, list[str]]:
    L = int(cfg["L"])
    names = list(VISION_NAMES)
    of_work = ensure_dir(Path(cfg["paths"]["output"]) / "_tmp_openface")
    by_sid = {p["sid"]: p for p in prepared}

    rows, masks, supports, ids, feats = [], [], [], [], []
    for i, sid in enumerate(sample_ids):
        item = by_sid[sid]
        intervals = np.asarray(token_intervals[i], dtype=np.float64)
        tmask = time_masks[i].astype(bool)
        feat = np.zeros((L, 35), dtype=np.float32)
        mask = np.zeros(L, dtype=np.int8)
        support = np.zeros(L, dtype=np.int8)
        n_timed = int(tmask.sum())
        n_valid = 0
        openface_ok = 0

        if item["alignment"] == "mfa" and n_timed > 0 and item["video"].exists():
            work = ensure_dir(of_work / sid.replace("/", "_").replace("$_$", "__"))
            try:
                src_i, src_v, src_q, _ = vision_source_openface(
                    item["video"], work, float(item["duration"]), cfg, sample_id=sid
                )
                openface_ok = int(len(src_v) > 0)
            except Exception:  # noqa: BLE001
                src_i = np.zeros((0, 2))
                src_v = np.zeros((0, 35), dtype=np.float32)
                src_q = np.zeros(0)
                openface_ok = 0

            if openface_ok:
                tgt = intervals[tmask]
                v_feat, v_obs, _ = aggregate(
                    tgt, src_i, src_v, src_q if len(src_q) else np.zeros(0)
                )
                # 视觉支持：区间与 [0,D) 相交且有观测
                v_sup = np.ones(len(tgt), dtype=np.int8)
                for j, (s, e) in enumerate(tgt):
                    if not (min(float(e), float(item["duration"])) > max(float(s), 0.0)):
                        v_sup[j] = 0
                valid = v_obs.astype(np.int8) * v_sup
                feat[tmask] = v_feat * valid[:, None]
                mask[tmask] = valid
                support[tmask] = v_sup
                n_valid = int(valid.sum())

        rows.append(
            {
                "sample_id": sid,
                "openface_ok": openface_ok,
                "n_timed_tokens": n_timed,
                "n_valid_vision_slots": n_valid,
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
                    "openface_ok": int(rows[i]["openface_ok"]),
                    "alignment": rows[i]["alignment"],
                }
            )
    return pd.DataFrame(detail_rows), np.stack(masks), np.stack(supports), ids, np.stack(feats), names
