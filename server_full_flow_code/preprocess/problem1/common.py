# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：GPT-6 Astra
# 版本/型号：GPT-6 Astra（gpt-6-astra）
# 开发机构/公司：OpenAI
# 版本颁布日期：2026年9月3日
# 发布信息来源：https://openai.com/index/safety-overview-gpt-6-astra/

from __future__ import annotations
import argparse, json, os, re, shutil, subprocess
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parent
COVAREP_CACHE = ROOT / "AAAdata" / "processed" / "problem1_covarep"
OPENFACE_CACHE = ROOT / "AAAdata" / "processed" / "problem1_openface"
COVAREP_NAMES = [
    "F0", "VUV", "NAQ", "QOQ", "H1H2", "PSP", "MDQ", "peakSlope", "Rd", "Rd_conf", "creak",
    *[f"MCEP_{i}" for i in range(25)], *[f"HMPDM_{i}" for i in range(25)], *[f"HMPDD_{i}" for i in range(13)],
]
AUS = ("01", "02", "04", "05", "06", "07", "09", "10", "12", "14", "15", "17", "20", "23", "25", "26", "45")
VISION_NAMES = [f"AU{x}_r" for x in AUS] + [f"AU{x}_c" for x in (*AUS[:-1], "28", "45")]

def resolve_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p

def _resolve_cfg_entry(rel: str) -> str:
    # 裸命令名（如 matlab / mfa）保持原样，走 PATH；其余相对路径相对 ROOT
    if not rel or Path(rel).is_absolute():
        return rel
    if "/" not in rel and "\\" not in rel:
        return rel
    return str(ROOT / rel)

def load_config(path: Path | None = None) -> dict[str, Any]:
    path = Path(path or PKG / "config_q1.json")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key, rel in list(cfg["paths"].items()):
        cfg["paths"][key] = _resolve_cfg_entry(str(rel))
    return cfg

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_samples(cfg: dict[str, Any]) -> pd.DataFrame:
    labels = pd.read_excel(cfg["paths"]["labels"])
    labels["clip_id"] = labels["clip_id"].astype(int)
    labels["sample_id"] = labels["video_id"].astype(str) + "$_$" + labels["clip_id"].astype(str)
    labels["rel_path"] = labels["video_id"].astype(str) + "/" + labels["clip_id"].astype(str) + ".mp4"
    qc_path = Path(cfg["paths"]["qc"])
    if qc_path.exists():
        qc = pd.read_csv(qc_path)
        qc["clip_id"] = qc["clip_id"].astype(int)
        keep = [c for c in ("video_id", "clip_id", "container_duration_sec", "is_silent_audio", "usable_for_audio") if c in qc.columns]
        labels = labels.merge(qc[keep], on=["video_id", "clip_id"], how="left")
    return labels

def parse_extract_args():
    p = argparse.ArgumentParser()
    p.add_argument("action", nargs="?", default="extract", choices=["extract"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--keep-tmp", action="store_true")
    return p.parse_args()

def setup_run(args, default_cfg):
    # default_cfg 可为相对 PKG 的文件名，或相对 ROOT / 绝对路径
    cfg_path = Path(default_cfg)
    if not cfg_path.is_absolute() and not cfg_path.is_file():
        local = PKG / default_cfg
        cfg_path = local if local.is_file() else ROOT / default_cfg
    cfg = load_config(args.config or cfg_path)
    if args.limit is not None:
        cfg["paths"]["output"] += f"_smoke_{max(args.limit, 1)}"
    out = ensure_dir(Path(cfg["paths"]["output"]))
    samples = load_samples(cfg)
    if args.limit is not None:
        samples = samples.head(int(args.limit))
    return cfg, out, samples

def cleanup_tmps(out: Path, names, keep: bool):
    if not keep:
        for n in names:
            shutil.rmtree(out / n if isinstance(n, str) else n, ignore_errors=True)

def probe_duration(video: Path, fallback: float | None = None) -> float:
    if fallback is not None and np.isfinite(fallback) and float(fallback) > 0:
        return float(fallback)
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video)], text=True,
    ).strip()
    return float(out)

def extract_wav(video: Path, wav: Path, sr: int = 16000) -> bool:
    ensure_dir(wav.parent)
    try:
        subprocess.run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", str(wav)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return wav.exists() and wav.stat().st_size > 44
    except subprocess.CalledProcessError:
        return False

def common_hop_sec(cfg: dict[str, Any], default: float = 0.04) -> float:
    sync = cfg.get("av_sync") or {}
    return float(sync.get("common_hop_sec", cfg.get("common_hop_sec", default)))

def uniform_frame_grid(duration: float, hop: float) -> np.ndarray:
    d, h = max(float(duration), 0.0), float(hop)
    if d <= 0 or h <= 0:
        return np.zeros((0, 2), dtype=np.float64)
    n = max(int(np.ceil(d / h - 1e-12)), 0)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    starts = np.arange(n, dtype=np.float64) * h
    ends = np.minimum(starts + h, d)
    keep = ends > starts + 1e-12
    return np.stack([starts[keep], ends[keep]], axis=1)

def resample_to_grid(target, source, values, quality=None):
    target, source = np.asarray(target, dtype=np.float64), np.asarray(source, dtype=np.float64)
    values = np.asarray(values, dtype=np.float32)
    t = len(target)
    dim = int(values.shape[1]) if values.ndim == 2 and len(values) else (int(values.shape[-1]) if values.size else 0)
    if t == 0 or dim <= 0 or len(source) == 0 or len(values) == 0:
        return np.zeros((t, max(dim, 1)), dtype=np.float32), np.zeros(t, dtype=bool)
    n = min(len(source), len(values))
    source, values = source[:n], values[:n]
    quality = np.ones(n, dtype=np.float64) if quality is None else np.asarray(quality, dtype=np.float64).reshape(-1)[:n]
    if len(quality) < n:
        quality = np.pad(quality, (0, n - len(quality)))
    overlap = np.maximum(0.0, np.minimum(target[:, None, 1], source[None, :, 1]) - np.maximum(target[:, None, 0], source[None, :, 0]))
    weights = overlap * quality[None, :]
    denom = weights.sum(axis=1)
    feat = weights @ values.astype(np.float64) / np.maximum(denom[:, None], 1e-12)
    usable = denom > 1e-12
    feat = np.nan_to_num(feat.astype(np.float32), nan=0.0)
    feat[~usable] = 0.0
    return feat, usable

def unify_modality(duration, hop, source_intervals, values, quality=None):
    grid = uniform_frame_grid(duration, hop)
    feat, usable = resample_to_grid(grid, source_intervals, values, quality)
    return grid, feat, usable

def standardize_features(features, valid_mask, support_mask=None, eps=1e-6):
    x = np.asarray(features, dtype=np.float32).copy()
    if x.ndim != 3:
        return x, {"mean": np.zeros(1, dtype=np.float32), "std": np.ones(1, dtype=np.float32)}
    vm = np.asarray(valid_mask, dtype=np.int8)
    sm = np.asarray(support_mask, dtype=np.int8) if support_mask is not None else None
    usable = (vm == 1) & (sm == 1) if sm is not None else (vm == 1)
    flat = x[usable]
    if flat.size == 0:
        x[:] = 0.0
        return x, {"mean": np.zeros(x.shape[-1], dtype=np.float32), "std": np.ones(x.shape[-1], dtype=np.float32)}
    mean = flat.mean(axis=0).astype(np.float32)
    std = np.maximum(flat.std(axis=0).astype(np.float32), eps)
    x = (x - mean) / std
    x[~usable] = 0.0
    if sm is not None:
        x[sm == 0] = 0.0
    return x.astype(np.float32), {"mean": mean, "std": std}

def standardize_tav(tfeat, tmask, afeat, amask, vfeat, vmask, t_support=None, a_support=None, v_support=None):
    # 有效位 z-score，无效位置零；不另存 norm_stats
    tfeat, _ = standardize_features(tfeat, tmask, t_support)
    afeat, _ = standardize_features(afeat, amask, a_support)
    vfeat, _ = standardize_features(vfeat, vmask, v_support)
    return tfeat, afeat, vfeat

def sid_stem(sample_id: str) -> str:
    return str(sample_id).replace("/", "_").replace("$_$", "__")

def pad_trunc_av(x: np.ndarray, length: int, L: int, dim: int):
    feat, mask = np.zeros((L, dim), dtype=np.float32), np.zeros(L, dtype=np.int8)
    n = int(min(length, L))
    if n > 0 and x.size:
        feat[:n] = x[:n]
        mask[:n] = 1
    return feat, mask, n

def encode_bert_offsets(tokenizer, encoder, text, max_length, device, *, padding="max_length", truncation=True):
    batch = tokenizer(text, padding=padding, truncation=truncation, max_length=max_length,
                      add_special_tokens=True, return_tensors="pt", return_offsets_mapping=True)
    offsets = [(int(a), int(b)) for a, b in batch.pop("offset_mapping")[0].tolist()]
    ids_np, mask = batch["input_ids"][0].numpy(), batch["attention_mask"][0].numpy().astype(np.int8)
    content = ((ids_np != tokenizer.cls_token_id) & (ids_np != tokenizer.sep_token_id)
               & (ids_np != tokenizer.pad_token_id) & (mask > 0)).astype(np.int8)
    hidden = encoder(**{k: v.to(device) for k, v in batch.items()}).last_hidden_state[0]
    hidden = hidden.detach().cpu().numpy().astype(np.float32)
    hidden[mask == 0] = 0.0
    return ids_np.astype(np.int32), mask, content, offsets, hidden

def prepare_mfa_clip(text, video, work, align_cfg, fallback_duration: float = 0.0):
    try:
        info = media_info(video)
        audio, status = audio_track(video, work, info)
        words, intervals, method, _, failure = word_alignment(text, audio, status, info["duration"], work, align_cfg)
        return {"info": info, "audio": audio, "status": status, "words": words, "intervals": intervals,
                "method": method, "failure": failure, "duration": float(info["duration"]),
                "audio_offset": float(status.get("audio_offset", 0.0))}
    except Exception as exc:
        return {"info": {"duration": float(fallback_duration)}, "audio": None,
                "status": {"audio_status": "decode_failed", "audio_feature_available": False, "audio_offset": 0.0},
                "words": [], "intervals": np.zeros((0, 2), dtype=np.float64), "method": "mfa_failed",
                "failure": str(exc), "duration": float(fallback_duration), "audio_offset": 0.0}

def _stems(sample_id: str) -> list[str]:
    s = sample_id.replace("/", "_")
    out, seen = [], set()
    for a in (s, s.replace("$_$", "__"), s.replace("__", "$_$")):
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out

def _copy_if_needed(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.resolve() != src.resolve():
        shutil.copy2(src, dst)

def _find_cached(paths) -> Path | None:
    for p in paths:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None

def covarep_cached_mat(sample_id: str) -> Path | None:
    COVAREP_CACHE.mkdir(parents=True, exist_ok=True)
    return _find_cached(COVAREP_CACHE / f"{s}.mat" for s in _stems(sample_id))

def install_covarep_mat(sample_id: str, dest_mat: Path) -> bool:
    src = covarep_cached_mat(sample_id)
    if src is None:
        return False
    _copy_if_needed(src, dest_mat)
    return True

def publish_covarep_mat(sample_id: str, src_mat: Path) -> None:
    if src_mat.is_file() and src_mat.stat().st_size > 0:
        COVAREP_CACHE.mkdir(parents=True, exist_ok=True)
        _copy_if_needed(src_mat, COVAREP_CACHE / f"{_stems(sample_id)[0]}.mat")

def openface_cached_csv(sample_id: str, video_stem: str) -> Path | None:
    OPENFACE_CACHE.mkdir(parents=True, exist_ok=True)
    return _find_cached(OPENFACE_CACHE / s / f"{video_stem}.csv" for s in _stems(sample_id))

def install_openface_csv(sample_id: str, video_stem: str, dest_csv: Path) -> bool:
    src = openface_cached_csv(sample_id, video_stem)
    if src is None:
        return False
    _copy_if_needed(src, dest_csv)
    return True

def publish_openface_csv(sample_id: str, video_stem: str, src_csv: Path) -> None:
    if src_csv.is_file() and src_csv.stat().st_size > 0:
        OPENFACE_CACHE.mkdir(parents=True, exist_ok=True)
        _copy_if_needed(src_csv, OPENFACE_CACHE / _stems(sample_id)[0] / f"{video_stem}.csv")

def _covarep_matlab_dir() -> Path:
    for candidate in (PKG, ROOT / "problem1_v2", ROOT / "problem1_v3", ROOT / "problem1_v4", ROOT / "problem1"):
        if (candidate / "covarep_batch.m").is_file():
            return candidate
    return ROOT / "problem1_v2"

def run_covarep_dir(cfg: dict[str, Any], wav_dir: Path) -> None:
    if not list(wav_dir.glob("*.wav")):
        return
    matlab, covarep = Path(cfg["paths"]["matlab"]), Path(cfg["paths"]["covarep"])
    hop = float(cfg["audio"].get("covarep_hop_sec", 0.01))
    q1 = _covarep_matlab_dir()
    subprocess.run([str(matlab), "-batch",
                    f"addpath('{q1.as_posix()}'); covarep_batch('{covarep.as_posix()}', '{wav_dir.as_posix()}', {hop});"],
                   check=True, cwd=str(q1))

def batch_covarep_mats(cfg, wav_paths, pending_dir, *, dest_dir=None, sample_ids=None):
    if not wav_paths:
        return
    pending = ensure_dir(pending_dir)
    for p in list(pending.glob("*")):
        p.unlink()
    for wav in wav_paths:
        shutil.copy2(wav, pending / wav.name)
    run_covarep_dir(cfg, pending)
    for i, wav in enumerate(wav_paths):
        src = pending / f"{wav.stem}.mat"
        if not src.is_file():
            continue
        dst = (dest_dir if dest_dir is not None else wav.parent) / f"{wav.stem}.mat"
        _copy_if_needed(src, dst)
        sid = sample_ids[i] if sample_ids is not None else wav.stem.replace("__", "$_$")
        publish_covarep_mat(sid, dst)
    shutil.rmtree(pending, ignore_errors=True)

def load_covarep_mat(mat_path: Path, hop_sec: float = 0.01):
    from scipy.io import loadmat
    empty = (np.zeros((0, 2), dtype=np.float64), np.zeros((0, 74), dtype=np.float32), list(COVAREP_NAMES))
    if not mat_path.exists():
        return empty
    data = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    feat = np.asarray(data["features"], dtype=np.float64)
    if feat.ndim == 1:
        feat = feat.reshape(1, -1)
    if feat.shape[1] != 74:
        return empty
    names = list(COVAREP_NAMES)
    if "names" in data:
        try:
            parsed = [str(x) for x in np.atleast_1d(data["names"]).tolist()]
            if len(parsed) == 74:
                names = parsed
        except Exception:
            pass
    t = (np.arange(len(feat), dtype=np.float64) + 0.5) * hop_sec
    half = hop_sec * 0.5
    return np.stack([t - half, t + half], axis=1), np.nan_to_num(feat.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0), names

def bert_words(words, tokenizer, encoder, device) -> np.ndarray:
    counts = [len(tokenizer.tokenize(w)) for w in words]
    groups, start, size = [], 0, 0
    for i, count in enumerate(counts):
        count = min(count, 510)
        if size + count > 510:
            groups.append((start, i))
            start, size = i, 0
        size += count
    groups.append((start, len(words)))
    all_features = []
    with torch.no_grad():
        for lo, hi in groups:
            inputs = tokenizer(words[lo:hi], is_split_into_words=True, return_tensors="pt", truncation=False)
            ids = inputs.word_ids()
            hidden = encoder(**{k: v.to(device) for k, v in inputs.items()}).last_hidden_state[0].cpu().numpy()
            for word_index in range(hi - lo):
                positions = [j for j, wid in enumerate(ids) if wid == word_index]
                all_features.append(hidden[positions].mean(0) if positions else np.zeros(hidden.shape[-1]))
    return np.asarray(all_features, dtype=np.float32)

def audio_source_covarep(mat_path, hop_sec, audio_offset=0.0, duration=None, cfg=None):
    intervals, values, names = load_covarep_mat(mat_path, hop_sec)
    if len(intervals) == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 74), dtype=np.float32), np.zeros(0, dtype=np.float64), names
    intervals = intervals + float(audio_offset)
    good = np.isfinite(values).all(1).astype(np.float64)
    if duration is not None and cfg is not None:
        grid, synced, usable = unify_modality(float(duration), common_hop_sec(cfg), intervals, values, good)
        return grid, synced, usable.astype(np.float64), names
    return intervals, values, good, names

def vision_source_openface(video, work, duration, cfg, sample_id=None):
    of_bin, out_dir = Path(cfg["paths"]["openface"]), work / "openface"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, sid = out_dir / f"{video.stem}.csv", sample_id or work.name
    empty = (np.zeros((0, 2), dtype=np.float64), np.zeros((0, 35), dtype=np.float32), np.zeros(0, dtype=np.float64), list(VISION_NAMES))
    if not install_openface_csv(sid, video.stem, csv_path):
        command([str(of_bin), "-f", str(video), "-out_dir", str(out_dir), "-aus"], work / "openface.log", cwd=str(of_bin.parent))
        if csv_path.exists():
            publish_openface_csv(sid, video.stem, csv_path)
    if not csv_path.exists():
        return empty
    frame = pd.read_csv(csv_path)
    frame.columns = frame.columns.str.strip()
    for col in VISION_NAMES:
        if col not in frame.columns:
            frame[col] = 0.0
    conf_min = float(cfg["vision"]["confidence_min"])
    ts = (frame["timestamp"].to_numpy(dtype=float) if "timestamp" in frame.columns else np.arange(len(frame), dtype=float) / 30.0)
    half = float(np.median(np.diff(ts)) * 0.5) if len(ts) > 1 else 1.0 / 60.0
    intervals = np.stack([np.maximum(0.0, ts - half), np.minimum(duration, ts + half)], axis=1)
    values = np.nan_to_num(frame[VISION_NAMES].to_numpy(dtype=np.float32), nan=0.0)
    success = frame["success"].to_numpy() if "success" in frame.columns else np.ones(len(frame))
    conf = frame["confidence"].to_numpy() if "confidence" in frame.columns else np.ones(len(frame))
    quality = ((success == 1) & (conf >= conf_min) & np.isfinite(values).all(1)).astype(np.float64)
    grid, synced, usable = unify_modality(float(duration), common_hop_sec(cfg), intervals, values, quality)
    return grid, synced, usable.astype(np.float64), list(VISION_NAMES)

def word_audio_quality(signal, sr, intervals, tau_a, k_min, frame_ms, hop_ms, floor):
    K, out = len(intervals), np.zeros(len(intervals), dtype=np.float64)
    if signal is None or len(signal) == 0:
        return out
    fl, hop = max(1, int(round(sr * frame_ms / 1000))), max(1, int(round(sr * hop_ms / 1000)))
    for k, (s, e) in enumerate(intervals):
        i0, i1 = max(0, int(round(s * sr))), min(len(signal), int(round(e * sr)))
        seg = signal[i0:i1]
        if seg.size == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(seg.astype(np.float64))) + 1e-20))
        if len(seg) < fl:
            vn = int(rms >= floor)
        else:
            n = 1 + (len(seg) - fl) // hop
            vn = int(sum(np.mean(np.square(seg[i * hop: i * hop + fl].astype(np.float64))) >= floor for i in range(n)))
        out[k] = 1.0 if (vn >= k_min and rms >= tau_a) else 0.0
    return out

def support_overlap(intervals, end: float, dtype=np.float64):
    return np.array([1 if min(float(e), end) > max(float(s), 0.0) else 0 for s, e in intervals], dtype=dtype)

def command(args, log, cwd=None):
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    result = subprocess.run([str(x) for x in args], text=True, capture_output=True, cwd=cwd, env=env)
    Path(log).write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    return result

def media_info(video):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(video)],
                            check=True, capture_output=True, text=True)
    info = json.loads(result.stdout)
    duration, origin = float(info["format"]["duration"]), float(info["format"].get("start_time", 0))
    frame_result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
         "-show_entries", "frame=best_effort_timestamp_time,pkt_duration_time", "-of", "json", str(video)],
        check=True, capture_output=True, text=True)
    frames = json.loads(frame_result.stdout).get("frames") or []
    ts = np.array([float(f["best_effort_timestamp_time"]) - origin for f in frames]) if frames else np.zeros(0)
    if not len(ts) or (len(ts) > 1 and np.any(np.diff(ts) <= 0)):
        ts = np.linspace(0, duration, max(1, int(duration * 30)), endpoint=False)
        dt = duration / max(len(ts), 1)
    else:
        dt = float(frames[-1].get("pkt_duration_time", np.median(np.diff(ts)) if len(ts) > 1 else duration))
    return {"duration": duration, "origin": origin, "frames": ts, "last_frame_duration": dt, "streams": info["streams"]}

def audio_track(video, work, info):
    import soundfile as sf
    streams = [s for s in info["streams"] if s["codec_type"] == "audio"]
    no_audio = {"audio_status": "no_audio_track", "audio_track_present": False, "decode_ok": False,
                "audio_feature_available": False, "audio_observation_state": "unavailable", "silence_cause": "unknown"}
    if not streams:
        return None, no_audio
    wav = work / "all_channels.wav"
    result = command(["ffmpeg", "-y", "-v", "error", "-i", video, "-map", "0:a:0", "-ar", "16000", "-c:a", "pcm_f32le", wav],
                     work / "audio_decode.log")
    if result.returncode or not wav.exists():
        return None, {**no_audio, "audio_status": "decode_failed"}
    signal, sr = sf.read(wav, always_2d=True, dtype="float32")
    if not len(signal) or not np.isfinite(signal).all():
        return None, {**no_audio, "audio_status": "decode_failed"}
    peaks, rms = np.max(np.abs(signal), axis=0), np.sqrt(np.mean(signal.astype(np.float64) ** 2, axis=0))
    selected, silent = int(rms.argmax()), bool(np.all(peaks == 0))
    status = {"audio_status": "silent_track" if silent else "decoded", "silence_cause": "unknown",
              "audio_track_present": True, "decode_ok": True, "audio_feature_available": not silent,
              "sample_rate": sr, "channels": signal.shape[1], "decoded_samples": len(signal),
              "peak_per_channel": peaks, "rms_per_channel": rms, "selected_channel": selected,
              "silence_threshold": 0.0, "audio_observation_state": "unknown_silence" if silent else "signal_present",
              "audio_offset": float(streams[0].get("start_time", info["origin"])) - info["origin"]}
    sf.write(work / "speech.wav", signal[:, selected], sr, subtype="PCM_16")
    return None if silent else (signal[:, selected], sr), status

def words_of(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*", text.lower())

def _reuse_textgrid(work: Path, cfg: dict) -> bool:
    root = cfg.get("reuse_mfa_root")
    if not root:
        return False
    src_aligned = Path(root) / work.name / "aligned"
    if not src_aligned.is_dir():
        return False
    candidates = list(src_aligned.rglob("*.TextGrid"))
    if len(candidates) != 1:
        return False
    _copy_if_needed(candidates[0], ensure_dir(work / "aligned") / "clip.TextGrid")
    return True

def mfa_cfg(cfg: dict[str, Any], *, with_reuse: bool = True) -> dict[str, Any]:
    mfa = cfg.get("mfa") or {}
    out = {"mfa_executable": mfa.get("mfa_executable", "mfa"), "dictionary": cfg["paths"]["dictionary"],
           "acoustic": cfg["paths"]["acoustic"], "fallback_uniform": False, "batch_alignment": False}
    if with_reuse:
        out["reuse_mfa_root"] = str(resolve_path(mfa.get("reuse_mfa_root", "problem1_v3/results/_tmp_mfa")))
    return out

def content_token_mask(ids, tokenizer) -> np.ndarray:
    special = {int(tokenizer.cls_token_id), int(tokenizer.sep_token_id), int(tokenizer.pad_token_id)}
    mid = getattr(tokenizer, "mask_token_id", None)
    if mid is not None:
        special.add(int(mid))
    ids = np.asarray(ids).reshape(-1)
    return np.array([1 if int(i) not in special else 0 for i in ids], dtype=np.int8)

def lexical_char_spans(text: str):
    return list(re.finditer(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*", text))

def interval_to_hop_span(start, end, hop, L_av) -> tuple[int, int]:
    if not np.isfinite([start, end]).all() or end <= start or hop <= 0:
        return 0, 0
    lo = int(np.clip(np.floor(start / hop), 0, L_av))
    hi = int(np.clip(np.ceil(end / hop), 0, L_av))
    return lo, max(hi, lo)

def bridge_wordpiece_to_mfa(text, words, intervals, offset_mapping, input_ids, attention_mask, tokenizer, hop=None, L_av=None):
    L = len(input_ids)
    token_intervals, time_mask = np.zeros((L, 2), dtype=np.float64), np.zeros(L, dtype=np.int8)
    want_hop = hop is not None
    hop_spans = np.zeros((L, 2), dtype=np.int32) if want_hop else None
    intervals = np.asarray(intervals, dtype=np.float64)
    empty = (token_intervals, hop_spans, time_mask) if want_hop else (token_intervals, time_mask)
    if not words or not len(intervals) or len(words) != len(intervals):
        return empty
    spans = lexical_char_spans(text)
    if [s.group().lower() for s in spans] != words or len(spans) != len(words):
        return empty
    axis = [(int(sp.start()), int(sp.end()), float(a), float(b)) for sp, (a, b) in zip(spans, intervals)]
    content = content_token_mask(input_ids, tokenizer) & (np.asarray(attention_mask) > 0)
    for i, ((c0, c1), ok) in enumerate(zip(offset_mapping, content)):
        if not ok:
            continue
        hit = [w for w in axis if min(c1, w[1]) > max(c0, w[0])]
        if not hit:
            continue
        a, b = hit[0][2], hit[-1][3]
        token_intervals[i] = (a, b)
        if want_hop:
            hop_spans[i] = interval_to_hop_span(a, b, float(hop), int(L_av or 0))
        time_mask[i] = 1 if (want_hop or b > a) else 0
    return (token_intervals, hop_spans, time_mask) if want_hop else (token_intervals, time_mask)

def word_alignment(text, audio, status, duration, work, cfg):
    words, empty = words_of(text), np.zeros((0, 2), dtype=np.float64)
    if not words:
        return [], empty, "mfa_failed", "none", "无可用转写词"
    work = Path(work)
    if audio is None:
        return words, empty, "mfa_failed", "none", str(status.get("audio_status", "no_audio"))
    try:
        from praatio import textgrid
        aligned, reused = ensure_dir(work / "aligned"), _reuse_textgrid(work, cfg)
        if not cfg.get("batch_alignment") and not reused:
            corpus = ensure_dir(work / "corpus")
            shutil.copyfile(work / "speech.wav", corpus / "clip.wav")
            (corpus / "clip.lab").write_text(" ".join(words), encoding="utf-8")
            command([cfg["mfa_executable"], "align", corpus, resolve_path(cfg["dictionary"]),
                     resolve_path(cfg["acoustic"]), aligned, "--clean", "--single_speaker",
                     "--num_jobs", "2", "--temporary_directory", work / "mfa_temp"], work / "mfa.log")
        candidates = list(aligned.rglob("*.TextGrid"))
        if len(candidates) != 1:
            return words, empty, "mfa_failed", "none", "no unique TextGrid"
        tg = textgrid.openTextgrid(str(candidates[0]), includeEmptyIntervals=False)
        tier = next(tg.getTier(name) for name in tg.tierNames if name.lower().endswith("words"))
        entries = [(float(e.start), float(e.end), str(e.label).lower()) for e in tier.entries if str(e.label).strip()]
        if [e[2] for e in entries] != words:
            return words, empty, "mfa_failed", "none", "word mismatch"
        off = status["audio_offset"]
        return words, np.array([[a + off, b + off] for a, b, _ in entries]), "mfa", "estimated_by_forced_alignment", ""
    except Exception as exc:
        return words, empty, "mfa_failed", "none", str(exc)

def aggregate(target, source, values, quality):
    overlap = np.maximum(0, np.minimum(target[:, None, 1], source[None, :, 1]) - np.maximum(target[:, None, 0], source[None, :, 0]))
    weights = overlap * quality[None, :]
    denom = weights.sum(1)
    return ((weights @ values) / np.maximum(denom[:, None], 1e-12)).astype("float32"), denom > 0

def view50(intervals, features, masks):
    count = min(len(intervals), 50)
    observed = np.zeros((3, 50), dtype=bool)
    observed[:, :count] = masks[:, :count]
    times = np.zeros((50, 2), dtype="float32")
    times[:count] = intervals[:count]
    view = {}
    for m, name in enumerate(("text", "audio", "vision")):
        result = np.zeros((50, features[name].shape[1]), dtype="float32")
        result[:count] = features[name][:count] * masks[m, :count, None]
        view[name] = result
    return view, observed, times

def pad_av_with_usable(synced, usable, L, dim):
    if len(synced) == 0:
        return pad_trunc_av(np.zeros((0, dim), dtype=np.float32), 0, L, dim)
    feat, mask, n = pad_trunc_av(synced, len(synced), L, dim)
    qmask = np.zeros(L, dtype=np.int8)
    qmask[:n] = usable[:n].astype(np.int8)
    feat[qmask == 0] = 0.0
    return feat, qmask, n
