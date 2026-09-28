"""三模态特征提取器及固定字段定义。"""
import numpy as np
import pandas as pd
import torch
from shared.common import path
from problem1.media import command

AUS = ('01','02','04','05','06','07','09','10','12','14','15','17','20','23','25','26','45')
VISION_NAMES = [f'AU{x}_r' for x in AUS] + [f'AU{x}_c' for x in (*AUS[:-1], '28', '45')]


def bert_words(words, tokenizer, encoder, device):
    # 完整词列表按子词容量分块，无超过512后静默截断；保存每词实际上下文块。
    counts = [len(tokenizer.tokenize(w)) for w in words]
    groups, start, size = [], 0, 0
    for i, count in enumerate(counts):
        if count > 510: raise ValueError('单词子词数超过 BERT 容量')
        if size + count > 510:
            groups.append((start, i)); start, size = i, 0
        size += count
    groups.append((start, len(words)))
    all_features, mappings = [], []
    with torch.no_grad():
        for lo, hi in groups:
            inputs = tokenizer(words[lo:hi], is_split_into_words=True, return_tensors='pt', truncation=False)
            ids = inputs.word_ids()
            hidden = encoder(**{k: v.to(device) for k, v in inputs.items()}).last_hidden_state[0].cpu().numpy()
            for word_index in range(hi-lo):
                positions = [j for j, wid in enumerate(ids) if wid == word_index]
                if not positions: raise ValueError('转写词无子词表示')
                all_features.append(hidden[positions].mean(0))
                mappings.append({'word_index': lo+word_index, 'token_positions': positions,
                                 'token_ids': inputs['input_ids'][0, positions].tolist(),
                                 'context_word_range': [lo, hi]})
    return np.asarray(all_features, dtype='float32'), mappings



def audio_features(audio, status):
    import opensmile
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
    names = list(smile.feature_names)
    if len(names) != 25: raise ValueError(f'eGeMAPSv02 LLD 实际维度 {len(names)} != 25')
    if audio is None:
        return np.empty((0, 2)), np.empty((0, 25)), np.empty(0), names
    signal, sr = audio
    frame = smile.process_signal(signal, sr)
    starts = frame.index.get_level_values('start').total_seconds().to_numpy()
    ends = frame.index.get_level_values('end').total_seconds().to_numpy()
    intervals = np.stack([starts, ends], -1) + status['audio_offset']
    values = frame.to_numpy(dtype='float32')
    good = np.isfinite(values).all(1)
    return intervals, np.nan_to_num(values), good.astype(float), names



def vision_features(video, work, info, cfg):
    output = work / 'openface'; output.mkdir(exist_ok=True)
    executable = path(cfg['openface_executable'])
    command([executable, '-f', video, '-out_dir', output, '-aus'],
            work / 'openface.log', cwd=executable.parent)
    frame = pd.read_csv(output / f'{video.stem}.csv')
    frame.columns = frame.columns.str.strip()
    absent = set(VISION_NAMES) - set(frame.columns)
    if absent: raise ValueError(f'OpenFace 缺少锁定AU字段: {sorted(absent)}')
    indices = frame['frame'].to_numpy(dtype=int)-1
    if len(np.unique(indices)) != len(indices) or (indices < 0).any() or (indices >= len(info['frames'])).any():
        raise ValueError('OpenFace frame 无法对应 ffprobe 显示帧顺序')
    times = info['frames']
    mid = (times[:-1]+times[1:])/2
    edges = np.r_[max(0, times[0]), mid,
                  min(info['duration'], times[-1]+info['last_frame_duration'])]
    intervals = np.stack([edges[indices], edges[indices+1]], -1)
    values = frame[VISION_NAMES].to_numpy(dtype='float32')
    quality = frame['confidence'].to_numpy() * (frame['success'].to_numpy() == 1)
    quality *= (quality >= cfg['vision_confidence']) & np.isfinite(values).all(1)
    return intervals, np.nan_to_num(values), quality, indices
