"""强制对齐、区间交叠聚合与50位置组织。"""
import re
import shutil
import numpy as np
from shared.common import path
from problem1.media import command



def words_of(text):
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*", text.lower())



def word_alignment(text, audio, status, duration, work, cfg):
    words = words_of(text)
    if not words: raise ValueError('无可用转写词')
    failure = ''
    if audio is not None:
        try:
            from praatio import textgrid
            aligned = work / 'aligned'; aligned.mkdir(exist_ok=True)
            if not cfg.get('batch_alignment'):
                corpus = work / 'corpus'; corpus.mkdir(exist_ok=True)
                shutil.copyfile(work / 'speech.wav', corpus / 'clip.wav')
                (corpus / 'clip.lab').write_text(' '.join(words), encoding='utf-8')
                command([cfg['mfa_executable'], 'align', corpus, path(cfg['dictionary']),
                         path(cfg['acoustic']), aligned, '--clean', '--single_speaker',
                         '--num_jobs', '2', '--temporary_directory', work / 'mfa_temp'], work / 'mfa.log')
            candidates = list(aligned.rglob('*.TextGrid'))
            if len(candidates) != 1: raise ValueError('MFA 未输出唯一 TextGrid')
            tg = textgrid.openTextgrid(str(candidates[0]), includeEmptyIntervals=False)
            tier = next(tg.getTier(name) for name in tg.tierNames if name.lower().endswith('words'))
            entries = [(float(e.start), float(e.end), str(e.label).lower()) for e in tier.entries
                       if str(e.label).strip()]
            if [e[2] for e in entries] != words:
                raise ValueError('MFA 词序列与规范化转写不一致，不能静默错配')
            intervals = np.array([[a+status['audio_offset'], b+status['audio_offset']] for a,b,_ in entries])
            validate_intervals(intervals, duration)
            return words, intervals, 'mfa', 'estimated_by_forced_alignment', failure
        except (RuntimeError, ValueError, StopIteration, FileNotFoundError, ImportError) as exc:
            failure = str(exc)
    else:
        failure = status['audio_status']
    if not cfg['fallback_uniform']:
        raise RuntimeError(f'强制对齐不可用: {failure}')
    edges = np.linspace(0, duration, len(words)+1)
    return words, np.stack([edges[:-1], edges[1:]], -1), 'uniform_fallback', 'low_approximate', failure



def validate_intervals(intervals, duration):
    if (not np.isfinite(intervals).all() or np.any(intervals[:, 1] <= intervals[:, 0])
        or intervals[0, 0] < -1e-4 or intervals[-1, 1] > duration+1e-3
        or np.any(intervals[1:, 0] < intervals[:-1, 1]-1e-4)):
        raise ValueError('时间区间不合法/不单调/越界')



def aggregate(target, source, values, quality):
    """交叠时长×质量加权；零分母置零并返回不可用掩码及精确源索引。"""
    overlap = np.maximum(0, np.minimum(target[:, None, 1], source[None, :, 1]) -
                         np.maximum(target[:, None, 0], source[None, :, 0]))
    weights = overlap * quality[None, :]
    denom = weights.sum(1)
    features = (weights @ values) / np.maximum(denom[:, None], 1e-12)
    mapping = [{'indices': np.flatnonzero(w > 0).tolist(),
                'weights': (w[w > 0] / max(w.sum(), 1e-12)).tolist()} for w in weights]
    return features.astype('float32'), denom > 0, mapping



def view50(intervals, features, masks):
    """保留前50个位置，三模态和时间轴同步截断；短序列右侧补零。"""
    count = min(len(intervals), 50)
    observed = np.zeros((3, 50), dtype=bool)
    observed[:, :count] = masks[:, :count]
    times = np.zeros((50, 2), dtype='float32')
    times[:count] = intervals[:count]
    view = {}
    for m, name in enumerate(('text', 'audio', 'vision')):
        result = np.zeros((50, features[name].shape[1]), dtype='float32')
        result[:count] = features[name][:count] * masks[m, :count, None]
        view[name] = result
    return view, observed, times, [[i] for i in range(count)]


def view50_metadata(length):
    return {'view50_policy': 'truncate_prefix_50',
            'view50_index_map': [[i] for i in range(min(length, 50))],
            'view50_original_units': length, 'view50_retained_units': min(length, 50),
            'view50_dropped_units': max(length-50, 0),
            'view50_truncated': length > 50,
            'full_length_archive_retained': True}



def gaps(intervals, duration):
    result, cursor = [], 0.0
    for a, b in intervals:
        if a > cursor: result.append([cursor, float(a)])
        cursor = max(cursor, float(b))
    if cursor < duration: result.append([cursor, duration])
    return result
