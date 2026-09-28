"""媒体探测、音频质量与外部命令。"""
import json
import subprocess
import os
from pathlib import Path
import numpy as np



def command(args, log, cwd=None):
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    result = subprocess.run([str(x) for x in args], text=True, capture_output=True, cwd=cwd, env=env)
    Path(log).write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'命令失败 {args[0]}，日志 {log}')
    return result.stdout



def media_info(video):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_format', '-show_streams',
                             '-of', 'json', str(video)], check=True, capture_output=True, text=True)
    info = json.loads(result.stdout)
    duration = float(info['format']['duration'])
    origin = float(info['format'].get('start_time', 0))
    frame_result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                  '-show_frames', '-show_entries',
                                  'frame=best_effort_timestamp_time,pkt_duration_time',
                                  '-of', 'json', str(video)], check=True, capture_output=True, text=True)
    frames = json.loads(frame_result.stdout)['frames']
    ts = np.array([float(f['best_effort_timestamp_time']) - origin for f in frames])
    if not len(ts) or np.any(np.diff(ts) <= 0):
        raise ValueError('显示帧时间戳为空或非严格递增')
    dt = float(frames[-1].get('pkt_duration_time', np.median(np.diff(ts)) if len(ts)>1 else duration))
    return {'duration': duration, 'origin': origin, 'frames': ts,
            'last_frame_duration': dt, 'streams': info['streams']}



def audio_track(video, work, info):
    import soundfile as sf
    streams = [s for s in info['streams'] if s['codec_type'] == 'audio']
    if not streams:
        return None, {'audio_status': 'no_audio_track', 'audio_track_present': False,
                      'decode_ok': False, 'audio_feature_available': False,
                      'audio_observation_state': 'unavailable', 'silence_cause': 'unknown'}
    wav = work / 'all_channels.wav'
    command(['ffmpeg', '-y', '-v', 'error', '-i', video, '-map', '0:a:0', '-ar', '16000',
             '-c:a', 'pcm_f32le', wav], work / 'audio_decode.log')
    signal, sr = sf.read(wav, always_2d=True, dtype='float32')
    if not len(signal) or not np.isfinite(signal).all():
        raise ValueError('空音轨或非有限音频')
    peaks = np.max(np.abs(signal), axis=0)
    rms = np.sqrt(np.mean(signal.astype(np.float64)**2, axis=0))
    selected = int(rms.argmax())
    # 不用立体声均值，避免相位抵消造成假静音；仅全声道严格全零才跳过。
    silent = bool(np.all(peaks == 0))
    status = {'audio_status': 'silent_track' if silent else 'decoded',
              'silence_cause': 'unknown', 'audio_track_present': True, 'decode_ok': True,
              'audio_feature_available': not silent, 'sample_rate': sr,
              'channels': signal.shape[1], 'decoded_samples': len(signal),
              'peak_per_channel': peaks, 'rms_per_channel': rms,
              'selected_channel': selected, 'silence_threshold': 0.0,
              'audio_observation_state': 'unknown_silence' if silent else 'signal_present',
              'audio_offset': float(streams[0].get('start_time', info['origin']))-info['origin']}
    mono = signal[:, selected]
    sf.write(work / 'speech.wav', mono, sr, subtype='PCM_16')
    return None if silent else (mono, sr), status
