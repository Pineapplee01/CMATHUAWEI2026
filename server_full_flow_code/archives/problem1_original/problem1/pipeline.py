"""问题1全量提取与异常清单。"""
import shutil
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from shared.common import path, json_write, environment, fingerprint
from problem1.media import media_info, audio_track
from problem1.alignment import word_alignment, aggregate, view50, gaps, view50_metadata
from problem1.extractors import bert_words, audio_features, vision_features, VISION_NAMES
from problem1.visualization import alignment_plots
from problem1.preparation import prepare_batch



def extract(cfg):
    torch.set_num_threads(4)
    q = cfg['q1']; output = path(q['output']); output.mkdir(parents=True, exist_ok=True)
    root, work_root = path(q['root']), path(q['work'])
    for executable in ('ffmpeg', 'ffprobe', q['openface_executable']):
        if shutil.which(executable) is None:
            raise FileNotFoundError(f'必需工具未找到: {executable}')
    # 原始标签只读；QC 是辅助信息，不作为替换标签的来源。
    labels = pd.read_excel(path(q['labels']), sheet_name='label', dtype={'video_id': str, 'clip_id': str})
    if len(labels) != 100: raise ValueError('附件1必须保留全部100行')
    intensity = pd.to_numeric(labels['label'], errors='raise')
    if not np.isfinite(intensity).all() or (intensity.abs() > 3).any():
        raise ValueError('附件1强度标签非法')
    expected = np.array(['Negative','Neutral','Positive'])[np.sign(intensity).astype(int)+1]
    if not np.array_equal(expected, labels.annotation.to_numpy()):
        raise ValueError('附件1极性与强度符号不一致')
    keys = labels.video_id + '$_$' + labels.clip_id
    if keys.duplicated().any(): raise ValueError('附件1样本主键重复')
    if q.get('smoke_limit'):
        labels = labels.head(q['smoke_limit'])
    qc_lookup = {}
    if q.get('qc') and path(q['qc']).is_file():
        qc = pd.read_csv(path(q['qc']), dtype={'video_id': str, 'clip_id': str})
        qc_lookup = {(r.video_id, r.clip_id): r._asdict() for r in qc.itertuples(index=False)}
    else:
        print('旧质量清单不可用，将从原始视频重新核验媒体质量。', flush=True)
    encoder_directory = q.get('bert_path', cfg['bert_path'])
    tokenizer = AutoTokenizer.from_pretrained(path(encoder_directory), local_files_only=True, use_fast=True)
    encoder = AutoModel.from_pretrained(path(encoder_directory), local_files_only=True).to(cfg['device']).eval()
    encoder.requires_grad_(False)
    encoder_identity = fingerprint(encoder_directory)
    json_write(output / 'extraction_config.json', {'config':cfg, 'environment':environment(),
                                                 'encoder_sha256':encoder_identity})
    if encoder.config.hidden_size != 768: raise ValueError('BERT 宽度必须为768')
    prepared = prepare_batch(labels, cfg) if len(labels)>1 else {}
    manifest, failures, problems = [], [], []
    for row in labels.itertuples(index=False):
        sid = f'{row.video_id}$_${row.clip_id}'
        stem = f'{row.video_id}__{row.clip_id}'
        video = root / row.video_id / f'{row.clip_id}.mp4'
        work = work_root / stem; work.mkdir(parents=True, exist_ok=True)
        record = {'id': sid, 'video': str(video.relative_to(root)), 'label': float(row.label),
                  'annotation': row.annotation, 'text_width': 768, 'audio_width': 25, 'vision_width': 35,
                  'status': 'failed'}
        try:
            if prepared:
                item = prepared[sid]
                if 'error' in item: raise RuntimeError(item['error'])
                info,audio,status = item['info'],item['audio'],item['status']
            else:
                info = media_info(video)
                try:
                    audio, status = audio_track(video, work, info)
                except (ValueError, RuntimeError) as exc:
                    audio, status = None, {'audio_status': 'decode_failed', 'audio_feature_available': False,
                                           'decode_ok': False, 'silence_cause': 'unknown', 'error': str(exc)}
            if status['audio_status']=='decode_failed':
                failures.append({'id':sid,'stage':'audio_decode','error':status['error']})
            words, intervals, method, confidence, failure = word_alignment(str(row.text), audio, status, info['duration'], work, q)
            text, word_map = bert_words(words, tokenizer, encoder, cfg['device'])
            ai, av, aq, audio_names = audio_features(audio, status)
            vi, vv, vq, frame_ids = vision_features(video, work, info, q)
            schemas = {'text': [f'bert_last_hidden_{i}' for i in range(768)],
                       'audio': audio_names, 'vision': VISION_NAMES}
            metadata = {'id': sid, 'original_text': str(row.text), 'words': words,
                        'duration': info['duration'], 'audio_quality': status,
                        'existing_qc': qc_lookup.get((row.video_id, row.clip_id), {}),
                        'requested_alignment': 'mfa', 'actual_alignment': method,
                        'timing_confidence': confidence, 'failure_reason': failure,
                        'word_token_map': word_map, 'feature_schema_id': 'q1_bert768_egemaps25_au35_v1',
                        'feature_names': schemas, 'extractor_revision': environment(),
                        'openface_revision': q['openface_revision'], 'normalization_id': 'none',
                        'support_scope': {'text': 'BERT full retained chunk; see word_token_map',
                                          'audio': 'openSMILE analysis windows and smoothing context',
                                          'vision': 'OpenFace tracking context; not only display interval'},
                        'audio_source_intervals': ai, 'vision_source_intervals': vi,
                        'audio_source_quality': aq, 'vision_source_quality': vq,
                        'vision_frame_indices': frame_ids, 'display_frame_times': info['frames'],
                        'nonword_intervals': gaps(intervals, info['duration'])}
            for variant in ('word', 'uniform'):
                dest = output / variant; dest.mkdir(exist_ok=True)
                if variant == 'word': target = intervals
                else:
                    edges = np.linspace(0, info['duration'], len(words)+1)
                    target = np.stack([edges[:-1], edges[1:]], -1)
                t, tm, tmap = aggregate(target, intervals, text, np.ones(len(words)))
                a, am, amap = aggregate(target, ai, av, aq)
                v, vm, vmap = aggregate(target, vi, vv, vq)
                features = {'text': t, 'audio': a, 'vision': v}
                masks = np.stack([tm,am,vm])
                view, viewmask, times, groups = view50(target, features, masks)
                arrays = {**features, 'intervals': target.astype('float32'), 'observed_mask': masks,
                          'valid_mask': np.ones_like(masks), 'view50_intervals': times,
                          'view50_observed_mask': viewmask,
                          'view50_valid_mask': np.broadcast_to(np.arange(50)<min(len(target),50), (3,50))}
                arrays.update({f'view50_{name}': x for name,x in view.items()})
                np.savez_compressed(dest / f'{stem}.npz', **arrays)
                # JSON 不存 QC 中的 NaN；空值保留为 None。
                metadata['existing_qc'] = {k: None if isinstance(v,float) and not np.isfinite(v) else v
                                           for k,v in metadata['existing_qc'].items()}
                json_write(dest / f'{stem}.json', {**metadata, 'variant': variant,
                           'source_frame_map': {'text': tmap, 'audio': amap, 'vision': vmap},
                           **view50_metadata(len(target)),
                           'uniform_text_depends_on_word_timing': variant=='uniform'})
                manifest.append({**record, 'variant': variant, 'status': 'ok',
                                 'duration': info['duration'], 'units': len(target),
                                 'alignment': method if variant=='word' else 'uniform_bins',
                                 'word_timing_source': method, 'timing_confidence': confidence,
                                 'audio_status': status['audio_status'],
                                 'text_coverage': float(tm.mean()), 'audio_coverage': float(am.mean()),
                                 'vision_coverage': float(vm.mean()), 'filename': f'{variant}/{stem}.npz'})
                if len(manifest) <= 4:
                    alignment_plots(stem+'_'+variant, target, masks, path('problem1/figures'))
            # 边界扰动仅检查表示变化，不把自提特征输入附件2模型。
            rng = np.random.default_rng(cfg['seed']); sensitivity = []
            original_audio = aggregate(intervals, ai, av, aq)[0]
            original_vision = aggregate(intervals, vi, vv, vq)[0]
            for amplitude in (.05, .1, .2):
                for repeat in range(10):
                    shifted = intervals.copy()
                    for i,(left,right) in enumerate(intervals):
                        lo = 0 if i==0 else (intervals[i-1,1]+left)/2
                        hi = info['duration'] if i==len(intervals)-1 else (right+intervals[i+1,0])/2
                        shifted[i] = np.clip([left+rng.uniform(-amplitude,amplitude),
                                              right+rng.uniform(-amplitude,amplitude)], lo, hi)
                        if shifted[i,1] <= shifted[i,0]: shifted[i] = intervals[i]
                    sensitivity.append({'amplitude_seconds': amplitude, 'repeat': repeat,
                        'audio_mse': float(np.mean((aggregate(shifted,ai,av,aq)[0]-original_audio)**2)),
                        'vision_mse': float(np.mean((aggregate(shifted,vi,vv,vq)[0]-original_vision)**2))})
            json_write(output / 'sensitivity' / f'{stem}.json', sensitivity)
            if failure or status['audio_status'] != 'decoded':
                problems.append({'id': sid, 'audio_status': status['audio_status'],
                                 'alignment': method, 'reason': failure})
        except Exception as exc:
            # 失败样本仍有清单记录，绝不生成伪装成功的零特征。
            manifest = [r for r in manifest if r['id'] != sid]
            failures.append({'id': sid, 'stage': 'extract', 'error': repr(exc)})
            for variant in ('word','uniform'):
                manifest.append({**record, 'variant': variant, 'error': repr(exc)})
        pd.DataFrame(manifest).to_csv(output / 'manifest_all.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(failures, columns=['id','stage','error']).to_csv(output / 'failures.csv', index=False)
        pd.DataFrame(problems, columns=['id','audio_status','alignment','reason']).to_csv(output / 'problematic.csv', index=False)
        print({'processed':len(manifest)//2, 'expected':len(labels), 'id':sid,
               'status':manifest[-1]['status'], 'alignment':manifest[-1].get('word_timing_source'),
               'error':manifest[-1].get('error')}, flush=True)
    frame = pd.DataFrame(manifest)
    for variant in ('word','uniform'):
        selected = frame[frame.variant==variant]
        if len(selected) != len(labels): raise AssertionError('清单未保留全部计划样本')
        selected.to_csv(output / f'manifest_{variant}.csv', index=False, encoding='utf-8-sig')
    if failures:
        raise RuntimeError(f'提取存在 {len(failures)} 条失败记录，见 failures.csv；未通过完整性验收')
