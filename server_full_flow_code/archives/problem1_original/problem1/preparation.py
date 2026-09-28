"""批量准备媒体与MFA：100条共用一次词典导入，保留逐样本状态。"""
import shutil
from shared.common import path
from problem1.media import media_info, audio_track, command
from problem1.alignment import words_of


def prepare_batch(labels, cfg):
    q=cfg['q1']; root=path(q['root']); work_root=path(q['work'])
    corpus=work_root/'mfa_corpus'; aligned=work_root/'mfa_aligned'
    corpus.mkdir(parents=True,exist_ok=True); aligned.mkdir(parents=True,exist_ok=True)
    prepared={}
    for row in labels.itertuples(index=False):
        sid=f'{row.video_id}$_${row.clip_id}'; work=work_root/f'{row.video_id}__{row.clip_id}'
        work.mkdir(parents=True,exist_ok=True)
        try:
            video=root/row.video_id/f'{row.clip_id}.mp4'
            info=media_info(video)
            try:
                audio,status=audio_track(video,work,info)
            except (RuntimeError,ValueError) as exc:
                audio,status=None,{'audio_status':'decode_failed','audio_feature_available':False,
                                  'decode_ok':False,'silence_cause':'unknown','error':str(exc)}
            prepared[sid]={'info':info,'audio':audio,'status':status}
            speaker=corpus/row.video_id; speaker.mkdir(exist_ok=True)
            if audio is not None:
                shutil.copyfile(work/'speech.wav',speaker/f'{row.clip_id}.wav')
                (speaker/f'{row.clip_id}.lab').write_text(' '.join(words_of(str(row.text))))
        except Exception as exc:
            prepared[sid]={'error':repr(exc)}
    print(f'MFA批量对齐：准备{len(prepared)}条媒体，共用一次词典和声学模型。',flush=True)
    try:
        command([q['mfa_executable'],'align',corpus,path(q['dictionary']),path(q['acoustic']),
                 aligned,'--clean','--num_jobs','4','--temporary_directory',work_root/'mfa_temp'],
                work_root/'mfa_batch.log')
        for row in labels.itertuples(index=False):
            source=aligned/row.video_id/f'{row.clip_id}.TextGrid'
            target=work_root/f'{row.video_id}__{row.clip_id}'/'aligned'/'clip.TextGrid'
            if source.is_file():
                target.parent.mkdir(exist_ok=True); shutil.copyfile(source,target)
    except RuntimeError as exc:
        print(f'MFA批处理失败，各样本将显式记录降级原因：{exc}',flush=True)
    q['batch_alignment']=True
    return prepared
