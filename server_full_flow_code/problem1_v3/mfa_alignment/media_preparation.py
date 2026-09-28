"""附件批量 MFA（可选）。来源：problem3/media_preparation.py，包路径改为本目录。"""
import shutil

from shared.common import path
from shared.data import load_fields

from problem1_v3.mfa_alignment.media import audio_track, command, media_info
from problem1_v3.mfa_alignment.word_alignment import words_of


def prepare_media(files, cfg, work_root="problem1_v3/work"):
    q = cfg["q1"]
    root = path(cfg.get("evidence_work", work_root)) / "mfa_batch"
    corpus, aligned = root / "corpus", root / "aligned"
    corpus.mkdir(parents=True, exist_ok=True)
    prepared = {}
    for file in files:
        work = root / file.stem
        work.mkdir(exist_ok=True)
        try:
            info = media_info(file.parent / "videos" / f"{file.stem}.mp4")
            audio, status = audio_track(
                file.parent / "videos" / f"{file.stem}.mp4", work, info
            )
            prepared[file.stem] = (info, audio, status)
            if audio is not None:
                speaker = corpus / file.stem
                speaker.mkdir(exist_ok=True)
                shutil.copyfile(work / "speech.wav", speaker / "clip.wav")
                raw = str(load_fields(file)["raw_text"][0])
                (speaker / "clip.lab").write_text(
                    " ".join(words_of(raw)), encoding="utf-8"
                )
        except Exception as exc:
            prepared[file.stem] = exc
    try:
        command(
            [
                q["mfa_executable"],
                "align",
                corpus,
                path(q["dictionary"]),
                path(q["acoustic"]),
                aligned,
                "--clean",
                "--num_jobs",
                "4",
                "--temporary_directory",
                root / "mfa_temp",
            ],
            root / "mfa_batch.log",
        )
    except RuntimeError as exc:
        print(f"批量对齐失败，将在逐条映射中记录降级：{exc}", flush=True)
    for file in files:
        source = aligned / file.stem / "clip.TextGrid"
        target = root / file.stem / "aligned" / "clip.TextGrid"
        if source.is_file():
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, target)
    return prepared
