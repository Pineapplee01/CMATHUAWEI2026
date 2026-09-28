"""标准化 npz → 管道 pkl 转换器：让现有训练/评估管道直接消费预处理产物。

一致性断言：标签/样本ID/词元与原始对齐 pkl 全等；z-score 后的音视频
非零掩码与预处理掩码 mA/mV 一致（保证 FeatureDataset 推导的 O 不漂移）。
XT（冻结BERT逐词元嵌入）写入 text 字段，由管道 text768 通路透传给
precomputed 编码器。
"""
import argparse
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from shared.data import load_fields
from shared.common import path


def convert_split(split, processed_dir, source_pkl, destination):
    d = np.load(path(processed_dir) / f'{split}.npz', allow_pickle=True)
    src = load_fields(path(source_pkl) / f'{split}.pkl', split)

    labels = d['classification_labels'].astype(int)
    assert np.array_equal(labels, np.asarray(src['classification_labels']).reshape(-1).astype(int)), f'{split}: 标签不一致'
    assert np.allclose(d['regression_labels'], np.asarray(src['regression_labels'], dtype=np.float32).reshape(-1)), f'{split}: 回归标签不一致'
    ids = [str(x) for x in d['id']]
    assert ids == [str(x) for x in src['id']], f'{split}: 样本顺序不一致'
    tb = np.asarray(src['text_bert'])
    assert np.array_equal(tb[:, 0, :], d['I']), f'{split}: 词元与 I 不一致'

    audio_z, vision_z = d['XA'], d['XV']
    mA, mV = d['mA'], d['mV']
    assert np.array_equal(np.any(audio_z != 0, axis=-1), mA), f'{split}: z-score后音频掩码漂移'
    assert np.array_equal(np.any(vision_z != 0, axis=-1), mV), f'{split}: z-score后视觉掩码漂移'

    out = {
        'raw_text': src['raw_text'],
        'audio': audio_z.astype(np.float32),
        'vision': vision_z.astype(np.float32),
        'id': ids,
        'text_bert': tb.copy(),
        'classification_labels': labels,
        'regression_labels': np.asarray(src['regression_labels']).reshape(-1),
        'text': d['XT'].astype(np.float32),
    }
    destination.mkdir(parents=True, exist_ok=True)
    with open(destination / f'{split}.pkl', 'wb') as f:
        import pickle
        pickle.dump(out, f)
    print(f'{split}: {len(ids)} 条 -> {destination}/{split}.pkl (XT {out["text"].shape})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed', default='AAAdata/Appendix_2/标准化/对齐版本/processed')
    parser.add_argument('--source', default='AAAdata/Appendix_2/原始未预处理/对齐版本')
    parser.add_argument('--destination', default='AAAdata/Appendix_2/标准化/对齐版本/pkl')
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'])
    args = parser.parse_args()
    destination = path(args.destination)
    for split in args.splits:
        convert_split(split, args.processed, args.source, destination)
