"""只读附件；P/O/B 分离。默认范围规则是待验证假设而非真实时间元数据。"""
import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from shared.common import MODALITIES


class NumpyCompatibleUnpickler(pickle.Unpickler):
    """兼容NumPy 2写入的内部模块名，不修改用户的原始pickle。"""
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        return super().find_class(module, name)


def load_fields(filename, split=None, allow_unaligned_pool=False):
    # 仅用于用户提供且可信的本地 pickle 文件。
    with Path(filename).open('rb') as f:
        data = NumpyCompatibleUnpickler(f).load()
    if split and split in data:
        data = data[split]
    elif 'test' in data and isinstance(data['test'], dict):
        data = data['test']
    if np.asarray(data['text_bert']).ndim == 2:
        data = {k: np.expand_dims(v, 0) for k, v in data.items()}
    n = len(data['text_bert'])
    audio = np.asarray(data['audio'])
    vision = np.asarray(data['vision'])
    if allow_unaligned_pool and audio.ndim == 3 and audio.shape[1] == 500:
        al = data.get('audio_lengths')
        vl = data.get('vision_lengths')
        if al is None:
            al = [audio.shape[1]] * n
        if vl is None:
            vl = [vision.shape[1]] * n
        al = np.atleast_1d(np.asarray(al, dtype=np.int64))
        vl = np.atleast_1d(np.asarray(vl, dtype=np.int64))
        data['audio'] = np.stack([_pool_time_to_50(audio[i], int(al[i])) for i in range(n)])
        data['vision'] = np.stack([_pool_time_to_50(vision[i], int(vl[i])) for i in range(n)])
        data['unaligned_pool'] = True
    for name, shape in [('text_bert', (n, 3, 50)), ('audio', (n, 50, 74)),
                        ('vision', (n, 50, 35))]:
        arr = np.asarray(data[name])
        if arr.shape != shape or not np.isfinite(arr).all():
            raise ValueError(f'{filename}: {name} 应为 {shape} 且全部有限，实际 {arr.shape}')
    tokens = np.asarray(data['text_bert'])
    if not np.equal(tokens, np.floor(tokens)).all():
        raise ValueError('text_bert 必须为整数')
    if not np.isin(tokens[:, 1], [0, 1]).all():
        raise ValueError('attention_mask 必须为 0/1')
    return data


def _pool_time_to_50(x, valid_len):
    """将未对齐 hop 序列按有效长度等分池化到 50 槽（仅非全零帧参与均值）。"""
    x = np.asarray(x, dtype=np.float64)
    t = int(max(1, min(valid_len, x.shape[0])))
    edges = np.linspace(0, t, 51)
    out = np.zeros((50, x.shape[1]), dtype=np.float64)
    for i in range(50):
        a = int(edges[i]); b = int(edges[i + 1])
        if b <= a:
            b = min(a + 1, t)
        seg = x[a:b]
        mask = np.any(seg != 0, axis=-1)
        if mask.any():
            out[i] = seg[mask].mean(0)
    return out


class FeatureDataset(Dataset):
    def __init__(self, fields, source, rule='union_extent', labels=False):
        self.fields, self.source, self.rule = fields, str(source), rule
        self.has_labels = labels
        self.ids = [str(x) for x in fields.get('id',
                    [f'{Path(source).stem}:{i}' for i in range(len(fields['text_bert']))])]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError('样本 id 重复')
        if labels:
            y = np.asarray(fields['regression_labels']).reshape(-1)
            c = np.asarray(fields['classification_labels']).reshape(-1)
            if not np.isfinite(y).all() or np.any(abs(y) > 3):
                raise ValueError('强度标签超界')
            if not np.array_equal(c, np.sign(y).astype(int) + 1):
                raise ValueError('分类与强度符号不一致')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        f = self.fields
        tb = np.asarray(f['text_bert'][i], dtype=np.int64)
        special = np.isin(tb[0], [0, 101, 102])
        ot = (tb[1] > 0) & ~special
        oa = np.any(np.asarray(f['audio'][i]) != 0, axis=-1)
        ov = np.any(np.asarray(f['vision'][i]) != 0, axis=-1)
        support = (tb[1] > 0) | oa | ov
        if self.rule == 'union_extent':
            positions = np.flatnonzero(support)
            p = np.arange(50) <= (positions[-1] if len(positions) else -1)
        elif self.rule == 'attention_extent':
            positions = np.flatnonzero(tb[1] > 0)
            p = np.arange(50) <= (positions[-1] if len(positions) else -1)
        elif self.rule == 'all_slots':
            p = np.ones(50, dtype=bool)
        else:
            raise ValueError(f'未知有效范围假设: {self.rule}')
        P = np.stack([p & ~np.isin(tb[0], [101, 102]), p, p])
        O = np.stack([ot, oa, ov]) & P
        result = {'tokens': torch.from_numpy(tb.copy()),
                  'audio': torch.tensor(f['audio'][i], dtype=torch.float32),
                  'vision': torch.tensor(f['vision'][i], dtype=torch.float32),
                  'P': torch.from_numpy(P), 'O': torch.from_numpy(O),
                  'id': self.ids[i], 'validity_uncertain': True}
        if 'text' in f:
            result['text768'] = torch.from_numpy(np.ascontiguousarray(
                f['text'][i], dtype=np.float32))
        if self.has_labels:
            result['y'] = torch.tensor(float(np.asarray(f['regression_labels'][i]).item()))
            result['c'] = torch.tensor(int(np.asarray(f['classification_labels'][i]).item()))
        return result


def assert_disjoint(*datasets):
    for i, a in enumerate(datasets):
        for b in datasets[i + 1:]:
            if set(a.ids) & set(b.ids):
                raise ValueError('划分之间样本重叠')
            if {s.split('$_$')[0] for s in a.ids} & {s.split('$_$')[0] for s in b.ids}:
                raise ValueError('划分之间原视频重叠')


def fit_normalizer(train):
    stats = {}
    for m, name in enumerate(MODALITIES[1:], 1):
        total = np.zeros(train.fields[name].shape[-1], dtype=np.float64)
        square, count = total.copy(), 0
        for i in range(len(train)):
            item = train[i]
            rows = item[name].numpy()[item['O'][m].numpy()]
            total += rows.sum(0, dtype=np.float64)
            square += np.square(rows.astype(np.float64)).sum(0)
            count += len(rows)
        if count == 0:
            raise ValueError(f'train 中 {name} 没有有效观测')
        mean = total / count
        std = np.sqrt(np.maximum(square / count - mean**2, 1e-8))
        stats[name] = {'mean': mean.tolist(), 'std': std.tolist(), 'count': count}
    return stats


def block_mask(batch, rate, modalities, rng, position='random'):
    """在序列上的单个连续区间遮挡；实际新增遮挡数量另行记录。"""
    B = torch.ones_like(batch['O'])
    if not 0 <= rate <= 1:
        raise ValueError('缺失比例必须位于 [0,1]')
    for i in range(B.shape[0]):
        for m in modalities:
            valid = torch.where(batch['P'][i, m])[0].cpu().numpy()
            if not len(valid) or rate == 0:
                continue
            length = max(1, int(round(rate * len(valid))))
            # 按有效区间跨度选择连续块，特殊词元/原缺失不会成为重建目标。
            lo, hi = int(valid[0]), int(valid[-1]) + 1
            length = min(length, hi - lo)
            start = {'start': lo, 'middle': lo + (hi-lo-length)//2,
                     'end': hi-length}.get(position)
            if start is None:
                start = int(rng.integers(lo, hi - length + 1))
            B[i, m, start:start + length] = False
    return B


def verify_tokenizer(train, tokenizer):
    """只用 train 核对全部三路输入；不匹配时阻止 BERT 语义错配。"""
    failures = []
    for i, raw in enumerate(train.fields['raw_text']):
        encoded = tokenizer(str(raw), padding='max_length', truncation=True, max_length=50)
        candidate = np.array([encoded['input_ids'], encoded['attention_mask'],
                              encoded.get('token_type_ids', [0] * 50)])
        if not np.array_equal(candidate, train.fields['text_bert'][i]):
            failures.append(train.ids[i])
    if failures:
        raise ValueError(f'tokenizer 身份核验失败 {len(failures)}/{len(train)}，示例 {failures[:5]}；'
                         '请核实原始分词配置，禁止绕过后训练')
    return {'checked': len(train), 'matched': len(train), 'max_length': 50,
            'padding': 'max_length', 'truncation': True}
