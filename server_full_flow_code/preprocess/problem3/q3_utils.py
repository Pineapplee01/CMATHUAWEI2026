# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：GPT-6 Astra
# 版本/型号：GPT-6 Astra（gpt-6-astra）
# 开发机构/公司：OpenAI
# 版本颁布日期：2026年9月3日
# 发布信息来源：https://openai.com/index/safety-overview-gpt-6-astra/

"""路径解析、随机种子、JSON输出及附件数据读取。"""
from pathlib import Path
import json
import random
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_ROOT = ROOT / "AAA提交版代码及结果"
LABELS = ('Negative', 'Neutral', 'Positive')


def path(value):
    """数据和模型相对提交目录的父目录；提交结果跟随实际提交目录。"""
    p = Path(value)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == 'AAA提交版代码及结果':
        return SUBMISSION_ROOT.joinpath(*p.parts[1:])
    return ROOT / p



def resolve_data_config(config, layout):
    """将既有检查点中的附件2旧目录映射到当前数据位置，保留自定义数据路径。"""
    result = dict(config)
    current = f'AAAdata/Appendix_2/{layout}'
    known = [current, f'AAAdata/Appendix_2/标准化/{layout}/processed_po']
    if layout == '对齐版本':
        known.append('AAAdata/Appendix_2/标准化/对齐版本_retrain_20260925/processed_po')
    original = str(result.get('data_dir', '')).replace('\\', '/').rstrip('/')
    if any(original == item or original.endswith('/' + item) for item in known):
        result['data_dir'] = str(path(current))
    return result


def seed_all(seed):
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def json_write(filename, value):
    p = Path(filename)
    p.parent.mkdir(parents=True, exist_ok=True)
    def convert(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert,
                            allow_nan=False), encoding='utf-8')


import pickle


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
