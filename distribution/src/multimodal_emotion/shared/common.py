from pathlib import Path
import json
import random
import hashlib
import importlib.metadata
import numpy as np
import torch

from multimodal_emotion.shared.paths import distribution_root

ROOT = distribution_root()
MODALITIES = ('text', 'audio', 'vision')
LABELS = ('Negative', 'Neutral', 'Positive')


def path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def config(filename):
    return json.loads(path(filename).read_text(encoding='utf-8'))


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


def fingerprint(directory):
    """锁定本地编码器与词表内容，不依赖可变的模型名称。"""
    result = {}
    names = ['config.json', 'tokenizer.json', 'model.safetensors']
    names += ['vocab.txt'] if (path(directory)/'vocab.txt').exists() else ['vocab.json','merges.txt']
    for name in names:
        p = path(directory) / name
        h = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(block)
        result[name] = h.hexdigest()
    return result


def environment():
    versions = {}
    for name in ('torch', 'transformers', 'numpy', 'pandas', 'opensmile', 'praatio',
                 'scikit-learn', 'soundfile'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
