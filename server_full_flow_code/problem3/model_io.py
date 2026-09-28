"""加载 problem2_retrain_v2 对齐 F1（样本级门控），并把附件4转为模型输入。"""
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from shared.common import path
from shared.data import load_fields
from problem2_retrain_v2.data import sha256
from problem2_retrain_v2.inference import raw_to_inputs
from problem2_retrain_v2.model import build_model
from problem2_retrain_v2.output_contract import project_intensity


def load_model(cfg):
    ckpt = path(cfg['checkpoint'])
    digest = sha256(ckpt)  # 仅记录，不强制与 config 匹配
    saved = torch.load(ckpt, map_location='cpu', weights_only=False)
    if saved.get('format') != 'aligned_v4':
        raise ValueError('需要对齐 aligned_v4 检查点')
    train = saved['config']
    if train.get('architecture', 'f1') not in ('f1', 'f3'):
        raise ValueError(f"期望 F1 检查点，得到 architecture={train.get('architecture')}")
    device = cfg.get('device', 'cpu')
    model = build_model(train, pretrained=False).to(device)
    model.load_state_dict(saved['model'], strict=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(path('AAAmodel/bert-base-uncased'), local_files_only=True)
    return model, train, saved, tok, digest


def appendix_batch(file, train_cfg, device):
    fields = load_fields(file)
    # encoder=None：finetune_bert 检查点由模型在线 BERT；仍建 P/O 并做音视一次 scaler
    batch = raw_to_inputs(fields, train_cfg['data_dir'], encoder=None)
    batch['I'] = torch.from_numpy(np.asarray(fields['text_bert'])[:, 0].astype(np.int64))
    out = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }
    text = str(fields['raw_text'][0])
    tokens = AutoTokenizer.from_pretrained(
        path('AAAmodel/bert-base-uncased'), local_files_only=True
    ).convert_ids_to_tokens(out['I'][0].tolist())
    return out, text, tokens, str(fields.get('id', [Path(file).stem])[0])


@torch.no_grad()
def predict(model, batch):
    out = model(batch)
    p = out['logits'].softmax(-1)[0]
    c = int(p.argmax())
    raw = float(out['intensity'][0])
    y = float(project_intensity(np.array([c]), np.array([raw]))[0])
    return {'p': p.cpu().numpy(), 'c': c, 'y': y, 'raw': raw}


def masked(batch, keep):
    """keep: (3,L) bool；文本缺失时清零对应 token（在线 BERT 前向会读 I/O）。"""
    b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    b['O'] = keep.bool() & batch['O'] & batch['P']
    return b
