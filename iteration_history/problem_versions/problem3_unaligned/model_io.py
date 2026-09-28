"""加载问题二 SoftAlignment，附件4未对齐 50/500/500 输入（不池化到50）。"""
from pathlib import Path
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer
from shared.common import path
from problem2.data import sha256
from problem2.model import build_model
from problem2.output_contract import project_intensity


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        return super().find_class(module, name)


def load_model(cfg):
    ckpt = path(cfg['checkpoint'])
    digest = sha256(ckpt)  # 仅记录，不强制与 config 匹配
    saved = torch.load(ckpt, map_location='cpu', weights_only=False)
    if saved.get('format') != 'unaligned_v1':
        raise ValueError('需要 unaligned_v1 检查点')
    train = saved['config']
    if train.get('architecture') != 'soft_alignment':
        raise ValueError('需要 soft_alignment 架构')
    device = cfg.get('device', 'cpu')
    model = build_model(train, pretrained=False).to(device)
    model.load_state_dict(saved['model'], strict=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(path('AAAmodel/bert-base-uncased'), local_files_only=True)
    return model, train, saved, tok, digest


def _load_pkl(file):
    with Path(file).open('rb') as f:
        data = _Unpickler(f).load()
    if np.asarray(data['text_bert']).ndim == 2:
        data = {k: (np.expand_dims(v, 0) if not isinstance(v, str) else np.array([v])) for k, v in data.items()}
    return data


def appendix_batch(file, train_cfg, device):
    fields = _load_pkl(file)
    ids = np.asarray(fields['text_bert'])[0, 0].astype(np.int64)
    attn = np.asarray(fields['text_bert'])[0, 1].astype(bool)
    xa = np.asarray(fields['audio'], dtype=np.float64)
    xv = np.asarray(fields['vision'], dtype=np.float64)
    if xa.ndim == 2:
        xa, xv = xa[None], xv[None]
    la = int(np.asarray(fields.get('audio_lengths', [xa.shape[1]]))[0])
    lv = int(np.asarray(fields.get('vision_lengths', [xv.shape[1]]))[0])
    la, lv = max(1, min(la, 500)), max(1, min(lv, 500))

    p_t = attn & ~np.isin(ids, [0, 101, 102])
    o_t = p_t.copy()
    p_a = np.zeros(500, dtype=bool); p_a[:la] = True
    p_v = np.zeros(500, dtype=bool); p_v[:lv] = True
    o_a = p_a & np.any(xa[0] != 0, axis=-1)
    o_v = p_v & np.any(xv[0] != 0, axis=-1)

    scaler = Path(train_cfg['data_dir']) / 'scaler_params.npz'
    with np.load(scaler) as z:
        xa = (xa - z['mu_A']) / z['sigma_A']
        xv = (xv - z['mu_V']) / z['sigma_V']
    xa[0, ~o_a] = 0
    xv[0, ~o_v] = 0

    batch = {
        'I': torch.from_numpy(ids[None].astype(np.int64)),
        'XA': torch.from_numpy(xa.astype(np.float32)),
        'XV': torch.from_numpy(xv.astype(np.float32)),
        'P_T': torch.from_numpy(p_t[None]), 'O_T': torch.from_numpy(o_t[None]),
        'P_A': torch.from_numpy(p_a[None]), 'O_A': torch.from_numpy(o_a[None]),
        'P_V': torch.from_numpy(p_v[None]), 'O_V': torch.from_numpy(o_v[None]),
    }
    batch = {k: v.to(device) for k, v in batch.items()}
    text = str(np.asarray(fields['raw_text']).reshape(-1)[0])
    tok = AutoTokenizer.from_pretrained(path('AAAmodel/bert-base-uncased'), local_files_only=True)
    tokens = tok.convert_ids_to_tokens(ids.tolist())
    sid = str(np.asarray(fields.get('id', [Path(file).stem])).reshape(-1)[0])
    return batch, text, tokens, sid


@torch.no_grad()
def predict(model, batch, return_attention=False):
    out = model(batch, return_attention=return_attention)
    p = out['logits'].softmax(-1)[0]
    c = int(p.argmax())
    raw = float(out['intensity'][0])
    y = float(project_intensity(np.array([c]), np.array([raw]))[0])
    result = {'p': p.cpu().numpy(), 'c': c, 'y': y, 'raw': raw}
    if return_attention:
        for key in ('attention_audio', 'attention_vision'):
            if key in out:
                result[key] = out[key][0].mean(0).detach().cpu().numpy()  # 50×500
    return result


def masked(batch, modality=None, spans=None):
    """modality: 'T'/'A'/'V' 整模态删；或 spans={(m_letter): [(lo,hi),...]}。"""
    b = {k: v.clone() for k, v in batch.items()}
    if modality == 'T':
        b['O_T'].zero_()
    elif modality == 'A':
        b['O_A'].zero_()
    elif modality == 'V':
        b['O_V'].zero_()
    if spans:
        for letter, segs in spans.items():
            key = f'O_{letter}'
            for lo, hi in segs:
                b[key][:, lo:hi] = False
    return b
