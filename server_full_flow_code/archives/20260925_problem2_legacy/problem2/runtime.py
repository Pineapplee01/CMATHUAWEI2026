"""数据加载与检查点保存/恢复。"""
import torch
from torch.utils.data import DataLoader
from shared.common import path, fingerprint, environment
from shared.data import FeatureDataset, load_fields
from problem2.model import EmotionModel



def dataset(cfg, split):
    filename = path(cfg['data_dir']) / f'{split}.pkl'
    return FeatureDataset(load_fields(filename, split), filename, cfg['mask_rule'], labels=True)



def loader(ds, cfg, shuffle=False):
    return DataLoader(ds, batch_size=cfg['batch_size'], shuffle=shuffle, num_workers=0)



def save_checkpoint(model, cfg, stats, identity, tokenizer_check, epoch, score):
    filename = path(cfg['checkpoint'])
    filename.parent.mkdir(parents=True, exist_ok=True)
    trainable = {n for n,p in model.named_parameters() if p.requires_grad}
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()
             if not k.startswith('text.backbone.') or k in trainable}
    temporary = filename.with_suffix('.tmp')
    torch.save({'model': state, 'config': cfg, 'stats': stats, 'encoder_sha256': identity,
                'tokenizer_check': tokenizer_check, 'epoch': epoch, 'selection_score': score,
                'environment': environment()}, temporary)
    temporary.replace(filename)



def restore(filename, device):
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    checkpoint = torch.load(path(filename), map_location='cpu', weights_only=False)
    cfg = dict(checkpoint['config']); cfg['device'] = device
    if fingerprint(cfg['bert_path']) != checkpoint['encoder_sha256']:
        raise ValueError('本地 BERT/词表与训练检查点不一致')
    if cfg.get('input_tokenizer_path'):
        if fingerprint(cfg['input_tokenizer_path']) != checkpoint['tokenizer_check'].get('input_tokenizer_sha256'):
            raise ValueError('输入BERT词表资源身份改变')
    model = EmotionModel(cfg, checkpoint['stats']).to(device)
    # v1检查点没有后续添加的决策阈值偏置；显式迁移为0，保持原预测。
    if 'head.decision_bias' not in checkpoint['model']:
        checkpoint['model']['head.decision_bias'] = torch.zeros(4)
    missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
    trainable = {n for n,p in model.named_parameters() if p.requires_grad}
    if unexpected or any(not k.startswith('text.backbone.') or k in trainable for k in missing):
        raise ValueError(f'检查点缺失/多余参数: {missing}, {unexpected}')
    model.eval()
    return model, cfg
