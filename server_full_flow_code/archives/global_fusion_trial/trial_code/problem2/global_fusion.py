"""句级模态融合：全局投影、残差前馈或三token注意力、单模态辅助监督。"""
import torch
from torch import nn
from torch.nn import functional as F


class GlobalFusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg['hidden']
        dropout = cfg.get('global_dropout', .3)
        # 音视频额外保留帧间变化；文本本来是句级向量，不重复标准差。
        self.project = nn.ModuleList([
            nn.Sequential(nn.Linear(width, d), nn.LayerNorm(d), nn.GELU())
            for width in (768, 148, 70)])
        self.attention = None
        if cfg['global_fusion'] == 'attention':
            self.attention = nn.TransformerEncoderLayer(
                d, cfg['heads'], dim_feedforward=2*d, dropout=dropout,
                batch_first=True, norm_first=True)
            self.modality = nn.Parameter(torch.randn(1, 3, d) * .02)
        elif cfg['global_fusion'] != 'concat':
            raise ValueError('global_fusion必须为concat或attention')
        width = 3*d
        self.norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(width, width))
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(width, 4)
        # 热启动与已有最佳模型完全一致，再学习新的全局残差。
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.aux = nn.ModuleList([nn.Linear(d, 3) for _ in range(3)]) if cfg.get('unimodal_aux', 0) else None

    @staticmethod
    def summarize(value, mask, dispersion):
        weight = mask[..., None].to(value)
        # 在计算统计量前去除隐藏位置，禁止隐藏值通过方差泄漏。
        clean = torch.where(mask[..., None], value, torch.zeros_like(value))
        count = weight.sum(1).clamp_min(1)
        mean = clean.sum(1) / count
        if not dispersion:
            return mean
        variance = ((clean-mean[:, None]).square()*weight).sum(1)/count
        std = (variance + 1e-6).sqrt() * mask.any(1, keepdim=True)
        return torch.cat([mean, std], -1)

    def forward(self, values, observed):
        available = observed.any(-1)
        tokens = torch.stack([
            layer(self.summarize(value, observed[:, m], m > 0))
            for m, (layer, value) in enumerate(zip(self.project, values))], 1)
        tokens = tokens * available[..., None]
        auxiliary = torch.stack([head(tokens[:, m]) for m, head in enumerate(self.aux)], 1) if self.aux is not None else None
        if self.attention is not None:
            padding = ~available.clone()
            # 全缺失时临时开放空token防NaN，注意力后再次清零。
            padding[~available.any(1), 0] = False
            tokens = self.attention(tokens + self.modality, src_key_padding_mask=padding)
            tokens = tokens * available[..., None]
        flat = tokens.flatten(1)
        fused = self.norm(flat + self.ffn(self.norm(flat)))
        score = self.out(self.dropout(fused)) * available.any(1, keepdim=True)
        return score, auxiliary, available


def auxiliary_loss(output, labels):
    logits = output['aux_logits']
    valid = output['aux_available']
    if logits is None or not valid.any():
        return output['logits'].sum() * 0
    targets = labels[:, None].expand(-1, 3)
    return F.cross_entropy(logits[valid], targets[valid])
