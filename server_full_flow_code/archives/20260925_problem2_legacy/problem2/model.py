"""多模态条件补全与加性情感预测。

推导见《模型方法与公式推导》2-1～2-17：先编码实际观测M，
再用固定观测memory补全；方差门控只作用于缺失槽；文本先验单独入账。
"""
import torch
from torch import nn
from torch.nn import functional as F
from problem2.text_encoder import TextEncoder, PrecomputedTextEncoder
from problem2.sentiment_encoder import SentimentEncoder
from problem2.completion import CompletionLayer
from problem2.temporal import TemporalReadout
from problem2.heads import AdditiveHead

DIMS = (768, 74, 35)

class EmotionModel(nn.Module):
    def __init__(self, cfg, stats):
        super().__init__()
        self.cfg = cfg
        d = cfg['hidden']
        if d % cfg['heads']:
            raise ValueError('hidden 必须被 heads 整除')
        if cfg.get('encoder_type') == 'sentiment_roberta':
            self.text = SentimentEncoder(cfg['bert_path'], cfg['input_tokenizer_path'])
        elif cfg.get('encoder_type') == 'precomputed':
            self.text = PrecomputedTextEncoder()
        else:
            self.text = TextEncoder(cfg['bert_path'])
        self.text.cls_mix = cfg.get('cls_mix', 0.0)
        self.text.backbone_dropout = cfg.get('backbone_dropout', False)
        if cfg.get('lora_rank', 0):
            self.text.enable_adapters(cfg['lora_rank'], cfg['lora_alpha'], cfg['lora_layers'], cfg.get('lora_ffn', False))
        if cfg.get("train_text_classifier", False):
            self.text.enable_classifier_tuning()
        if cfg.get("bias_tuning", False):
            self.text.enable_bias_tuning()
        if cfg.get("finetune_layers", 0):
            self.text.enable_last_layers(cfg["finetune_layers"])
        if cfg.get("finetune_embeddings", False):
            self.text.backbone.embeddings.requires_grad_(True)
        text_dim = cfg.get('text_dim', DIMS[0])
        self.project = nn.ModuleList([nn.Linear(width, d) for width in (text_dim, *DIMS[1:])])
        self.position = nn.Parameter(torch.randn(50, d) * 0.02)
        self.modality = nn.Parameter(torch.randn(3, d) * 0.02)
        self.query = nn.Parameter(torch.randn(3, d) * 0.02)
        self.null = nn.Parameter(torch.zeros(1, 1, d))
        self.layers = nn.ModuleList([CompletionLayer(d, cfg['heads'], cfg['dropout'])
                                     for _ in range(cfg['layers'])])
        widths = (cfg.get('text_dim', DIMS[0]), *DIMS[1:])
        self.mean = nn.ModuleList([nn.Linear(d, width) for width in widths])
        self.logvar = nn.ModuleList([nn.Linear(d, width) for width in widths])
        self.temporal = TemporalReadout(d) if cfg.get('temporal_readout', False) else None
        self.head = AdditiveHead(d, cfg['window'], cfg.get('normalize_windows', False), cfg.get('triple_rank', 0))
        self.feature_dropout = nn.Dropout(cfg.get('feature_dropout', 0.0))
        self.register_buffer('tau', torch.tensor(cfg['tau'], dtype=torch.float32))
        if (self.tau <= 0).any():
            raise ValueError('tau 必须为正')
        for name in ('audio', 'vision'):
            self.register_buffer(name + '_mean', torch.tensor(stats[name]['mean'], dtype=torch.float32))
            self.register_buffer(name + '_std', torch.tensor(stats[name]['std'], dtype=torch.float32))

    def encode(self, batch, M):
        if self.cfg.get('encoder_type') == 'precomputed':
            values = [self.text(batch['text768'], M[:, 0])]
        else:
            values = [self.text(batch['tokens'], M[:, 0])]
        for m, name in enumerate(('audio', 'vision'), 1):
            x = (batch[name] - getattr(self, name + '_mean')) / getattr(self, name + '_std')
            if self.cfg.get('standardized_clip'):
                x = x.clamp(-self.cfg['standardized_clip'], self.cfg['standardized_clip'])
            values.append(x * M[:, m, :, None])
        return values

    def forward(self, batch, B=None):
        P, O = batch['P'], batch['O']
        M = P & O if B is None else P & O & B
        values = self.encode(batch, M)
        n = P.shape[0]
        offset = self.position[None, None] + self.modality[None, :, None]
        embeddings = self.feature_dropout(torch.stack([F.gelu(layer(x)) for layer, x in zip(self.project, values)], 1))
        memory = (embeddings + offset).reshape(n, 150, -1)
        # null 仅无证据样本开放，其他样本只能读真实观测；预测不作新观测回灌。
        has_observation = M.flatten(1).any(1)
        memory = torch.cat([memory, self.null.expand(n, -1, -1)], 1)
        padding = torch.cat([~M.flatten(1), has_observation[:, None]], 1)
        queries = (offset + self.query[None, :, None]).expand(n, -1, -1, -1)
        if self.cfg.get('contextual_fusion', False):
            queries = queries + embeddings * M[..., None]
        queries = queries.reshape(n, 150, -1)
        for layer in self.layers:
            queries = layer(queries, memory, padding)
        queries = queries.reshape(n, 3, 50, -1)
        mu = [layer(queries[:, m]) for m, layer in enumerate(self.mean)]
        logvar = [layer(queries[:, m]).clamp(-6, 4) for m, layer in enumerate(self.logvar)]
        reliability = torch.stack([torch.exp(-lv.exp().mean(-1) / self.tau[m])
                                   for m, lv in enumerate(logvar)], 1)
        if self.cfg['ablation'] == 'no_reliability':
            reliability = torch.ones_like(reliability)
        elif self.cfg['ablation'] == 'no_completion':
            reliability = torch.zeros_like(reliability)
        elif self.cfg['ablation'] != 'main':
            raise ValueError('未知消融配置')
        q = torch.where(M, torch.ones_like(reliability), reliability) * P
        completed = torch.stack([F.gelu(layer(x)) for layer, x in zip(self.project, mu)], 1)
        observed_h = embeddings
        if self.cfg.get('contextual_fusion', False):
            observed_h = F.layer_norm(embeddings + self.cfg.get('fusion_strength', .5)*queries,
                                     (embeddings.shape[-1],))
        h = observed_h * M[..., None] + completed * (P & ~M)[..., None] * q[..., None]
        if self.temporal is not None:
            h = self.temporal(h, P)
        pooled = self.head.pool(h, P, M, q)
        scores, ledger = self.head(pooled)
        if self.cfg.get('sentiment_prior', False):
            count = M[:,0].sum(1,keepdim=True).clamp_min(1)
            prior = values[0][:,:,-3:].sum(1) / count
            prior = self.cfg.get('sentiment_prior_scale', 1.) * prior
            prior_score = torch.cat([prior, torch.zeros_like(prior[:,:1])], -1)
            scores = scores + prior_score
            ledger['text_prior'] = prior_score
        return {'logits': scores[:, :3], 'score': scores[:, 3],
                'intensity': 3 * scores[:, 3].tanh(), 'mu': mu, 'logvar': logvar,
                'q': q, 'M': M, 'P': P, 'values': values, 'pooled': pooled,
                'ledger': ledger, 'low_evidence': ~has_observation}
