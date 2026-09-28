"""冻结BERT：词元遮挡后重新编码。"""
import torch
from collections import OrderedDict
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel
from shared.common import path
from problem2.adapters import attach_adapters



class PrecomputedTextEncoder(nn.Module):
    """消费预处理 npz 固化的冻结BERT逐词元嵌入（50×768，train有效位z-score）。"""

    def __init__(self):
        super().__init__()

    def forward(self, text768, observed):
        return text768 * observed.unsqueeze(-1)

    def train(self, mode=True):
        return self


class TextEncoder(nn.Module):
    def __init__(self, directory):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(path(directory), local_files_only=True)
        if self.backbone.config.hidden_size != 768:
            raise ValueError('文本骨干必须输出 768 维')
        self.backbone.requires_grad_(False)
        self.cache = OrderedDict()
        self.cache_limit = 16000
        self.adapters_active = False
        self.cls_mix = 0.0
        self.backbone_dropout = False

    def enable_adapters(self, rank, alpha, layers, ffn=False):
        attach_adapters(self.backbone, rank, alpha, layers, ffn)
        self.adapters_active = True
        self.cache.clear()

    def enable_last_layers(self, count):
        layers = self.backbone.encoder.layer
        if not 1 <= count <= len(layers):
            raise ValueError("微调层数越界")
        for layer in layers[-count:]:
            layer.requires_grad_(True)
        self.adapters_active = True
        self.cache.clear()

    def enable_bias_tuning(self):
        for name, parameter in self.backbone.named_parameters():
            if name.endswith("bias"):
                parameter.requires_grad_(True)
        self.adapters_active = True
        self.cache.clear()

    def train(self, mode=True):
        if mode and self.adapters_active:
            self.cache.clear()
        super().train(mode and self.adapters_active and self.backbone_dropout)
        return self

    def forward(self, tokens, observed):
        if self.adapters_active and torch.is_grad_enabled():
            # 训练适配参数时禁用最终表示缓存，避免陈旧表征；eval/no_grad才复用。
            return self.encode_uncached(tokens, observed)
        # 缓存键含全部词元和当前掩码，缺失视图不能误用完整句表示。
        host_tokens, host_mask = tokens.detach().cpu().numpy(), observed.detach().cpu().numpy()
        keys = [(t.tobytes(), m.tobytes()) for t, m in zip(host_tokens, host_mask)]
        missing = [i for i, key in enumerate(keys) if key not in self.cache]
        fresh = {}
        if missing:
            values = self.encode_uncached(tokens[missing], observed[missing]).cpu()
            for i, value in zip(missing, values):
                fresh[keys[i]] = value.clone()
                self.cache[keys[i]] = fresh[keys[i]]
        result = torch.stack([fresh[key] if key in fresh else self.cache[key] for key in keys]).to(tokens.device)
        for key in keys:
            self.cache.move_to_end(key)
        while len(self.cache) > self.cache_limit:
            self.cache.popitem(last=False)
        return result

    def encode_uncached(self, tokens, observed):
        ids = tokens[:, 0].clone()
        # 所有不可观测内容在编码前去除；只保留结构性 CLS/SEP。
        structural = (ids == 101) | (ids == 102)
        attention = observed | structural
        ids[~attention] = 0
        empty = ~attention.any(1)
        ids[empty, 0] = 101
        attention[empty, 0] = True  # 仅防止全掩码注意力，输出仍被 observed 清零
        segments = tokens[:, 2].clone()
        segments[~attention] = 0
        result = self.backbone(input_ids=ids, attention_mask=attention.long(),
                               token_type_ids=segments).last_hidden_state
        if self.cls_mix:
            # CLS是保留词元的句级上下文，不是额外观测；干预后必须重新计算。
            result = (1-self.cls_mix)*result + self.cls_mix*result[:, :1]
        # 文本逐位置 LayerNorm 无跨样本拟合，不把全句预计算 text 混入输入。
        result = F.layer_norm(result, (768,))
        return result * observed.unsqueeze(-1)
