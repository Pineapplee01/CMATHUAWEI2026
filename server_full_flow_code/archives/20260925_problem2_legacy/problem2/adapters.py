"""低秩任务适配：冻结既有线性层，仅学习 BA 残差，不增加骨干层数/宽度。
参考：Hu et al., LoRA, https://arxiv.org/abs/2106.09685
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class LowRankLinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank,base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features,rank))
        nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + self.scale*F.linear(F.linear(x,self.lora_A),self.lora_B)


def attach_adapters(backbone, rank, alpha, layers, ffn=False):
    if not 1 <= layers <= len(backbone.encoder.layer):
        raise ValueError('适配层数超出BERT编码层范围')
    for layer in backbone.encoder.layer[-layers:]:
        attention = layer.attention.self
        attention.query = LowRankLinear(attention.query,rank,alpha)
        attention.value = LowRankLinear(attention.value,rank,alpha)

        if ffn:
            layer.intermediate.dense = LowRankLinear(layer.intermediate.dense, rank, alpha)
            layer.output.dense = LowRankLinear(layer.output.dense, rank, alpha)
