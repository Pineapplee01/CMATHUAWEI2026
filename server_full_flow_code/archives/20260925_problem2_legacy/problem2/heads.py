"""固定窗口加性预测及可重建贡献账本（推导2-13～2-16）。

窗口以P平均，附加观测/缺失/可靠性三维；单项与交互项按训练背景
中心化并按有效窗口归一化。四维账本在logit/pre-tanh尺度可加。
"""
import itertools
import torch
from torch import nn

PAIRS = tuple(itertools.combinations(range(3), 2))

class AdditiveHead(nn.Module):
    def __init__(self, hidden, window, normalize=False, triple_rank=0):
        super().__init__()
        if not 1 <= window <= 50:
            raise ValueError('window 必须为 1..50')
        self.window = window
        self.normalize = normalize
        self.windows = (50 + window - 1) // window
        self.bias = nn.Parameter(torch.zeros(4))
        self.register_buffer('decision_bias', torch.zeros(4))
        self.single = nn.ModuleList([nn.Sequential(nn.Linear(hidden + 3, hidden),
                                     nn.Tanh(), nn.Linear(hidden, 4)) for _ in range(3)])
        self.pair = nn.ModuleList([nn.Sequential(nn.Linear(2 * (hidden + 3), hidden),
                                   nn.Tanh(), nn.Linear(hidden, 4)) for _ in PAIRS])
        self.triple_rank = triple_rank
        if triple_rank:
            self.triple_factors = nn.ModuleList([nn.Linear(hidden+3,triple_rank) for _ in range(3)])
            self.triple_out = nn.Linear(triple_rank,4,bias=False)
            nn.init.zeros_(self.triple_out.weight)
        self.register_buffer('background', torch.zeros(3, self.windows, hidden + 3))

    def pool(self, h, P, M, q):
        features = torch.cat([h, M[..., None].float(),
                              (P & ~M)[..., None].float(), q[..., None]], -1)
        pooled = []
        for lo in range(0, 50, self.window):
            hi = min(50, lo + self.window)
            mask = P[:, :, lo:hi, None]
            pooled.append((features[:, :, lo:hi] * mask).sum(2) /
                          mask.sum(2).clamp_min(1))
        return torch.stack(pooled, 2)

    def forward(self, pooled):
        singles = torch.stack([head(pooled[:, m]) - head(self.background[m])
                               for m, head in enumerate(self.single)], 1)
        pairs = torch.stack([head(torch.cat([pooled[:, a], pooled[:, b]], -1)) -
                             head(torch.cat([self.background[a], self.background[b]], -1))
                             for head, (a, b) in zip(self.pair, PAIRS)], 1)
        if self.normalize:
            # P=0窗口不参与预测；只按真正有效窗口归一化，避免长度捷径。
            active = (pooled[..., -3] + pooled[..., -2] > 0).float()
            singles = singles * (active / active.sum(-1, keepdim=True).clamp_min(1))[..., None]
            pair_active = torch.stack([active[:, a]*active[:, b] for a,b in PAIRS],1)
            pairs = pairs * (pair_active / pair_active.sum(-1,keepdim=True).clamp_min(1))[..., None]
        triple = None
        if self.triple_rank:
            factors = [torch.tanh(layer(pooled[:,m])) for m,layer in enumerate(self.triple_factors)]
            base = [torch.tanh(layer(self.background[m])) for m,layer in enumerate(self.triple_factors)]
            triple = self.triple_out(factors[0]*factors[1]*factors[2] - base[0]*base[1]*base[2])
            if self.normalize:
                active3 = active.prod(1)
                triple = triple * (active3/active3.sum(-1,keepdim=True).clamp_min(1))[...,None]
        bias = self.bias + self.decision_bias
        score = bias + singles.sum((1, 2)) + pairs.sum((1, 2))
        ledger = {'bias':bias, 'single':singles, 'pair':pairs}
        if triple is not None:
            score = score + triple.sum(1)
            ledger['triple'] = triple
        return score, ledger
