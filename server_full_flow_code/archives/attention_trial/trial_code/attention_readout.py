"""质量感知的窗口注意力；标记统计继续按有效位置平均。"""
import torch
from torch import nn


class AttentionPool(nn.Module):
    def __init__(self,hidden,window):
        super().__init__();self.window=window
        self.score=nn.ModuleList([nn.Sequential(nn.Linear(hidden,16),nn.Tanh(),nn.Linear(16,1)) for _ in range(3)])
        for layer in self.score:nn.init.zeros_(layer[-1].weight);nn.init.zeros_(layer[-1].bias)

    def forward(self,h,P,M,q):
        logits=torch.stack([net(h[:,m]).squeeze(-1) for m,net in enumerate(self.score)],1)
        pooled=[]
        flags=torch.stack([M.float(),(P&~M).float(),q],-1)
        for lo in range(0,50,self.window):
            hi=min(50,lo+self.window);valid=P[:,:,lo:hi]
            # padding无权重；补全项依可靠性衰减；全空窗避免全-inf softmax。
            scores=logits[:,:,lo:hi].masked_fill(~valid,-1e4)
            weight=scores.softmax(-1)*valid*q[:,:,lo:hi]
            weight=weight/weight.sum(-1,keepdim=True).clamp_min(1e-12)
            content=(weight[...,None]*h[:,:,lo:hi]).sum(2)
            status=(flags[:,:,lo:hi]*valid[...,None]).sum(2)/valid.sum(2,keepdim=True).clamp_min(1)
            pooled.append(torch.cat([content,status],-1))
        return torch.stack(pooled,2)
