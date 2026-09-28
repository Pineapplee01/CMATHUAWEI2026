"""音视频时间变化读出：小型双向GRU，排除填充位置。"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence


class TemporalReadout(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        if hidden%2:raise ValueError('双向GRU需要偶数hidden')
        self.encoders=nn.ModuleList([nn.GRU(hidden,hidden//2,batch_first=True,bidirectional=True) for _ in range(2)])

    def forward(self, h, valid):
        values=[h[:,0]]
        for m,encoder in enumerate(self.encoders,1):
            lengths=torch.where(valid[:,m],torch.arange(h.shape[2],device=h.device)+1,0).max(-1).values
            packed=pack_padded_sequence(h[:,m]*valid[:,m,:,None],lengths.clamp_min(1).cpu(),
                                       batch_first=True,enforce_sorted=False)
            result,_=encoder(packed)
            result,_=pad_packed_sequence(result,batch_first=True,total_length=h.shape[2])
            values.append(F.layer_norm(h[:,m]+result,(h.shape[-1],))*valid[:,m,:,None])
        return torch.stack(values,1)
