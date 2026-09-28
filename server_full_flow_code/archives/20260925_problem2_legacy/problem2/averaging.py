"""训练参数指数平均：降低小样本训练波动，推理不增加参数量。"""
from contextlib import contextmanager
import torch


class ExponentialAverage:
    def __init__(self, model, decay):
        if not 0 < decay < 1: raise ValueError('EMA衰减必须位于(0,1)')
        self.decay=decay;self.steps=0
        self.shadow={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        self.steps+=1
        decay=min(self.decay,(1+self.steps)/(10+self.steps))
        for name,p in model.named_parameters():
            if name in self.shadow:self.shadow[name].lerp_(p.detach(),1-decay)

    @contextmanager
    def average_parameters(self, model):
        original={}
        with torch.no_grad():
            for name,p in model.named_parameters():
                if name in self.shadow:
                    original[name]=p.detach().clone();p.copy_(self.shadow[name])
        if model.text.adapters_active:model.text.cache.clear()
        try: yield
        finally:
            with torch.no_grad():
                for name,p in model.named_parameters():
                    if name in original:p.copy_(original[name])
            if model.text.adapters_active:model.text.cache.clear()
