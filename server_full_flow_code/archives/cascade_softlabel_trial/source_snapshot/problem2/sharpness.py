"""SAM参数邻域扰动：只扰动可训练参数，异常时也恢复原值。
参考 https://arxiv.org/abs/2010.01412
"""
from contextlib import contextmanager
import torch


@contextmanager
def sharpness_neighborhood(model, radius):
    parameters=[p for p in model.parameters() if p.requires_grad and p.grad is not None]
    norm=torch.stack([p.grad.detach().norm(2) for p in parameters]).norm(2)
    if not torch.isfinite(norm):
        raise FloatingPointError('SAM梯度范数非有限')
    original=[p.detach().clone() for p in parameters]
    try:
        with torch.no_grad():
            for p in parameters:
                p.add_(p.grad, alpha=radius/(float(norm)+1e-12))
        yield
    finally:
        with torch.no_grad():
            for p,value in zip(parameters,original):p.copy_(value)
