"""跨问题接口：旧检查点显式拒绝加载，避免问题三继续使用不合规模型。"""
from problem2.training import restore
from torch.utils.data import DataLoader


def loader(ds,cfg,shuffle=False):
    return DataLoader(ds,batch_size=cfg['batch_size'],shuffle=shuffle,num_workers=0)


def dataset(cfg,split):
    raise RuntimeError('问题三旧逐样本接口尚不适用于aligned_v4；请先迁移解释协议，勿复用旧解释结果。')
