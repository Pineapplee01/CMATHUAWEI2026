"""情感监督、重建和一致性损失（推导2-10、2-18、2-19）。

高斯NLL只监督人工隐藏且原来有真值的H；完整文本目标detach。
最终配置lambda_con=0、mixup_strength=0，保留函数不代表启用。
"""
import torch
from torch.nn import functional as F



def objective(full, missing, batch, cfg, class_weight):
    def task(output):
        return task_loss(output, batch, cfg, class_weight)
    H = batch['P'] & batch['O'] & ~missing['M']
    terms = []
    for m in range(3):
        if H[:, m].any():
            err = (full['values'][m].detach() - missing['mu'][m]).square()
            lv = missing['logvar'][m]
            terms.append((0.5 * (err * (-lv).exp() + lv))[H[:, m]].mean())
    rec = torch.stack(terms).mean() if terms else full['logits'].sum() * 0
    if cfg['ablation'] == 'no_completion':
        rec = rec * 0
    con = F.kl_div(missing['logits'].log_softmax(-1),
                   full['logits'].detach().softmax(-1), reduction='batchmean') + \
          F.mse_loss(missing['intensity'], full['intensity'].detach())
    loss = task(full) + cfg['lambda_miss'] * task(missing) + cfg['lambda_rec'] * rec + cfg['lambda_con'] * con
    return loss, {'loss': float(loss.detach()), 'reconstruction_nll': float(rec.detach())}


def mixup_regularizer(head, pooled, batch, strength, alpha):
    """同一比例混合三个模态的窗口表征和监督，抑制小样本记忆。"""
    if strength <= 0:
        return pooled.sum() * 0
    fraction = torch.distributions.Beta(alpha, alpha).sample().to(pooled.device)
    order = torch.randperm(len(pooled), device=pooled.device)
    mixed = fraction * pooled + (1-fraction) * pooled[order]
    score, _ = head(mixed)
    classification = fraction * F.cross_entropy(score[:, :3], batch['c']) + \
        (1-fraction) * F.cross_entropy(score[:, :3], batch['c'][order])
    target = fraction * batch['y'] + (1-fraction) * batch['y'][order]
    return strength * (classification + .1 * F.huber_loss(3*score[:, 3].tanh(), target))


def supervised_contrastive(full, missing, labels, temperature=.2):
    """完整/缺失视图共享类别，增强同类跨样本表征的一致性。"""
    features=torch.cat([full.mean(2).flatten(1),missing.mean(2).flatten(1)],0)
    features=F.normalize(features,dim=-1)
    classes=labels.repeat(2)
    logits=features@features.T/temperature
    diagonal=torch.eye(len(classes),dtype=torch.bool,device=features.device)
    logits=logits-logits.max(-1,keepdim=True).values.detach()
    log_probability=logits-torch.log((logits.exp()*~diagonal).sum(-1,keepdim=True).clamp_min(1e-12))
    positive=(classes[:,None]==classes[None,:])&~diagonal
    return -((log_probability*positive).sum(-1)/positive.sum(-1).clamp_min(1)).mean()


def soft_neutral_weight(batch, cfg):
    """弱极性样本的中性软标签权重。

    硬标签在边界自相矛盾（y=±0.17 标极性、内容相同的 y=0 标中性，
    模型在train上中性召回仅29.3%）。此处按 |y| 连续地允许目标偏向中性；
    可选乘 roberta-sentiment 的中性概率，让先验只参与训练期监督。
    """
    delta = cfg.get('soft_label_delta', 0.0)
    if delta <= 0:
        return None
    w = torch.sigmoid((delta - batch['y'].abs()) / cfg.get('soft_label_temperature', 0.15))
    if cfg.get('soft_label_use_prior') and 'prior_neu' in batch:
        w = w * batch['prior_neu'].to(w.dtype)
    w = torch.where(batch['c'] == 1, torch.zeros_like(w), w)
    return w.clamp(0, cfg.get('soft_label_max', 0.6))


def task_loss(output, batch, cfg, class_weight=None):
    """分类/强度监督；软标签分支用连续目标替代硬CE，类别权重不适用。"""
    w = soft_neutral_weight(batch, cfg)
    if w is not None:
        q = F.one_hot(batch['c'], 3).to(output['logits']) * (1 - w[:, None])
        q[:, 1] += w
        loss = -(q * output['logits'].log_softmax(-1)).sum(-1).mean()
    else:
        loss = F.cross_entropy(output['logits'], batch['c'], weight=class_weight,
                               label_smoothing=cfg.get('label_smoothing', 0.0))
    loss = loss + cfg.get('regression_weight', 1.0) * F.huber_loss(output['intensity'], batch['y'])
    if cfg.get('ordinal_weight', 0):
        distribution = output['logits'].softmax(-1).cumsum(-1)[:, :2]
        target = F.one_hot(batch['c'], 3).to(distribution).cumsum(-1)[:, :2]
        loss = loss + cfg['ordinal_weight'] * (distribution-target).square().mean()
    return loss


def dropout_consistency(first, second):
    """双向KL，两个随机分支均可接收梯度，不detach任一教师。"""
    a, b = first.log_softmax(-1), second.log_softmax(-1)
    return .5 * (F.kl_div(a, b.exp(), reduction='batchmean') +
                 F.kl_div(b, a.exp(), reduction='batchmean'))
