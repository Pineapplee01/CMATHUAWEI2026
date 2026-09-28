"""一次训练视图的联合目标；由普通优化和SAM二次前向共用。"""
from problem2.losses import objective, mixup_regularizer, supervised_contrastive, task_loss, dropout_consistency


def batch_objective(model, batch, B, epoch_cfg, weights):
    cfg=epoch_cfg
    full, missing = model(batch), model(batch, B)
    loss, log = objective(full, missing, batch, epoch_cfg, weights)
    loss = loss + mixup_regularizer(model.head, full['pooled'], batch,
                 cfg.get('mixup_strength', 0), cfg.get('mixup_alpha', .4))
    if cfg.get('supervised_contrastive_weight', 0):
        loss = loss + cfg['supervised_contrastive_weight'] * supervised_contrastive(
            full['pooled'], missing['pooled'], batch['c'])
    if cfg.get('rdrop_weight', 0):
        repeated = model(batch)
        # 原完整CE与第二次CE取均值，缺失与重建项保留原权重。
        loss = loss + .5 * (task_loss(repeated, batch, epoch_cfg, weights) -
                            task_loss(full, batch, epoch_cfg, weights))
        loss = loss + cfg['rdrop_weight'] * dropout_consistency(full['logits'], repeated['logits'])
    return loss, log
