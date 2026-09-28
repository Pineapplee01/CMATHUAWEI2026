"""问题二提交输出：按已预测极性投影强度，原回归头输出另存审计。"""
import numpy as np


def project_intensity(classes, raw_intensity, epsilon=1e-6):
    classes = np.asarray(classes)
    raw = np.asarray(raw_intensity, dtype=float)
    if classes.shape != raw.shape or not np.isin(classes, [0,1,2]).all():
        raise ValueError('极性类别或形状非法')
    if not np.isfinite(raw).all() or not 0 < epsilon < 3:
        raise ValueError('非有限强度或非法数值边界')
    # 在固定类别可行域上求距原预测最近的强度；epsilon仅实现开区间，不是调参阈值。
    return np.where(classes == 0, np.clip(raw, -3, -epsilon),
                    np.where(classes == 1, 0., np.clip(raw, epsilon, 3)))

