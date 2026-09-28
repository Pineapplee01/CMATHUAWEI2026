"""人工边界标注的误差核验。"""
import numpy as np
import pandas as pd
from shared.common import json_write



def boundary_validation(filename, output):
    """人工参考不由程序伪造。CSV 每行包含方法、估计/人工起止秒数。"""
    frame = pd.read_csv(filename)
    required = {'id','method','estimated_start','estimated_end','reference_start','reference_end'}
    if not required.issubset(frame.columns): raise ValueError(f'需要列 {sorted(required)}')
    frame = frame[frame.method=='mfa'].copy()
    if frame.empty: raise ValueError('没有 MFA 人工核验记录；fallback 不能当强制对齐成功')
    errors = np.r_[abs(frame.estimated_start-frame.reference_start),
                   abs(frame.estimated_end-frame.reference_end)]
    if not np.isfinite(errors).all(): raise ValueError('人工核验包含缺失/非有限值')
    json_write(output, {'count_words': len(frame), 'mean_boundary_absolute_error_sec': float(errors.mean()),
                       'within_50ms': float((errors<=.05).mean()),
                       'within_100ms': float((errors<=.1).mean()), 'within_200ms': float((errors<=.2).mean())})

