"""附件3全量局部缺失预测、覆盖验证及模型来源记录。"""
import hashlib
import numpy as np
import pandas as pd
import torch
from shared.common import path, to_device, LABELS, json_write
from shared.data import FeatureDataset, load_fields
from problem2.runtime import loader
from problem2.output_contract import project_intensity



@torch.no_grad()
def predict_appendix3(model, cfg):
    model.eval()
    files = sorted(path(cfg['appendix3']).glob('*.pkl'))
    if len(files) != 30:
        raise ValueError(f'附件3应有30个文件，实际 {len(files)}')
    rows, mask_rows = [], []
    for file in files:
        fields = load_fields(file)
        if any(k in fields for k in ('regression_labels', 'classification_labels', 'annotations')):
            raise ValueError('专项推理只接受无标签输入')
        ds = FeatureDataset(fields, file, cfg['mask_rule'])
        if len(ds) != 1:
            raise ValueError(f'{file}: 专项文件应包含一个样本')
        for raw in loader(ds, cfg):
            out = model(to_device(raw, cfg['device']))
            for i in range(len(raw['id'])):
                c, raw_y = int(out['logits'][i].argmax()), float(out['intensity'][i])
                y = float(project_intensity(c, raw_y)) if cfg.get('output_rule') == 'class_constrained' else raw_y
                rows.append({'sample_id': file.stem, 'source_file': file.name, 'class': c,
                             'polarity': LABELS[c], 'intensity': y, 'raw_intensity':raw_y,
                             'head_consistent': c == np.sign(y) + 1,
                             'low_evidence': bool(out['low_evidence'][i]),
                             'validity_uncertain': True, 'mask_rule': cfg['mask_rule']})
                for m, name in enumerate(('text','audio','vision')):
                    missing = (raw['P'][i,m] & ~raw['O'][i,m]).numpy()
                    starts = np.flatnonzero(missing & ~np.r_[False,missing[:-1]])
                    ends = np.flatnonzero(missing & ~np.r_[missing[1:],False])+1
                    mask_rows.append({'sample_id':file.stem,'modality':name,
                                      'effective_slots':int(raw['P'][i,m].sum()),
                                      'observed_slots':int(raw['O'][i,m].sum()),
                                      'zero_intervals':';'.join(f'{a}:{b}' for a,b in zip(starts,ends)),
                                      'extent_inferred':True})
    output = path(cfg['output']); output.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows)
    if detail.sample_id.duplicated().any() or set(detail.source_file) != {f.name for f in files}:
        raise AssertionError('附件3样本覆盖错误')
    if not np.isfinite(detail.intensity).all() or detail.intensity.abs().gt(3).any():
        raise AssertionError('输出强度非法')
    if cfg.get('output_rule') == 'class_constrained' and not detail.head_consistent.all():
        raise AssertionError('极性与强度定义不一致')
    detail.to_csv(output/'附件3_推理审计.csv',index=False,encoding='utf-8-sig')
    detail[['sample_id','source_file','polarity','intensity']].to_csv(
        output/'附件3_预测结果.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(mask_rows).to_csv(output/'附件3_缺失区间.csv',index=False,encoding='utf-8-sig')
    json_write(output/'附件3_提交核验.json', {
        'samples':len(detail),'complete_coverage':True,'labels_available':False,
        'feature_version':'aligned_50','mask_rule':cfg['mask_rule'],
        'padding_missing_ambiguity':'zero-valued trailing positions cannot always be distinguished from padding',
        'checkpoint':cfg['checkpoint'],
        'checkpoint_sha256':hashlib.sha256(path(cfg['checkpoint']).read_bytes()).hexdigest(),
        'output_rule':cfg.get('output_rule','independent_heads'),
        'raw_head_disagreements':int((detail['class'] != np.sign(detail.raw_intensity)+1).sum()),
        'final_head_disagreements':int((~detail.head_consistent).sum()),
        'class_counts':detail.polarity.value_counts().to_dict(),
        'metrics_not_reported_reason':'unlabeled special test set',
        'duration_unit':'aligned_slot, not seconds'})
