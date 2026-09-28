"""precomputed_main 在 test 上的完整观测与缺失30%评估。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from shared.common import config, to_device
from problem2.runtime import restore, dataset, loader
from problem2.validation import evaluate
from problem2.evaluation import metrics

cfg = config('configs/precomputed_sentiment.json')
model, saved = restore(cfg['checkpoint'], 'cuda:3')
ds = dataset(saved, 'test')
full, _ = evaluate(model, ds, saved)
miss, _ = evaluate(model, ds, saved, saved['validation_rate'], reconstruction=False)
print('== precomputed_main (标准化npz数据) test ==')
print('完整观测:', {k: round(v, 4) for k, v in metrics(full).items() if k in ('accuracy', 'macro_f1', 'mae', 'pearson')})
print('缺失30% :', {k: round(v, 4) for k, v in metrics(miss).items() if k in ('accuracy', 'macro_f1', 'mae', 'pearson')})
cm = pd.crosstab(full.true_class, full['class'], margins=True)
print(cm)
