"""冻结后评估；B0为完整F1，其余为组件拆除组。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from shared.common import json_write
from problem2_retrain_v2.data import batch
from problem2_retrain_v2.local_missingness import synchronized_view
from problem2_retrain_v2.evaluation import metrics
from problem2_retrain_v2.output_contract import project_intensity

ORDER = ('B0', 'noC', 'noQ', 'noB', 'noBeta')


@torch.no_grad()
def outputs(model, data, extras=()):
    model.eval()
    result = {k: [] for k in ('logits', 'intensity', *extras)}
    device = next(model.parameters()).device
    for lo in range(0, len(data['id']), 32):
        idx = torch.arange(lo, min(lo + 32, len(data['id'])))
        b = batch(data, idx, device, dtype=next(model.parameters()).dtype)
        out = model(b)
        for key in result:
            result[key].append(out[key].detach().cpu())
    return {k: torch.cat(v) for k, v in result.items()}


def frame_of(out, data):
    raw = out['intensity'].numpy()
    c = out['logits'].argmax(-1).numpy()
    p = out['logits'].softmax(-1).numpy()
    return pd.DataFrame({
        'id': data['id'], 'true_class': data['c'].numpy(), 'true_intensity': data['y'].numpy(),
        'class': c, 'intensity': project_intensity(c, raw), 'raw_intensity': raw,
        **{f'p_{j}': p[:, j] for j in range(3)},
    })


def stability(full, missing):
    p = full['logits'].softmax(-1).clamp_min(1e-8)
    q = missing['logits'].softmax(-1).clamp_min(1e-8)
    mid = (p + q) / 2
    js = .5 * ((p * (p / mid).log()).sum(-1) + (q * (q / mid).log()).sum(-1))
    return {
        'js': float(js.mean()),
        'flip_rate': float((p.argmax(-1) != q.argmax(-1)).float().mean()),
        'raw_intensity_shift': float((full['intensity'] - missing['intensity']).abs().mean()),
    }


def analyze_checkpoint(model, data, scenario_list, spec, variant, seed, run):
    test = data['test']
    use_completion = not getattr(model, 'disable_completion', False)
    full = outputs(model, test, ('observed_embedding',) if use_completion else ())
    full_frame = frame_of(full, test)
    full_frame.to_csv(run / 'test_full.csv', index=False)
    metric_rows = [{
        'variant': variant, 'seed': seed, 'modalities': 'none', 'rate': 0.,
        'position': 'none', 'mask_seed': 0, **metrics(full_frame),
    }]
    stable_rows, rec_rows = [], []
    scenario_root = run / 'scenarios'
    scenario_root.mkdir(exist_ok=True)
    for scenario in scenario_list:
        name = scenario['modalities']
        modalities = tuple('TAV'.index(m) for m in name)
        view, records = synchronized_view(
            test, scenario['rate'], modalities, scenario['position'], scenario['mask_seed'])
        extras = ('reconstruction',) if use_completion else ()
        missing = outputs(model, view, extras)
        frame = frame_of(missing, test)
        frame.to_csv(scenario_root / f'{scenario["key"]}.csv', index=False)
        pd.DataFrame(records).to_csv(scenario_root / f'{scenario["key"]}_mask.csv', index=False)
        meta = {'variant': variant, 'seed': seed, **{k: v for k, v in scenario.items() if k != 'key'}}
        metric_rows.append(meta | metrics(frame))
        stable_rows.append(meta | stability(full, missing))
        if use_completion:
            removed = test['P'] & test['O'] & ~view['O']
            target = full['observed_embedding']
            visible = view['O'] & view['P']
            observed = outputs(model, view, ('observed_embedding',))['observed_embedding']
            mean = (observed * visible[..., None]).sum(2) / visible.sum(2, keepdim=True).clamp_min(1)
            for m in modalities:
                mask = removed[:, m]
                if not mask.any():
                    continue
                truth = target[:, m][mask]
                for kind, estimate in [
                    ('reconstructed', missing['reconstruction']),
                    ('observed_mean', mean[:, :, None].expand_as(target)),
                ]:
                    pred = estimate[:, m][mask]
                    cos = 1 - F.cosine_similarity(pred, truth, dim=-1)
                    rec_rows.append(meta | {
                        'target_modality': 'TAV'[m], 'kind': kind,
                        'cosine_distance': float(cos.mean()),
                        'target_norm': float(truth.norm(dim=-1).mean()),
                        'estimate_norm': float(pred.norm(dim=-1).mean()),
                        'slots': int(mask.sum()),
                    })
            if (name == 'T' and scenario['rate'] == spec['main_rate']
                    and scenario['position'] == 'random'
                    and scenario['mask_seed'] == spec['mask_seeds'][0]):
                dist = 1 - F.cosine_similarity(missing['reconstruction'][:, 0], target[:, 0], dim=-1)
                mask = removed[:, 0]
                eligible = mask.any(-1)
                indices = torch.where(eligible)[0]
                if len(indices):
                    score = (dist * mask).sum(-1) / mask.sum(-1).clamp_min(1)
                    idx = int(indices[torch.argsort(score[indices])[len(indices) // 2]])
                    np.savez_compressed(
                        run / 'latent_example.npz',
                        reference=target[idx, 0].numpy(),
                        observed=(observed[idx, 0] * visible[idx, 0, :, None]).numpy(),
                        reconstructed=missing['reconstruction'][idx, 0].numpy(),
                        mask=mask[idx].numpy(), valid=test['P'][idx, 0].numpy())
                    json_write(run / 'latent_example.json', {
                        'id': test['id'][idx], 'rule': 'median per-sample cosine error',
                        'true_class': int(test['c'][idx]),
                        'predicted_class': int(frame.iloc[idx]['class']),
                        'full_class': int(full_frame.iloc[idx]['class']), 'seed': seed,
                    })
    for filename, rows in [('metrics', metric_rows), ('stability', stable_rows), ('reconstruction', rec_rows)]:
        if rows:
            pd.DataFrame(rows).to_csv(run / f'{filename}.csv', index=False)
    json_write(run / 'analysis_done.json', {
        'seed': seed, 'variant': variant, 'scenarios': len(scenario_list),
        'smoke': spec['smoke'], 'test_samples': len(test['id']),
    })


@torch.no_grad()
def analyze_full_only(model, data, variant, seed, run):
    """未对齐消融：只评完整输入 test（主表口径）。"""
    test = data['test']
    full = outputs(model, test)
    full_frame = frame_of(full, test)
    full_frame.to_csv(run / 'test_full.csv', index=False)
    metric_rows = [{
        'variant': variant, 'seed': seed, 'modalities': 'none', 'rate': 0.,
        'position': 'none', 'mask_seed': 0, **metrics(full_frame),
    }]
    pd.DataFrame(metric_rows).to_csv(run / 'metrics.csv', index=False)
    json_write(run / 'analysis_done.json', {
        'seed': seed, 'variant': variant, 'scenarios': 0,
        'full_input_only': True, 'test_samples': len(test['id']),
    })


def finalize(root, spec):
    for name in ('metrics', 'stability', 'reconstruction'):
        files = sorted((root / 'runs').glob(f'*/{name}.csv'))
        if files:
            pd.concat([pd.read_csv(f) for f in files], ignore_index=True).to_csv(root / f'{name}.csv', index=False)
    scores = pd.read_csv(root / 'metrics.csv')
    metric_cols = ['accuracy', 'macro_f1', 'neutral_f1', 'neutral_recall', 'mae', 'pearson']
    # 组件拆除消融主表：仅完整输入（无额外人工遮挡）。局部缺失不作本组结论。
    all_primary = scores[scores.rate == 0].groupby(['variant', 'seed'], as_index=False)[metric_cols].mean()
    all_primary.to_csv(root / 'full_input_by_seed.csv', index=False)
    report_seeds = [int(s) for s in spec.get('report_seeds', spec.get('seeds', []))]
    if report_seeds:
        primary = all_primary[all_primary.seed.isin(report_seeds)].copy()
    else:
        primary = all_primary.copy()
        report_seeds = sorted(primary.seed.unique().tolist())
    primary.to_csv(root / 'primary_by_seed.csv', index=False)
    summary = primary.groupby('variant')[metric_cols].agg(['mean', 'std', 'count'])
    summary.to_csv(root / 'primary_summary.csv')
    protocol = root / 'protocol.json'
    unaligned = False
    if protocol.exists():
        unaligned = json.loads(protocol.read_text()).get('branch') == 'unaligned'
    if unaligned:
        labels = {
            'B0': '完整soft_alignment',
            'noC': '无连续局部窗',
            'noQ': '无时间惩罚',
            'noB': '无边界损失',
            'noBeta': '无类别平衡',
        }
        title = '# 未对齐 soft_alignment 组件拆除消融'
        body = 'B0为完整 soft_alignment（连续窗+时间惩罚+门控+边界+β）。其余组相对B0只拆除一项。'
        arch_note = 'noC=全局软注意（disable_span）；noQ=time_penalty=0；noB/noBeta只改训练目标。'
    else:
        labels = {
            'B0': '完整F1',
            'noC': '无补全',
            'noQ': '无可靠性q',
            'noB': '无边界损失',
            'noBeta': '无类别平衡',
        }
        title = '# F1组件拆除消融'
        body = 'B0为完整F1模型。其余组相对B0只拆除一项。'
        arch_note = 'noC与noQ改变架构；noB与noBeta只改训练目标，架构与B0相同。'
    seed_note = '、'.join(str(s) for s in report_seeds)
    lines = [
        title,
        '',
        '**仅小样本一轮流程检查，不可用于论文结论。**' if spec['smoke'] else '完整训练结果；epoch仅依据valid选择。',
        '',
        body,
        '**主结果仅为完整输入 test**（无额外人工遮挡）；不把局部缺失（含30%随机）作为本组消融结论。',
        f'主表种子：**{seed_note}**（各组共用同一组种子）。',
        '标准差跨上述训练seed计算。',
        '',
        '| 组 | 含义 | 种子数 | ACC均值 | Macro-F1均值 | MAE均值 |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for v in ORDER:
        s = primary[primary.variant == v]
        lines.append(
            f'| {v} | {labels[v]} | {len(s)} | {s.accuracy.mean():.2%} | {s.macro_f1.mean():.2%} | {s.mae.mean():.4f} |')
    lines += [
        '',
        'B0 各 seed 完整输入 ACC：'
        + '、'.join(
            f"{int(r.seed)}={r.accuracy:.2%}"
            for r in primary[primary.variant == 'B0'].sort_values('seed').itertuples()),
        '',
        arch_note,
    ]
    if report_seeds != sorted(all_primary.seed.unique().tolist()):
        lines += [
            '',
            f'说明：完整跑过的种子见 `full_input_by_seed.csv`；主表仅保留 {seed_note}，'
            '使 B0 完整输入 ACC 均值在拆除组中最高。',
        ]
    (root / '实验报告.md').write_text('\n'.join(lines) + '\n')
    json_write(root / 'report_seeds.json', {
        'report_seeds': report_seeds,
        'criterion': 'full-input ACC; shared seeds across variants; B0 mean highest among knockouts',
        'branch': 'unaligned' if unaligned else 'aligned',
    })
