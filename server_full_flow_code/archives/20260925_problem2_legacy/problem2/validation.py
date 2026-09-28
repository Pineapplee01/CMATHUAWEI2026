"""完整/缺失场景评估与可靠性检查。"""
import itertools
import numpy as np
import pandas as pd
import torch
from shared.common import path, to_device, json_write
from shared.data import block_mask, assert_disjoint
from problem2.runtime import dataset, loader
from problem2.evaluation import metrics, grouped_bootstrap, error_plots, robustness_plot
from problem2.output_contract import coherent_frame, project_intensity

COMBINATIONS = [p for n in (1, 2, 3) for p in itertools.combinations(range(3), n)]

@torch.no_grad()
def evaluate(model, ds, cfg, rate=0, modalities=(0, 1, 2), position='random', noise=0, mask_output=None, reconstruction=True):
    model.eval()
    rng = np.random.default_rng(cfg['seed'] + 10000)
    noise_rng = torch.Generator(device=cfg['device']).manual_seed(cfg['seed'] + 20000)
    rows, reconstruction_rows, retained_masks = [], [], []
    for raw in loader(ds, cfg):
        batch = to_device(raw, cfg['device'])
        B = block_mask(batch, rate, modalities, rng, position)
        if mask_output is not None:
            retained_masks.append(B.cpu().numpy())
        full = model(batch) if rate and reconstruction else None
        if noise:
            # 只扰动真实观测；不改变 O/P，不将缺失占位改成观测。
            for m, name in enumerate(('audio', 'vision'), 1):
                perturbation = torch.randn(batch[name].shape, device=cfg['device'], generator=noise_rng)
                batch[name] = batch[name] + noise * perturbation * getattr(model, name + '_std') * batch['O'][:, m, :, None]
        out = model(batch, B)
        probabilities = out['logits'].softmax(-1)
        for i, sid in enumerate(batch['id']):
            rows.append({'id': sid, 'class': int(probabilities[i].argmax()),
                         'intensity': float(out['intensity'][i]),
                         'true_class': int(batch['c'][i]), 'true_intensity': float(batch['y'][i]),
                         'low_evidence': bool(out['low_evidence'][i]),
                         'new_missing_fraction': float((batch['O'][i] & ~B[i]).sum() / batch['O'][i].sum().clamp_min(1)),
                         'validity_uncertain': True})
        if full is not None:
            H = batch['P'] & batch['O'] & ~out['M']
            for m, name in enumerate(('text', 'audio', 'vision')):
                mse = (full['values'][m] - out['mu'][m]).square().mean(-1)
                hidden = H[:, m].cpu().numpy()
                mse_np = mse.cpu().numpy()
                q_np = out['q'][:, m].cpu().numpy()
                errors = (out['intensity'] - batch['y']).abs().cpu().numpy()
                if cfg.get('output_rule') == 'class_constrained':
                    final_y = project_intensity(out['logits'].argmax(-1).cpu().numpy(), out['intensity'].cpu().numpy())
                    errors = np.abs(final_y-batch['y'].cpu().numpy())
                for i, t in zip(*np.where(hidden)):
                    reconstruction_rows.append({'id': batch['id'][i], 'modality': name,
                        'position': int(t), 'q': float(q_np[i, t]), 'mse': float(mse_np[i, t]),
                        'prediction_error': float(errors[i])})
    if mask_output is not None:
        mask_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(mask_output, B=np.concatenate(retained_masks), ids=np.asarray(ds.ids))
    frame = pd.DataFrame(rows)
    if cfg.get('output_rule') == 'class_constrained':
        frame = coherent_frame(frame)
    return frame, pd.DataFrame(reconstruction_rows)



def evaluate_suite(model, cfg, split, noise=0, quick=False):
    ds = dataset(cfg, split)
    # 冻结后测试仍检查和 train/valid 的隔离。
    if split == 'test':
        assert_disjoint(dataset(cfg, 'train'), dataset(cfg, 'valid'), ds)
    output = path(cfg['output']) / split; output.mkdir(parents=True, exist_ok=True)
    frame, _ = evaluate(model, ds, cfg, noise=noise)
    frame.to_csv(output / 'predictions.csv', index=False)
    json_write(output / 'metrics.json', {'metrics': metrics(frame),
               'video_group_bootstrap_95ci': grouped_bootstrap(frame, cfg['bootstrap_repeats'], cfg['seed'])})
    error_plots(frame, path('problem2/figures') / output.parent.name / split)
    strata = []
    for name, mask in [('near_neutral', frame.true_intensity.abs() <= .5),
                       ('strong', frame.true_intensity.abs() >= 2),
                       ('other', (frame.true_intensity.abs() > .5) & (frame.true_intensity.abs() < 2))]:
        subset = frame[mask]
        if len(subset):
            strata.append({'stratum': name, 'count': len(subset), **metrics(subset)})
    pd.DataFrame(strata).to_csv(output / 'error_strata.csv', index=False)
    if quick:
        missing, _ = evaluate(model, ds, cfg, cfg['validation_rate'], reconstruction=False)
        missing.to_csv(output / 'predictions_missing30.csv', index=False)
        result = pd.DataFrame([
            {'modalities':'TAV','rate':0,'position':'random',**metrics(frame)},
            {'modalities':'TAV','rate':cfg['validation_rate'],'position':'random',**metrics(missing)}])
        result.to_csv(output / 'quick_metrics.csv',index=False)
        print(result.to_string(index=False),flush=True)
        return
    rows, groups = [], []
    for modalities in COMBINATIONS:
        for rate in [0, *cfg.get('evaluation_rates', [.1, .3, .5, .7]), 1.0]:
            for position in (['random'] if rate in (0, 1) else ['start', 'middle', 'end', 'random']):
                scenario_name = ''.join('TAV'[m] for m in modalities) + f'_{rate}_{position}'
                pred, rec = evaluate(model, ds, cfg, rate, modalities, position,
                                     mask_output=output / 'masks' / f'{scenario_name}.npz')
                scenario = {'modalities': ''.join('TAV'[m] for m in modalities), 'rate': rate,
                            'position': position, 'stress_test': rate >= .7,
                            'actual_new_missing_fraction': float(pred.new_missing_fraction.mean())}
                rows.append({**scenario, **metrics(pred)})
                if len(rec):
                    rec['q_bin'] = pd.cut(rec.q, np.linspace(0, 1, 6), include_lowest=True).astype(str)
                    for (modality, qbin), r in rec.groupby(['modality', 'q_bin']):
                        groups.append({**scenario, 'modality': modality, 'q_bin': qbin,
                                       'count': len(r), 'mean_q': float(r.q.mean()),
                                       'mse': float(r.mse.mean()), 'prediction_error': float(r.prediction_error.mean())})
    result = pd.DataFrame(rows)
    result.to_csv(output / '结果表格.csv', index=False)
    pd.DataFrame(groups).to_csv(output / 'reliability_groups.csv', index=False)
    robustness_plot(result, path('problem2/figures') / output.parent.name / split)
