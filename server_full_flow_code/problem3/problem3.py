"""问题三对齐版入口：python -m problem3.problem3 [--smoke]"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from shared.common import path, seed_all, json_write
from problem3.model_io import load_model, appendix_batch
from problem3.explain import explain_sample
from problem3.report import (
    evaluate_split, write_explanation_cards_md, plot_figures,
    plot_split_summary, write_deliverable_report, _jsonable,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='problem3/config.json')
    p.add_argument('--device')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    cfg = json.loads(path(args.config).read_text(encoding='utf-8'))
    if args.device:
        cfg['device'] = args.device
    seed_all(cfg.get('seed', 2026))

    model, train, saved, tok, digest = load_model(cfg)
    files = sorted(path(cfg['appendix4']).glob('*.pkl'))
    if not files:
        raise SystemExit('未找到附件4')
    if args.smoke:
        files = files[:1]

    out = path(cfg['results']); out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / 'figures'; fig_dir.mkdir(exist_ok=True)

    rows, cards, batches = [], [], []
    for i, f in enumerate(files, 1):
        batch, text, tokens, sid = appendix_batch(f, train, cfg['device'])
        card = explain_sample(model, batch, tokens, cfg)
        card.update(sample_id=sid, source_file=f.name, raw_text=text)
        cards.append(card)
        batches.append(batch)
        rows.append({
            'sample_id': sid, 'raw_text': text,
            'polarity': card['polarity'], 'intensity': card['intensity'],
            'I_text': card['internal_attribution']['text'],
            'I_audio': card['internal_attribution']['audio'],
            'I_vision': card['internal_attribution']['vision'],
            'phi_text': (card.get('modality_marginal') or {}).get('phi_p', [None])[0],
            'phi_audio': (card.get('modality_marginal') or {}).get('phi_p', [None, None])[1],
            'phi_vision': (card.get('modality_marginal') or {}).get('phi_p', [None, None, None])[2],
            'main_support': card['main_support_modality'],
            'key_text': card['key_evidence'].get('text'),
            'key_audio': card['key_evidence'].get('audio'),
            'key_vision': card['key_evidence'].get('vision'),
            'comprehensiveness': card['comprehensiveness'],
            'sufficiency': card['sufficiency'],
            'stability': card.get('stability'),
        })
        print(f'[{i}/{len(files)}] {sid} {card["polarity"]} y={card["intensity"]:.3f} '
              f'support={card["main_support_modality"]}', flush=True)

    pd.DataFrame(rows).to_csv(out / '附件4_可解释结果.csv', index=False)
    json_write(out / '解释卡片.json', _jsonable({
        'checkpoint_sha256': digest, 'best_epoch': saved.get('best_epoch'),
        'method': 'grad_input + nonempty_context_marginal + continuous_evidence',
        'cards': cards,
    }))
    write_explanation_cards_md(cards, out / '典型样本解释卡.md', limit=3)
    write_explanation_cards_md(cards, out / '附件4_全部解释卡.md', limit=None)

    # 全部样本出图
    id_to_batch = {c['sample_id']: b for c, b in zip(cards, batches)}
    for card in cards:
        plot_figures(model, id_to_batch[card['sample_id']], card, fig_dir,
                     eta=float(cfg.get('evidence_eta', 0.7)))
        print(f'FIG {card["sample_id"]}', flush=True)

    # 测试集
    print('评价测试集…', flush=True)
    test_frame, summary = evaluate_split(model, train, saved, cfg['device'], split='test')
    test_frame.to_csv(out / '测试集预测.csv', index=False)
    error_info = plot_split_summary(test_frame, summary, fig_dir, split='test')
    error_info['n_test'] = len(test_frame)
    json_write(out / '测试集指标.json', {**summary, **error_info})

    write_deliverable_report(
        cfg=cfg, digest=digest, train_cfg=train, saved=saved, cards=cards,
        summary=summary, error_info=error_info, out_dir=out, variant='aligned',
    )
    print(json.dumps({
        'done': True, 'n': len(cards), 'test_acc': round(summary['accuracy'], 4),
        'report': str(out / '问题三交付说明.md'),
    }, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
