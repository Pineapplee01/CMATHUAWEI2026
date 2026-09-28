"""从已有解释卡片导出全部 20 条解释卡与图（不对 Stab 重算）。"""
import json
from pathlib import Path
from problem3.report import write_explanation_cards_md, plot_figures, write_deliverable_report
from problem3.model_io import load_model, appendix_batch
from shared.common import path


def main(device='cuda:2'):
    cfg = json.loads(Path('problem3/config.json').read_text(encoding='utf-8'))
    cfg['device'] = device
    out = Path(cfg['results'])
    fig_dir = out / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    pack = json.loads((out / '解释卡片.json').read_text(encoding='utf-8'))
    cards = pack['cards']
    write_explanation_cards_md(cards, out / '典型样本解释卡.md', limit=3)
    write_explanation_cards_md(cards, out / '附件4_全部解释卡.md', limit=None)
    print(f'cards md: {len(cards)}', flush=True)
    model, train, saved, tok, digest = load_model(cfg)
    files = {f.stem: f for f in sorted(path(cfg['appendix4']).glob('*.pkl'))}
    for i, card in enumerate(cards, 1):
        stem = Path(card['source_file']).stem
        batch, *_ = appendix_batch(files[stem], train, cfg['device'])
        plot_figures(model, batch, card, fig_dir, eta=float(cfg.get('evidence_eta', 0.7)))
        print(f'FIG {i}/20 {card["sample_id"]}', flush=True)
    metrics = json.loads((out / '测试集指标.json').read_text(encoding='utf-8'))
    summary = {k: metrics[k] for k in (
        'accuracy', 'macro_f1', 'mae', 'pearson', 'split_path', 'split_sha256', 'identity_checked'
    ) if k in metrics}
    error_info = {
        'n_error': metrics['n_error'],
        'error_rate_by_true_class': metrics['error_rate_by_true_class'],
        'n_test': metrics.get('n_test'),
        'confusion': metrics.get('confusion'),
    }
    write_deliverable_report(
        cfg=cfg, digest=pack.get('checkpoint_sha256'), train_cfg={},
        saved={'best_epoch': pack.get('best_epoch')}, cards=cards,
        summary=summary, error_info=error_info, out_dir=out, variant='aligned',
    )
    n_imp = len(list(fig_dir.glob('*_importance.pdf')))
    print(json.dumps({'done': True, 'n_cards': len(cards), 'n_importance_pdf': n_imp}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda:2')
    main(p.parse_args().device)
