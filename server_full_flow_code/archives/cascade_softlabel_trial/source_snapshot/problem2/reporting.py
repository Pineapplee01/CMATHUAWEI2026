"""将保存的真实指标与历史汇总为优化图表，禁止填造缺失实验。"""
import json
import pandas as pd
from shared.common import path, config, json_write
from shared.plotting import pyplot


def summarize():
    cfg=config('configs/final.json')
    directory=path('problem2/results/optimization')
    comparison=pd.read_csv(directory/'comparison.csv')
    frozen=json.loads((directory/'frozen_selection.json').read_text())
    selected=frozen['selected']
    rows=[]
    for name in ['mixup','no_completion','no_reliability']:
        history=pd.read_csv(path(f'problem2/results/{name}/training_history.csv'))
        row=history.loc[history.selection_score.idxmin()]
        rows.append({'round':'round1_historical', 'variant':name, 'best_epoch':int(row.epoch),
                     'valid_accuracy':row.full_accuracy,'valid_missing30_accuracy':row.missing_accuracy,
                     'valid_macro_f1':row.full_macro_f1,'valid_missing30_macro_f1':row.missing_macro_f1})
    ablation=pd.DataFrame(rows); ablation.to_csv(directory/'ablation_comparison.csv',index=False)
    figure=path('problem2/figures/optimization');figure.mkdir(parents=True,exist_ok=True)
    plt=pyplot();fig,ax=plt.subplots(figsize=(16,5))
    ax.bar(comparison.run, comparison.full_accuracy*100)
    ax.axhline(70,color='red',linestyle='--',label='70% reference (previous target)')
    ax.set(ylabel='Validation three-class accuracy (%)',ylim=(50,72))
    ax.tick_params(axis='x',rotation=65);ax.legend()
    fig.savefig(figure/'optimization_accuracy.png',dpi=300,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name in ['baseline','mixup',selected]:
        h=pd.read_csv(path(f'problem2/results/{name}/training_history.csv'))
        axes[0].plot(h.epoch,h.train_loss,label=name)
        axes[1].plot(h.epoch,h.full_accuracy,label=name)
    axes[0].set(xlabel='Epoch',ylabel='Training loss (objectives differ)')
    axes[1].set(xlabel='Epoch',ylabel='Validation accuracy');axes[1].legend()
    fig.savefig(figure/'learning_curves.png',dpi=300,bbox_inches='tight');plt.close(fig)
    valid=json.loads((path(cfg['output'])/'valid/metrics.json').read_text())
    test=json.loads((path(cfg['output'])/'test/metrics.json').read_text())
    robust=pd.read_csv(path(cfg['output'])/'valid/结果表格.csv')
    reliability=pd.read_csv(path(cfg['output'])/'valid/reliability_groups.csv')
    fidelity=pd.read_csv(path(cfg['explanation_output'])/'valid_explanations/fidelity_summary.csv')
    fidelity.groupby('modality')[['key_drop','random_mean_drop','ranking_stability']].mean().to_csv(
        path(cfg['explanation_output'])/'valid_fidelity_means.csv')
    target_split = cfg['target_split']
    target_accuracy = cfg.get('target_accuracy', .7)
    acceptance = {'valid': valid, 'test': test}[target_split]
    report={'target_split': target_split, 'target_accuracy': target_accuracy, 'selected':selected,'ablation_scope':'round1 only, not evidence for round2 selected model','valid':valid,'test':test,'valid_scenarios':len(robust),
            'reliability_groups':len(reliability),'primary_target_met':acceptance['metrics']['accuracy']>=target_accuracy,
            'test_target_met':test['metrics']['accuracy']>=target_accuracy,
            'parameter_count':int(comparison.set_index('run').loc[selected,'trainable_parameters'])}
    json_write(directory/'final_summary.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__': summarize()
