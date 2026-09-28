"""从逐样本预测重算一致输出指标，等权汇总类型/位置/长度及消融。"""
import argparse
import json
import numpy as np
import pandas as pd
from shared.common import path, json_write
from shared.plotting import pyplot
from problem2.output_contract import coherent_frame
from problem2.evaluation import metrics


def summarize(directory, figures=None):
    directory = path(directory)
    full = coherent_frame(pd.read_csv(directory/'complete_predictions.csv'))
    complete = metrics(full)
    original = json.loads((directory/'complete_metrics.json').read_text())
    table = pd.read_csv(directory/'scenarios.csv')
    rows = []
    for row in table.to_dict('records'):
        key = f"{row['modalities']}_{row['rate']}_{row['position']}_{row['seed']}"
        source = directory/'scenarios'/key/'predictions.csv'
        pred = pd.read_csv(source)
        if set(pred.id) != set(full.id) or pred.id.duplicated().any():
            raise AssertionError(f'{source}: 样本覆盖错误')
        pred = pred.set_index('id').loc[full.id].reset_index()
        if not (pred.true_class.to_numpy() == full.true_class.to_numpy()).all():
            raise AssertionError('缺失前后标签改变')
        np.testing.assert_allclose(pred.true_intensity, full.true_intensity, rtol=0, atol=1e-10)
        raw = metrics(pred)
        final = metrics(coherent_frame(pred))
        row.update(final)
        row.update(raw_mae=raw['mae'], raw_pearson=raw['pearson'],
                   raw_head_disagreement=raw['head_disagreement'],
                   accuracy_drop=complete['accuracy']-final['accuracy'],
                   mae_increase=final['mae']-complete['mae'])
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(directory/'coherent_scenarios.csv',index=False)
    keys = ['modalities','rate','position']
    columns = ['accuracy','macro_f1','weighted_f1','mae','pearson','accuracy_drop',
               'mae_increase','actual_removed_fraction','mean_interval_slots']
    # 随机位置先平均三个seed，然后与三个固定位置等权，避免随机位置被赋予三倍权重。
    by_position = table.groupby(keys,as_index=False)[columns].mean()
    by_position.to_csv(directory/'position_effects.csv',index=False)
    by_duration = by_position.groupby(['modalities','rate'],as_index=False)[columns].mean()
    by_duration.to_csv(directory/'duration_effects.csv',index=False)
    table[table.position=='random'].groupby(['modalities','rate'])[['accuracy','macro_f1','mae','pearson']].agg(['mean','std']).to_csv(directory/'random_seed_variation.csv')
    missing_score = (1-by_position.macro_f1+by_position.mae/3).mean()
    selection = .5*(1-complete['macro_f1']+complete['mae']/3)+.5*missing_score
    report = {'complete_final':complete,'complete_raw':original,'scenarios':len(table),
              'local_mean':by_position[columns].mean().to_dict(),
              'aggregation':'seeds averaged first; positions, rates and modality combinations equally weighted',
              'selection_score':float(selection),
              'selection_scope':'meaningful across models only when evaluated with identical scenario grid',
              'output_rule':'class_constrained, epsilon=1e-6; raw values retained',
              'duration_unit':'aligned slots; no seconds inferred'}
    json_write(directory/'summary.json',report)
    if figures:
        figures = path(figures);figures.mkdir(parents=True,exist_ok=True)
        plt = pyplot()
        for filename, columns, labels in [
            ('accuracy_mae',['accuracy','mae'],['Accuracy','MAE']),
            ('f1_pearson',['macro_f1','pearson'],['Macro-F1','Pearson'])]:
            fig, axes = plt.subplots(1,2,figsize=(11,4))
            for name, group in by_duration.groupby('modalities'):
                for ax, column, label in zip(axes,columns,labels):
                    ax.plot(group.rate,group[column],marker='o',label=name)
                    ax.set(xlabel='Requested interval / effective span',ylabel=label)
            axes[0].legend(fontsize=8);fig.tight_layout()
            fig.savefig(figures/f'{filename}.png',dpi=300);plt.close(fig)
        matrix = by_position[by_position.rate==.3].pivot(index='modalities',columns='position',values='accuracy_drop')
        matrix=matrix.reindex(columns=['start','middle','end','random'])
        fig, ax = plt.subplots(figsize=(7,4))
        im=ax.imshow(matrix.to_numpy()*100,cmap='coolwarm')
        ax.set(xticks=range(len(matrix.columns)),xticklabels=matrix.columns,
               yticks=range(len(matrix)),yticklabels=matrix.index,
               title='Accuracy drop at 30% local missingness (percentage points)')
        for i in range(len(matrix)):
            for j in range(len(matrix.columns)):
                ax.text(j,i,f'{matrix.iloc[i,j]*100:.1f}',ha='center',va='center',fontsize=9)
        fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(figures/'position_effects.png',dpi=300);plt.close(fig)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True)
    parser.add_argument('--figures')
    args=parser.parse_args();print(json.dumps(summarize(args.directory,args.figures),ensure_ascii=False,indent=2))
