"""验证集六张图：沿用测试图样式，仅读取验证推理结果。"""
from 论文图表样式 import NAMES, save_clean, ablation, duration, update_report
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/'valid_experiments'
FIG=HERE/'valid_figures';FIG.mkdir(exist_ok=True)
TABLE=OUT/'plot_data';TABLE.mkdir(exist_ok=True)
SEEDS={'aligned':[2027,2028,2035],'unaligned':[2026,2031,2033]}
VAR=['B0','noC','noQ','noB','noBeta']
short_labels={'aligned':['完整','去补全','去可靠性','去边界','去平衡']}
COLORS=['#BD5148','#44789A','#8574A4','#528B78','#C39A4B']
MARKERS=['o','s','D','^','v']
LAYOUT_NAMES={'aligned':'对齐','unaligned':'未对齐'}
GROUP_NAMES={
    'aligned':dict(zip(VAR,['完整模型','无补全','无可靠性','无边界损失','无类别平衡'])),
    'unaligned':dict(zip(VAR,['完整模型','无局部窗','无时间惩罚','无边界损失','无类别平衡'])),
}
def group_labels(layout):
    return [GROUP_NAMES[layout][v].replace('无边界损失','无边界\n损失').replace('无类别平衡','无类别\n平衡').replace('无时间惩罚','无时间\n惩罚')+'\n'+v for v in VAR]

from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
font_manager.fontManager.addfont(HERE/'fonts/simsun.ttc')
# 英文、数字先匹配Times New Roman；中文逐字回退至实际宋体字体。
font_manager.findfont('Times New Roman',fallback_to_default=False)
font_manager.findfont('SimSun',fallback_to_default=False)
plt.rcParams.update({
    'font.family':['Times New Roman','SimSun'], 'font.size':11,
    'axes.titlesize':12,'axes.titleweight':'normal','axes.titlepad':12,
    'axes.labelsize':11,'text.color':'black','axes.labelcolor':'black',
    'xtick.color':'black','ytick.color':'black',
    'axes.edgecolor':'#B8C3CC','axes.linewidth':.65,
    'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True,
    'grid.color':'#D9E1E7','grid.linewidth':.65,
    'xtick.major.size':3,'ytick.major.size':3,
    'legend.frameon':False,'legend.fontsize':10,
    'lines.linewidth':1.8,'lines.markersize':5,
    'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
    'savefig.dpi':360,'pdf.fonttype':42,
    'figure.constrained_layout.h_pad':.12,'figure.constrained_layout.w_pad':.10,
    'figure.constrained_layout.hspace':.10,'figure.constrained_layout.wspace':.10,
})
heat_cmap=LinearSegmentedColormap.from_list('paper_blue',['#F2F6F8','#CEE0E8','#82B3C7','#3C809F','#174C6B'])

def save(fig,name):
    save_clean(fig,FIG,name)

def per_seed(df,keys,values):
    # 随机掩码先平均，再平均位置，最后平均缺失集合；保留seed为独立重复层级。
    first=df.groupby(['variant','seed','modalities','rate','position'],as_index=False)[values].mean()
    return first.groupby(keys,as_index=False)[values].mean()

def plot_validation_confusion():
    """对齐和未对齐分别汇总三个种子的完整验证预测，禁止漏画任一分支。"""
    fig,axes=plt.subplots(1,2,figsize=(9.2,4.8),layout='constrained')
    conf={}; records=[]; reference_labels=None
    for ax,(layout,seeds) in zip(axes,SEEDS.items()):
        mats=[]
        for seed in seeds:
            if layout=='aligned':
                source=OUT/'runs'/f'B0_seed{seed}'/'valid_full.csv'
            else:
                suffix='five_seeds' if seed<=2030 else 's2031_2035'
                source=ROOT/f'problem2_retrain_v2/results/unaligned/ablations/softalign_gate_component_ablation_{suffix}/runs/B0_seed{seed}/valid_full.csv'
            pred=pd.read_csv(source)
            assert len(pred)==728 and pred.id.nunique()==728,source
            labels=pred.set_index('id').true_class.sort_index()
            if reference_labels is None:reference_labels=labels
            else:pd.testing.assert_series_equal(labels,reference_labels)
            cm=confusion_matrix(pred.true_class,pred['class'],labels=[0,1,2],normalize='true')
            assert np.allclose(cm.sum(axis=1),1),source
            mats.append(cm)
            for i in range(3):
                for j in range(3):
                    records.append({'layout':layout,'seed':seed,'true_class':i,
                                    'predicted_class':j,'proportion':float(cm[i,j]),
                                    'source':str(source.relative_to(ROOT))})
        cm=np.mean(mats,axis=0)*100;conf[layout]=cm.tolist()
        im=ax.imshow(cm,vmin=0,vmax=100,cmap=heat_cmap)
        for i in range(3):
            for j in range(3):
                ax.text(j,i,f'{cm[i,j]:.1f}%',ha='center',va='center',color='black',fontsize=14)
        ax.set(xlabel='预测类别',ylabel='真实类别',xticks=range(3),yticks=range(3),
               xticklabels=['负向','中性','正向'],yticklabels=['负向','中性','正向'])
        ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,3,1),minor=True)
        ax.grid(which='minor',color='white',linewidth=2);ax.tick_params(which='both',length=0)
        for spine in ax.spines.values():spine.set_visible(False)
    assert set(conf)=={'aligned','unaligned'}
    fig.colorbar(im,ax=axes,label='Row-normalized proportion (%)',shrink=.82,aspect=25,pad=.025)
    save(fig,'06_confusion')
    pd.DataFrame(records).to_csv(TABLE/'完整验证集类别混淆统计.csv',index=False)
    return conf


def report():
    metrics={'aligned':pd.read_csv(OUT/'metrics.csv')}
    stable=pd.read_csv(OUT/'stability.csv');rec=pd.read_csv(OUT/'reconstruction.csv')
    full={x:df[df.rate==0].copy() for x,df in metrics.items()}
    assert len(full['aligned'])==15 and set(full['aligned'].seed)==set(SEEDS['aligned'])
    # 已保存的未对齐完整验证预测，逐文件重新计算四项指标。
    from problem2_retrain_v2.evaluation import metrics as evaluate_metrics
    extra=[]
    for seed in SEEDS['unaligned']:
        suffix='five_seeds' if seed<=2030 else 's2031_2035'
        for variant in VAR:
            path=ROOT/f'problem2_retrain_v2/results/unaligned/ablations/softalign_gate_component_ablation_{suffix}/runs/{variant}_seed{seed}/valid_full.csv'
            pred=pd.read_csv(path);assert len(pred)==728 and pred.id.nunique()==728
            extra.append({'seed':seed,'variant':variant,**evaluate_metrics(pred)})
    full['unaligned']=pd.DataFrame(extra)
    full['unaligned'].to_csv(TABLE/'unaligned_full_validation.csv',index=False)
    summary=[]
    for layout,data in full.items():
        summary.extend(ablation(data,layout,FIG))
    pd.DataFrame(summary).to_csv(TABLE/'full_summary.csv',index=False)

    partial=metrics['aligned'].query('rate>0').copy()
    # 这里只比较三个单模态随机缺失；组合模态只有30%，不虚构跨跨度网格。
    curves=per_seed(partial.query("position=='random' and modalities in ['T','A','V']"),['variant','seed','rate'],['accuracy','mae'])
    curves.to_csv(TABLE/'duration_by_seed.csv',index=False)
    duration(curves,FIG)

    main=partial[np.isclose(partial.rate,.3)]
    heat=main.groupby(['variant','seed','modalities','position'],as_index=False).accuracy.mean()
    positions=['start','middle','end','random'];mods=['T','A','V','TA','TV','AV','TAV']
    matrices={v:heat.query('variant==@v').groupby(['modalities','position']).accuracy.mean().unstack().reindex(index=mods,columns=positions).to_numpy()*100 for v in VAR}
    lo=min(a.min() for a in matrices.values());hi=max(a.max() for a in matrices.values())
    fig,grid=plt.subplots(2,3,figsize=(11.5,8),layout='constrained')
    axes=grid.flatten()
    for k,(ax,v) in enumerate(zip(axes,VAR)):
        im=ax.imshow(matrices[v],vmin=lo,vmax=hi,cmap=heat_cmap,aspect='auto')
        for i in range(7):
            for j in range(4):ax.text(j,i,f'{matrices[v][i,j]:.1f}',ha='center',va='center',fontsize=11,color='white' if matrices[v][i,j]>lo+.58*(hi-lo) else '#243443')
        ax.set(title=f'({chr(97+k)}) {GROUP_NAMES["aligned"][v]}（{v}）',xticks=range(4),xticklabels=['起始','中间','末尾','随机'],yticks=range(7),yticklabels=['文本','音频','视觉','文本＋音频','文本＋视觉','音频＋视觉','三模态'],xlabel='缺失位置')
        ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,7,1),minor=True)
        ax.grid(which='minor',color='white',linewidth=1.5);ax.tick_params(which='both',length=0)
        for spine in ax.spines.values():spine.set_visible(False)
    axes[-1].axis('off')
    axes[-1].text(.04,.9,'图中符号说明',fontsize=12,transform=axes[-1].transAxes)
    axes[-1].text(.04,.77,'T　文本\nA　音频\nV　视觉\n\n组合：对应模态共同缺失\n\n单元格：Mean accuracy (%)\n三个指定训练种子的均值',fontsize=11,linespacing=1.7,color='#526170',va='top',transform=axes[-1].transAxes)
    fig.colorbar(im,ax=axes[:5].tolist(),label='Accuracy (%)',shrink=.78,aspect=35,pad=.025)
    save(fig,'03_type_position')
    
    st=stable[np.isclose(stable.rate,.3)]
    stseed=per_seed(st,['variant','seed'],['flip_rate','raw_intensity_shift','js'])
    stseed.to_csv(TABLE/'stability_by_seed.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(13.4,4.9),layout='constrained')
    for ax,metric,label,scale in zip(axes,['flip_rate','raw_intensity_shift','js'],['Class flip rate (%)','Raw intensity absolute shift','JS divergence'],[100,1,1]):
        values=[stseed.query('variant==@v')[metric].to_numpy()*scale for v in VAR]
        low=min(min(x.min(),x.mean()-x.std(ddof=1)) for x in values)
        high=max(max(x.max(),x.mean()+x.std(ddof=1)) for x in values)
        span=max(high-low,1e-6)
        for i,(v,c,vals) in enumerate(zip(VAR,COLORS,values)):
            ax.errorbar(i,vals.mean(),yerr=vals.std(ddof=1),fmt='none',ecolor=c,capsize=5,elinewidth=1.6,capthick=1.3,zorder=2)
            ax.scatter(i+np.array([-.13,0,.13]),vals,s=17,color=c,alpha=.30,zorder=3)
            ax.scatter(i,vals.mean(),s=76,marker=MARKERS[i],color=c,edgecolors='white',linewidths=1,zorder=4)
            top=max(vals.max(),vals.mean()+vals.std(ddof=1))
            digits=2 if metric=='flip_rate' else (4 if metric=='js' else 3)
            ax.text(i,top+.065*span,f'{vals.mean():.{digits}f}',ha='center',va='bottom',fontsize=12,color=c)
        ax.set(xticks=range(5),xticklabels=short_labels['aligned'],ylabel=label,xlim=(-.5,4.5),ylim=(max(0,low-.13*span),high+.27*span))
        ax.grid(axis='y',alpha=.45);ax.tick_params(axis='x',length=0,pad=9)
    for i,ax in enumerate(axes):ax.set_title(['(a) Class flip rate ↓','(b) Intensity shift ↓','(c) JS divergence ↓'][i],fontsize=13,pad=14)
    save(fig,'04_stability')
    
    r=rec.query("variant=='B0' and position=='random' and modalities in ['T','A','V']").copy()
    r=r[r.target_modality==r.modalities]
    rseed=r.groupby(['seed','target_modality','rate','kind'],as_index=False).cosine_distance.mean()
    rseed.to_csv(TABLE/'completion_by_seed.csv',index=False)
    # 保留横向折线图：手动控制标题、图例和绘图区，避免大块底部留白。
    fig,axes=plt.subplots(1,3,figsize=(11.6,4.2),sharex=True,sharey=True)
    fig.subplots_adjust(left=.073,right=.985,top=.93,bottom=.22,wspace=.16)
    
    for panel_index,(ax,m) in enumerate(zip(axes,['T','A','V'])):
        groups={kind:rseed[(rseed.target_modality==m)&(rseed.kind==kind)].groupby('rate').cosine_distance.agg(['mean','std'])
                for kind in ['reconstructed','observed_mean']}
        x=groups['reconstructed'].index.to_numpy()*100
        for kind,color,label,marker,style in [('reconstructed','#AD5354','学习补全','o','-'),('observed_mean','#2F748A','可见槽均值填充','s',(0,(4,2.5)))]:
            g=groups[kind];y=g['mean'].to_numpy();e=g['std'].to_numpy()
            # 均值线为视觉主体；误差条单独降透明度，保留真实统计范围。
            ax.errorbar(x,y,yerr=e,fmt='none',ecolor=color,elinewidth=.8,capsize=2.2,capthick=.8,alpha=.55,zorder=2)
            ax.plot(x,y,color=color,linestyle=style,linewidth=1.65,marker=marker,markersize=4.8,
                    markerfacecolor='white',markeredgecolor=color,markeredgewidth=1.1,label=label,zorder=3)
            ax.annotate(f'{y[-1]:.3f}',xy=(50,y[-1]),xytext=(5,0),textcoords='offset points',fontsize=10.5,color=color,ha='left',va='center')
        ax.set(xticks=[10,20,30,40,50],yticks=[0,.25,.5,.75,1.0],xlim=(7,57),ylim=(-.02,1.02))
        ax.set_title({'T':'(a) 文本','A':'(b) 音频','V':'(c) 视觉'}[m],loc='left',fontsize=12,pad=10)
        ax.grid(axis='y',color='#E8EBED',linewidth=.55)
        ax.spines['left'].set_color('#B8C1C6');ax.spines['bottom'].set_color('#B8C1C6')
        ax.tick_params(axis='both',length=2.5,width=.6,pad=5)
        if panel_index>0:ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    axes[0].set_ylabel('Cosine distance',fontsize=12,labelpad=9)
    fig.supxlabel('缺失跨度（%）',x=.53,y=.035,fontsize=12)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.53,-.01),ncol=2,
               fontsize=11,columnspacing=3,handlelength=2.8,handletextpad=.7)
    save(fig,'05_completion')
    
    conf=plot_validation_confusion()

    write_report(summary, curves, stseed, rseed, conf)


def write_report(summary,curves,stseed,rseed,conf):
    s=pd.DataFrame(summary)
    lines=['# 对齐模型的验证集实验与六张分析图','',
        '使用已保存的对齐消融模型，在附件2验证集上补做完整输入及连续局部缺失推理，不重新训练、不使用test或无标签专项数据计算指标。对齐模型训练seed为2027、2028、2035；未对齐完整输入消融使用2026、2031、2033，均与对应测试图一致。未对齐四项指标从已保存的728条完整验证预测重新计算，缺失场景分析仍针对对齐模型；超参数分析另用seed=2026，两者分开呈现。','',
        '## 实验与统计口径','',
        '| 组别 | 对齐版 | 未对齐版 |','|---|---|---|',
        '| B0 | 保留全部现有模块 | 保留全部现有模块 |','| noC | 关闭观测条件补全 | 取消连续局部窗 |','| noQ | 关闭补全可靠性调制 | 关闭时间惩罚 |',
        '| noB | 关闭中性边界损失 | 关闭中性边界损失 |','| noBeta | 关闭类别频数平衡 | 关闭类别频数平衡 |','',
        '各组均保留模态门控；当前实验并非六组增量消融，不能据此单独证明门控、一致性损失或显式重构损失的贡献。','',
        '验证集共728个唯一样本。完整输入指未额外注入人工缺失，仍保留数据本身的有效观测掩码。局部缺失沿有效时间轴遮去连续区间，不表示整个模态消失。各组共享缺失掩码：随机位置mask seed为5026、5027、5028，起始/中间/末尾位置使用固定场景。','',
        '场景共78个：T/A/V单模态在10%、20%、30%、40%、50%跨度下随机连续缺失；T、A、V、TA、TV、AV、TAV七种集合在30%跨度下比较起始、中间、末尾和随机位置。随机掩码先在训练seed内部平均，模型间统计再取三个训练seed的均值和样本标准差（ddof=1）；三个mask seed不是三个独立训练重复。','',
        '推理统一使用float32、最高float32矩阵计算精度及batch_size=64，完整预测与历史验证摘要逐项核对；若旧推理设置下的临界样本类别与本次不同，单独记录差异，本报告统一使用本次推理结果。各模型在验证集上选过检查点，因此这些图用于验证集上的选模、鲁棒性和机制诊断，不等同于独立测试性能。','',
        '分类报告Accuracy、Macro-F1；回归报告MAE、Pearson。MAE及Pearson使用当前输出协议的极性一致性投影强度，稳定性图另用原始回归输出，二者不混用。','',
        '## 主图1：完整验证输入的组件消融','',
        '![完整验证输入消融](valid_figures/01_full_ablation.png)','',
        '**图类型与坐标：** 四面板点区间图。横轴是五个实际模型组，纵轴依次为Accuracy、Macro-F1、MAE、Pearson r。','',
        '**图注：** 每个大点为三个训练seed的均值，浅色小点为各seed实测值，误差棒为±1个样本标准差，上方标注均值。每次推理均使用同一验证集的728个样本；本图不额外注入缺失。Accuracy、Macro-F1和Pearson越高越好，MAE越低越好。','',
        '**论证目标：** 检查模块拆除如何影响常规验证表现，并同时观察分类与强度回归的取舍。','',
        '| 组别 | Accuracy (%) | Macro-F1 (%) | MAE | Pearson r |','|---|---:|---:|---:|---:|']
    for variant in VAR:
        vals=[]
        for key,scale in [('accuracy',100),('macro_f1',100),('mae',1),('pearson',1)]:
            row=s[(s.layout=='aligned')&(s.variant==variant)&(s.metric==key)].iloc[0]
            vals.append(f'{row["mean"]*scale:.3f} ± {row["std"]*scale:.3f}')
        lines.append('| '+GROUP_NAMES['aligned'][variant]+' | '+' | '.join(vals)+' |')
    lines += ['', '未对齐完整输入消融（同样为2×2四指标图）：', '',
              '| 组别 | Accuracy (%) | Macro-F1 (%) | MAE | Pearson r |','|---|---:|---:|---:|---:|']
    for variant in VAR:
        vals=[]
        for key,scale in [('accuracy',100),('macro_f1',100),('mae',1),('pearson',1)]:
            row=s[(s.layout=='unaligned')&(s.variant==variant)&(s.metric==key)].iloc[0]
            vals.append(f'{row["mean"]*scale:.3f} ± {row["std"]*scale:.3f}')
        lines.append('| '+GROUP_NAMES['unaligned'][variant]+' | '+' | '.join(vals)+' |')
    best=s[(s.layout=='aligned')&(s.metric=='accuracy')].sort_values('mean',ascending=False).iloc[0]
    lines += ['',f'**实际结果：** 完整验证Accuracy均值最高的是{GROUP_NAMES["aligned"][best.variant]}（{best["mean"]*100:.2f}%）。其他指标见表；不因B0是完整模型就预设它在每个指标上都最好。','',
        '## 主图2：连续缺失时长与性能退化','',
        '![缺失时长](valid_figures/02_duration.png)','',
        '**图类型与坐标：** 两行五列折线小图，每列一个模型，上行为Accuracy、下行为投影强度MAE，横轴为人工连续缺失跨度10%—50%。','',
        '**图注：** 仅比较T/A/V单模态随机连续缺失；每个训练seed先平均三个mask seed，再等权平均三个缺失模态，最后对三个训练seed取均值。彩色线是当前模型，灰色虚线是相同场景下的B0参照。不给组合模态补造未评估的跨时长网格；图中不使用阴影。','',
        '**论证目标：** 比较连续缺失加长时的性能变化，并判断哪些模块有助于减缓退化。曲线波动按实测保留，不强制平滑或单调。','']
    b=curves[curves.variant=='B0'].groupby('rate')[['accuracy','mae']].mean()
    lines += [f'**实际结果：** B0从10%到50%缺失的Accuracy为{b.loc[.1,"accuracy"]*100:.2f}%→{b.loc[.5,"accuracy"]*100:.2f}%，MAE为{b.loc[.1,"mae"]:.3f}→{b.loc[.5,"mae"]:.3f}。','',
        '## 主图3：缺失模态与缺失位置','',
        '![缺失类型和位置](valid_figures/03_type_position.png)','',
        '**图类型与坐标：** 五模型热图。横轴为起始、中间、末尾、随机位置，纵轴为七种缺失模态集合，颜色和格内数字表示Accuracy。','',
        '**图注：** 人工缺失跨度固定30%；T、A、V分别表示文本、音频、视觉，组合表示相应模态在匹配的连续时段共同缺失。随机位置先平均三个mask seed，各单元格再平均三个训练seed。所有子图共用色标，格内数字为百分比。','',
        '**论证目标：** 分离缺失类型和位置的影响，识别最敏感场景；不把位置效应与缺失时长混为一谈。','',
        '## 辅助图4：完整—缺失预测稳定性','',
        '![预测稳定性](valid_figures/04_stability.png)','',
        '**图类型与坐标：** 横向三面板点区间图，横轴为模型组，纵轴分别为类别翻转率、原始强度绝对变化及JS散度。','',
        '**图注：** 固定30%缺失，每个样本与同一模型在完整输入下的预测比较。随机掩码先平均，再等权平均位置及七种模态集合；大点和误差棒分别表示三个训练seed的均值与样本标准差，并标出均值。类别翻转率比较argmax类别；强度变化为原始回归输出绝对差；JS比较完整与缺失的三类概率分布。三者均越低越稳定。','',
        '**论证目标：** 观察预测对局部信息扰动的敏感程度。稳定不等于正确，必须结合主图1—3的Accuracy、F1和回归误差解释，不能把恒定错误预测称为鲁棒。','',
        '## 辅助图5：人工遮蔽槽的隐空间补全诊断','',
        '![补全质量](valid_figures/05_completion.png)','',
        '**图类型与坐标：** 横向文本、音频、视觉三面板折线图；横轴为连续缺失跨度，纵轴为Cosine distance。','',
        '**图注：** 仅使用B0与对应单模态随机连续缺失。仅在原本有效观测、随后被人工遮蔽的槽位上，以完整输入时该检查点产生的投影表示为参照，计算1−cosine similarity；比较学习补全与同模态可见槽均值填充。先平均mask seed，线与误差棒分别为三个训练seed的均值和样本标准差。三幅图共用纵轴，50%处标出均值。参照是模型隐表示，并非原始信号真值；此图不能证明恢复了原始音视频。','',
        '**论证目标：** 检验现有补全是否在隐表示距离上优于简单均值填充，为信息补偿机制提供诊断，而非预设补全有效。','']
    means=rseed.groupby(['target_modality','rate','kind']).cosine_distance.mean().unstack('kind')
    count=int((means.reconstructed<means.observed_mean).sum())
    lines += [f'**实际结果：** 在三模态×五种跨度的15个比较点中，学习补全的平均Cosine distance低于可见槽均值填充的点数为 **{count}/15**。应按此结果判断补全质量，不把分类增益直接解释为重构更准确。','',
        '## 辅助图6：完整验证集混淆矩阵','',
        '![验证混淆矩阵](valid_figures/06_confusion.png)','',
        '**图类型与坐标：** 行归一化混淆矩阵；横轴为预测类别，纵轴为真实类别，顺序均为负向、中性、正向。','',
        '**图注：** 左图为对齐完整模型，训练seed为2027、2028、2035；右图为未对齐完整模型，训练seed为2026、2031、2033。每次使用同一批728条完整验证样本，分别按真实类别行归一化，再对三个seed逐单元格取均值。颜色与数值表示该真实类别样本流向各预测类别的比例，单位为%；对角线对应各类Recall，不能直接解释为Precision。','',
        '**论证目标：** 定位总体指标掩盖的类别混淆，特别关注中性被判为正负、弱极性被判为中性的方向性错误。','',
        f'**实际结果：** 对齐模型中性类平均Recall为{conf["aligned"][1][1]:.2f}%，误判为负向和正向分别为{conf["aligned"][1][0]:.2f}%和{conf["aligned"][1][2]:.2f}%；未对齐模型中性类平均Recall为{conf["unaligned"][1][1]:.2f}%，误判为负向和正向分别为{conf["unaligned"][1][0]:.2f}%和{conf["unaligned"][1][2]:.2f}%。','',
        '## 图表之间的关系与复现','',
        '主图1建立完整输入参照；主图2、3分别拆解缺失时长、缺失类型和位置；辅助图4解释输出变化；辅助图5检查隐空间补偿；辅助图6定位具体类别错误。六图共同展示性能、鲁棒性和机制诊断，避免用单一指标代替全部结论。','',
        '```bash','conda run -n CPMCM python "problem2_retrain_v2 copy/验证集六图实验.py"','```','',
        '完整入口仅推理冻结模型，完成的检查点分析可复用；加`--plots-only`仅重画六图。数据在`valid_experiments/`，图在`valid_figures/`，只保存PNG和PDF；中文宋体，英文及数字Times New Roman。场景CSV保留逐样本预测及人工缺失记录。','']
    audit_file=OUT/'historical_validation_comparison.csv'
    if audit_file.exists():
        differences=pd.read_csv(audit_file).query('changed_predictions>0')
        if not differences.empty:
            details='；'.join(f'{r.variant}（seed={int(r.seed)}）：{int(r.changed_predictions)}个样本' for _,r in differences.iterrows())
            lines += ['','数值核对记录：与历史完整验证预测相比，'+details+'的分类发生变化。本文统一使用本次推理的完整及缺失预测，不混用旧摘要指标；差异详见`valid_experiments/historical_validation_comparison.csv`，不解释为新模型带来的提升。','']
    (HERE/'验证集实验结果与图表.md').write_text(update_report('\n'.join(lines),'valid_figures'))
    print('完成验证集八张实验图与说明文档。')

if __name__=='__main__':report()
