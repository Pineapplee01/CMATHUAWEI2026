"""从现有逐seed结果重画论文图，并生成含相对图片链接的Markdown；不训练模型。"""
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
FIG=HERE/'figures';FIG.mkdir(exist_ok=True)
TABLE=HERE/'plot_data';TABLE.mkdir(exist_ok=True)
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

sources=[]

def load(layout,name):
    prefix='f1_gate' if layout=='aligned' else 'softalign_gate'
    frames=[]
    for suffix in ['five_seeds','s2031_2035']:
        folder=ROOT/f'problem2_retrain_v2/results/{layout}/ablations/{prefix}_component_ablation_{suffix}'
        f=folder/name
        if not f.exists():continue
        df=pd.read_csv(f);df=df[df.seed.isin(SEEDS[layout])].copy()
        df['source_folder']=str(folder.relative_to(ROOT))
        frames.append(df)
        sources.append({'path':str(f.relative_to(ROOT)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
    if not frames:return pd.DataFrame()
    result=pd.concat(frames,ignore_index=True)
    result.to_csv(TABLE/f'{layout}_{name}',index=False)
    return result

metrics={x:load(x,'metrics.csv') for x in SEEDS}
stable=load('aligned','stability.csv');rec=load('aligned','reconstruction.csv')
full={x:df[df.rate==0].copy() for x,df in metrics.items()}
for layout,df in full.items():
    assert len(df)==15 and not df.duplicated(['variant','seed']).any()
    assert set(df.seed)==set(SEEDS[layout]) and set(df.variant)==set(VAR)
    for _,r in df.iterrows():
        f=ROOT/r.source_folder/'runs'/f'{r.variant}_seed{r.seed}'/'test_full.csv'
        pred=pd.read_csv(f);assert len(pred)==727 and pred.id.nunique()==727
        assert abs((pred['class']==pred.true_class).mean()-r.accuracy)<1e-10
        sources.append({'path':str(f.relative_to(ROOT)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})

def save(fig,name):
    save_clean(fig,FIG,name)

def per_seed(df,keys,values):
    # 随机掩码先平均，再平均位置，最后平均缺失集合；保留seed为独立重复层级。
    first=df.groupby(['variant','seed','modalities','rate','position'],as_index=False)[values].mean()
    return first.groupby(keys,as_index=False)[values].mean()

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
    groups={kind:rseed.query('target_modality==@m and kind==@kind').groupby('rate').cosine_distance.agg(['mean','std'])
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

fig,axes=plt.subplots(1,2,figsize=(9.2,4.8),layout='constrained')
conf={}
for ax,layout in zip(axes,SEEDS):
    mats=[]
    for _,r in full[layout].query("variant=='B0'").iterrows():
        p=pd.read_csv(ROOT/r.source_folder/'runs'/f'B0_seed{r.seed}'/'test_full.csv')
        mats.append(confusion_matrix(p.true_class,p['class'],labels=[0,1,2],normalize='true'))
    cm=np.mean(mats,axis=0)*100;conf[layout]=cm.tolist()
    im=ax.imshow(cm,vmin=0,vmax=100,cmap=heat_cmap)
    for i in range(3):
        for j in range(3):ax.text(j,i,f'{cm[i,j]:.1f}%',ha='center',va='center',color='white' if cm[i,j]>58 else '#243443',fontsize=14)
    ax.set(title='(a) 对齐模型' if layout=='aligned' else '(b) 未对齐模型',xlabel='预测类别',ylabel='真实类别',xticks=range(3),yticks=range(3),xticklabels=['负向','中性','正向'],yticklabels=['负向','中性','正向'])
    ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,3,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=2);ax.tick_params(which='both',length=0)
    for spine in ax.spines.values():spine.set_visible(False)
fig.colorbar(im,ax=axes,label='Row-normalized proportion (%)',shrink=.82,aspect=25,pad=.025)
save(fig,'06_confusion')

s=pd.DataFrame(summary)
def val(layout,v,metric):return s.query('layout==@layout and variant==@v and metric==@metric').iloc[0]
lines=['# 基于现有代码与实际实验的消融分析','','本报告只使用已经完成的实验，重新生成下列六张图，不复用旧图片，不把旧实验改名为尚未实现的六组增量实验。',
'','## 现有模型与真实实验组','',
'| 组别 | 对齐版含义 | 未对齐版含义 |','|---|---|---|',
'| B0 | 补全＋可靠性＋模态门控＋边界损失＋类别平衡 | 连续单调局部窗＋窗内软注意力＋时间惩罚＋模态门控＋边界损失＋类别平衡 |',
'| noC | 关闭观测条件补全 | 取消连续局部窗，使用全局软注意力 |',
'| noQ | 关闭补全可靠性调制 | 关闭时间惩罚 |',
'| noB | 关闭中性边界损失 | 关闭中性边界损失 |',
'| noBeta | 关闭类别频数平衡 | 关闭类别频数平衡 |',
'', '**同名noC/noQ在两种布局中的含义不同，不能横向解释成相同模块。** 当前没有独立一致性约束组、拼接缺失基线或显式重构损失组；所有变体都保留模态门控，所以现有消融不能单独证明门控有效。',
'', '## 数据与统计口径','',
'对齐seed：**2027、2028、2035**；未对齐seed：**2026、2031、2033**。来自两个原始实验批次的逐seed记录，而非旧的筛选汇总表。五个变体在各自布局中使用同一组三个seed。完整输入共30份逐样本预测文件，均核对727个唯一ID，准确率与metrics.csv一致。',
'现存历史报告明确记录过依据测试结果重选seed；本报告按用户指定集合报告，属于选定子集的描述性统计，不能作为未筛选重复实验的显著性证据，也不与此前五seed标准差直接比较。',
'均值与误差条均在训练seed层级计算，标准差ddof=1（n=3，分母2）；随机位置的三个mask seed（5026/5027/5028）先在每个训练seed内平均，不当作额外训练重复。未对齐五组现有记录只有完整输入，因此连续缺失相关图仅绘对齐结果。MAE来自现存预测文件的极性投影后强度；稳定性图明确使用原始回归输出。',
'', '## 主图1：完整输入性能与组件拆除','', '![图1 完整输入五组消融](figures/01_full_ablation.png)',
'', '**类型/坐标：** 两行三列竖向点区间图；行分别对齐/未对齐，横轴为五个实际变体，纵轴分别为Accuracy、Macro-F1、Projected intensity MAE。各面板独立确定纵轴范围。',
'', '**图注：** 图1 完整测试输入下的组件拆除实验。B0为各布局的完整模型，noC/noQ/noB/noBeta的具体含义见上表。大点为三个指定训练seed的均值，误差条为±1样本标准差，浅色小点为单seed成绩；上方数值标注对应均值。横轴采用简写：“去边界”为去边界损失，“去平衡”为去类别平衡，“去时间项”为去时间惩罚；完整组名见表。所有组测试样本数均为727。ACC与Macro-F1越高越好，MAE越低越好。对齐和未对齐的seed集合不同，因此两行不构成同seed配对比较。该图检验组件对完整输入任务性能的贡献，不单独证明缺失鲁棒性。',
'', '| 布局 | 组别 | ACC均值±SD | Macro-F1均值±SD | MAE均值±SD | Pearson均值±SD |','|---|---|---:|---:|---:|---:|']
for layout in SEEDS:
    for v in VAR:
        a,f,m,r=[val(layout,v,k) for k in ['accuracy','macro_f1','mae','pearson']]
        lines.append(f'| {layout} | {v} | {a["mean"]*100:.2f}% ± {a["std"]*100:.2f}个百分点 | {f["mean"]*100:.2f}% ± {f["std"]*100:.2f}个百分点 | {m["mean"]:.4f} ± {m["std"]:.4f} | {r["mean"]:.4f} ± {r["std"]:.4f} |')
lines+=['','**论证目标与实际结论：** 指定seed子集内B0的完整输入平均ACC高于各拆除组；差距大小见表。不能据此断言未筛选种子总体同样成立，也不能把很小的差距称为统计显著。',
'', '## 主图2：连续缺失时长与预测性能','', '![图2 缺失时长](figures/02_duration.png)',
'', '**类型/坐标：** 两行五列折线图；每列对应一个模型，上行为Accuracy，下行为Projected intensity MAE。同一行共用纵轴范围，横轴为10/20/30/40/50%的连续缺失跨度。各拆除模型只与完整模型参照比较，不再叠加五组曲线。',
'', '**图注：** 图2 对齐模型在单模态随机连续缺失下的性能。每个seed、每个跨度先平均三个随机掩码重复，再对T/A/V三种单模态缺失等权平均；彩色实线为当前模型三个训练seed的均值，灰色虚线和空心圆为完整模型B0的均值参照；B0列不重复画参照线。本图不绘阴影或误差条，以突出均值趋势；这不表示训练波动为零，逐seed结果保存在plot_data/duration_by_seed.csv。两线重合表示实际均值接近，未通过平移数据制造间距。各变体使用相同场景定义。横轴为标称有效时间跨度，受取整及保留观测规则影响，不等同于精确删除观测比例。组合模态在现有记录中只有30%跨度，因此不将其混入本图，也不外推70%结果。',
'', '**论证目标：** 检验组件的贡献是否随连续缺失时长改变，并同时观察绝对性能和退化趋势；不能只依据曲线下降较慢判定某模型更好。',
'', '## 主图3：缺失模态与位置的敏感性','', '![图3 模态与位置](figures/03_type_position.png)',
'', '**类型/坐标：** 五张绝对性能热力图；横轴为起始/中间/末尾/随机，纵轴为缺失模态组合。所有面板的颜色与数字均表示Accuracy（%）。',
'', '**图注：** 图3 对齐模型在30%连续缺失下的场景级准确率。T/A/V分别为文本、音频、视觉，多模态组合使用同一相对时间段的缺失。随机位置先平均三个掩码重复，各格再平均三个训练seed。五张热力图共享同一准确率色标，可直接比较相同场景中的模型表现；颜色不代表统计显著性。该图仅包含现有实测场景，不含完整输入参考；平均值不表示每个seed均有相同排序。',
'', '**论证目标：** 找到对模型影响最大的模态组合与缺失位置，并检查完整模型是否在所有场景都占优；负面场景也完整保留。',
'', '## 辅助图4：完整—缺失输出稳定性','', '![图4 输出稳定性](figures/04_stability.png)',
'', '**类型/坐标：** 三面板竖向点区间图，横轴为实际变体，纵轴分别为Class flip rate、Raw intensity absolute shift、JS divergence；误差条为±1样本标准差，上方彩色数值为均值（翻转率保留两位小数、强度漂移三位、JS散度四位），与主图1的标注样式一致。',
'', '**图注：** 图4 对齐模型在30%缺失场景下的输出变化。每个模型以自身完整输入预测为参照，比较同一样本缺失输入的类别、原始回归值和分类分布；不是与另一个模型的输出比较。每个seed先平均随机掩码重复，再对四种位置及七种缺失组合等权平均；大点和误差条为三个seed的均值±样本标准差。三项均越低表示输出变化越小，但恒定错误预测也可能稳定，因此须结合图1–3的准确率判断。本代码未训练一致性约束，这张图是行为诊断，不是一致性损失的消融证明。',
'', '## 辅助图5：现有补全表示的质量','', '![图5 补全质量](figures/05_completion.png)',
'', '**类型/坐标：** 文本、音频、视觉横向并列的三个折线图；横轴为单模态随机连续缺失跨度，纵轴为Cosine distance。实线圆点表示学习补全，虚线菱形表示可见槽均值填充。三幅面板共用相同纵轴，上方为统一图例；50%缺失处标出均值。',
'', '**图注：** 图5 对齐B0的学习补全表示与可见槽均值填充的离线比较。评分只覆盖原先可观测而被人工删除的槽，排除padding及原始无效槽；参照为同一B0模型在完整输入下的投影特征。简单基线对每条样本当前仍可见的同模态投影特征求均值，并填入缺失位置。目标完整特征只用于离线评分，不送入缺失推理。每个点先平均三个mask seed，再报告三个训练seed的均值；圆点为学习补全，菱形为可见槽均值填充，彩色细误差条表示±1样本标准差。同一方法在不同缺失跨度的均值用折线连接；本图不绘阴影。50%缺失处的数字为该位置的均值；三种模态共享纵轴范围，余弦距离越低越好。本指标来自现存reconstruction.csv，是中间补全输出的相似度，不等同于最终可靠性调制后的表示，也不等同于显式重构监督损失。',
'', '**论证目标：** 检查现有补全是否优于简单可见均值；即使任务ACC有所提高，也不预设补全的余弦质量更好。',
'', '## 辅助图6：完整模型的类别错误结构','', '![图6 混淆矩阵](figures/06_confusion.png)',
'', '**类型/坐标：** 两张混淆矩阵，横轴预测类别、纵轴真实类别；两图共享0–100%色标。',
'', '**图注：** 图6 对齐及未对齐B0在完整test上的分类混淆矩阵。各seed分别计算按真实类别行归一化的矩阵，再对该布局的三个指定seed求平均；每行总和100%。对角线表示类别召回率，非对角线表示错误去向。类别顺序负、中、正；两布局使用同一测试样本划分，但seed不同。此图使用完整输入结果，不能解释为缺失测试的中性召回率。',
'', '**论证目标：** 查看总体准确率背后是否存在中性识别不足，避免用ACC掩盖类别差异。',
'', '## 图组关系与复现','',
'图1给出当前真实消融的整体性能，图2和图3分别分解缺失时长、模态和位置；图4解释输出变化，图5检验补全表示是否真的相似，图6揭示类别错误结构。这套图不声称验证当前代码没有实现的模块。',
'运行：`conda run -n CPMCM python "problem2_retrain_v2 copy/生成现有实验图表.py"`。无需训练、无需GPU，重新从原始结果生成PNG/PDF、统计表和本报告。',
'绘图输入副本与seed级聚合在`plot_data/`；源文件SHA与指定seed在`plot_data/provenance.json`，便于追溯。']
# Add measured diagnostics, avoiding predetermined conclusions.
for m in ['T','A','V']:
    g=rseed.query('target_modality==@m').groupby('kind').cosine_distance.mean()
    lines.append(f'\n补全诊断（{m}，五个跨度等权）：学习补全余弦距离{g["reconstructed"]:.4f}，可见均值{g["observed_mean"]:.4f}；较低者为{ "学习补全" if g["reconstructed"]<g["observed_mean"] else "可见均值" }。')
for layout in SEEDS:lines.append(f'\n{layout}完整输入中性召回率：{conf[layout][1][1]:.2f}%。')
lines.append('\n验证集上的选模、参数与阈值分析见[补充报告](验证集选模与参数分析.md)。')
(HERE/'现有实验结果与图表.md').write_text(update_report('\n'.join(lines)+'\n','figures'))
(TABLE/'provenance.json').write_text(json.dumps({'seeds':SEEDS,'std_ddof':1,'sources':sources,'figures':8,'verified_full_prediction_files':30},ensure_ascii=False,indent=2))
print(s.to_string(index=False))
print('Generated eight PNG/PDF figures and embedded Markdown report.')
