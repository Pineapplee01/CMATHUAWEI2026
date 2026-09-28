"""六张论文图由实验记录自动生成PNG/PDF；无数据时拒绝伪造。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from shared.plotting import pyplot

ORDER=['B0','Bm','G','C','R','F']
COLORS={'B0':'#777777','Bm':'#0072B2','G':'#E69F00','C':'#009E73','R':'#CC79A7','F':'#D55E00'}
CAPTIONS=[
 ('01_ablation','连续模态缺失下的模块消融','点区间图；横轴实验组，纵轴分别为ACC、Macro-F1、强度MAE。',
  'B0/Bm共用权重，仅评估输入不同。缺失组固定30%跨度，对随机掩码重复先平均，再对七种模态组合等权平均；点为训练seed，区间为跨seed均值±标准差。B0是完整输入参考，不是理论上限。',
  '检验缺失退化、单模块独立增益与组合总体收益，不能由六组直接推断模块协同。'),
 ('02_robustness','连续缺失跨度与预测鲁棒性','三列对应T/A/V缺失，上排Macro-F1，下排MAE；横轴连续缺失跨度比例。',
  '各组共享相同缺失区间。0%点为各模型完整输入性能；灰色参考线为B0。随机掩码重复在每个训练seed内先平均；阴影为跨seed标准差。跨度以有效槽轴定义，不虚构为秒。',
  '检验收益是否跨缺失严重程度持续存在，以及哪种模态缺失最敏感。'),
 ('03_location','缺失模态组合与位置的性能差异','热力图；行是七种模态组合，列是start/middle/end/random；显示Bm、F及F−Bm。',
  '固定主缺失比例。联合模态共用连续区间。随机位置重复先合并，所有位置等权；绝对性能共用色标，差值用以0为中心的色标，单位百分点。',
  '识别困难场景及局部退化，防止总体均值掩盖失败条件。'),
 ('04_stability','完整与缺失视图的预测稳定性','点区间图；横轴模型，纵轴JS散度、极性翻转率、原始强度变化。',
  '同模型、同样本配对比较完整与缺失视图。强度使用未投影原始输出。先合并掩码重复再对模态类型等权汇总。稳定性必须与准确性联合解释。',
  '检验一致性训练是否降低输出扰动，同时排除仅稳定地给出错误预测的解释。'),
 ('05_reconstruction','隐空间补偿质量与表示恢复','上排为T/A/V缺失跨度与余弦距离曲线，下排为中位误差样本的完整、缺失、重构隐态热力图。',
  '只评价人工遮挡且原本可观测的槽。完整参考来自同一模型同一层。观测均值补偿对照仅使用缺失视图重新编码的可见特征；不同模型坐标不逐维比较。样本按F首个seed的中位误差选择，非最佳案例；目标范数另存以检查坍塌。',
  '检验补偿是否比简单观测均值更接近完整表示；接近不自动等于任务性能改善。'),
 ('06_gating','动态门控响应与推理干预','上排为缺失边界对齐门控曲线，下排为动态、训练均值固定、样本间打乱门控的Macro-F1。',
  '上排分别显示T/A/V单模态缺失，区间[0,1]为缺失段；门控归一化到有效模态均值1，不是百分比贡献。下排固定TAV主缺失比例；固定门控从train估计，打乱保留同场景门控分布。这些为同权重推理诊断，不是额外训练组。',
  '检验门控是否对局部缺失响应，以及输入相关权重是否实际支持预测。')]


def curve(ax,frame,x,y,variant,linestyle='-',label=None):
    seedwise=frame.groupby(['seed',x],as_index=False)[y].mean()
    agg=seedwise.groupby(x)[y].agg(['mean','std']).sort_index()
    if agg.empty:return
    xs=agg.index.to_numpy();mean=agg['mean'].to_numpy();sd=agg['std'].to_numpy()
    ax.plot(xs,mean,linestyle=linestyle,marker='o' if linestyle=='-' else 's',markersize=3,color=COLORS[variant],label=label or variant)
    if np.isfinite(sd).all():ax.fill_between(xs,mean-sd,mean+sd,color=COLORS[variant],alpha=.12)


def dots(ax,frame,metric,order):
    for i,v in enumerate(order):
        values=frame.loc[frame.variant==v,metric].dropna().to_numpy()
        if not len(values):continue
        ax.scatter(i+np.linspace(-.08,.08,len(values)),values,s=15,alpha=.55,color=COLORS[v])
        sd=values.std(ddof=1) if len(values)>1 else None
        ax.errorbar(i,values.mean(),yerr=sd,fmt='o',color=COLORS[v],capsize=4,markersize=6)
    ax.set_xticks(range(len(order)),order);ax.grid(axis='y',alpha=.2)


def plot_all(root):
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text());spec=protocol['identity']['settings']
    required=['primary_by_seed','metrics','stability','reconstruction','gate_profile','gate_interventions']
    tables={name:pd.read_csv(root/f'{name}.csv') for name in required}
    plt=pyplot();plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    dest=root/'figures';dest.mkdir(exist_ok=True)
    def save(fig,index):
        if spec['smoke']:fig.text(.5,.005,'SMOKE TEST — NOT RESEARCH RESULTS',ha='center',color='red',fontsize=10)
        fig.savefig(dest/f'{CAPTIONS[index][0]}.png',dpi=300,bbox_inches='tight')
        fig.savefig(dest/f'{CAPTIONS[index][0]}.pdf',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.3))
    for ax,key,label in zip(axes,['accuracy','macro_f1','mae'],['Accuracy ↑','Macro-F1 ↑','Intensity MAE ↓']):
        dots(ax,tables['primary_by_seed'],key,ORDER);ax.set_ylabel(label)
    fig.suptitle('Module ablation under continuous missingness');fig.tight_layout();save(fig,0)
    scores=tables['metrics'];ref=tables['primary_by_seed'].query("variant=='B0'")
    fig,axes=plt.subplots(2,3,figsize=(11,6))
    for col,mod in enumerate('TAV'):
        for row,key in enumerate(['macro_f1','mae']):
            ax=axes[row,col]
            for variant in ORDER[1:]:
                frame=scores[(scores.variant==variant)&(scores.modalities==mod)&(scores.position=='random')]
                zero=scores[(scores.variant==variant)&(scores.rate==0)]
                curve(ax,pd.concat([frame,zero]),'rate',key,variant)
            ax.axhline(ref[key].mean(),color=COLORS['B0'],ls=':',label='B0 full input')
            ax.set(xlabel='Continuous span / valid span',ylabel=key,title=f'{mod} missing');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.tight_layout();save(fig,1)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    names=['T','A','V','TA','TV','AV','TAV'];positions=['start','middle','end','random']
    subset=scores[np.isclose(scores.rate,spec['main_rate'])]
    def heat(v):
        frame=subset[subset.variant==v].groupby(['seed','modalities','position']).macro_f1.mean().reset_index()
        return frame.groupby(['modalities','position']).macro_f1.mean().unstack().reindex(index=names,columns=positions).to_numpy()*100
    bm,full=heat('Bm'),heat('F');bound=max(1,float(np.nanmax(np.abs(full-bm))))
    for ax,array,title in zip(axes,[bm,full,full-bm],['Bm Macro-F1 (%)','F Macro-F1 (%)','F − Bm (pp)']):
        delta=title.startswith('F −')
        im=ax.imshow(array,cmap='RdBu_r' if delta else 'viridis',vmin=-bound if delta else 0,vmax=bound if delta else 100)
        for i in range(7):
            for j in range(4):ax.text(j,i,f'{array[i,j]:.1f}',ha='center',va='center',fontsize=8,
                                      color='black' if delta else 'white')
        ax.set(xticks=range(4),xticklabels=positions,yticks=range(7),yticklabels=names,title=title)
        fig.colorbar(im,ax=ax,fraction=.045)
    fig.tight_layout();save(fig,2)
    stable=tables['stability'];stable=stable[np.isclose(stable.rate,spec['main_rate'])&(stable.position=='random')]
    stable=stable.groupby(['variant','seed','modalities'],as_index=False)[['js','flip_rate','raw_intensity_shift']].mean()
    stable=stable.groupby(['variant','seed'],as_index=False)[['js','flip_rate','raw_intensity_shift']].mean()
    fig,axes=plt.subplots(1,3,figsize=(11,3.3))
    for ax,key in zip(axes,['js','flip_rate','raw_intensity_shift']):dots(ax,stable,key,ORDER[1:]);ax.set_ylabel(key+' ↓')
    fig.suptitle('Paired full/missing prediction stability');fig.tight_layout();save(fig,3)
    rec=tables['reconstruction'];fig,axes=plt.subplots(2,3,figsize=(12,6.5))
    for col,mod in enumerate('TAV'):
        ax=axes[0,col]
        for variant in ('R','F'):
            for kind,style in [('reconstructed','-'),('observed_mean','--')]:
                frame=rec[(rec.variant==variant)&(rec.modalities==mod)&(rec.position=='random')&(rec.kind==kind)]
                curve(ax,frame,'rate','cosine_distance',variant,style,f'{variant} {kind}')
        ax.set(xlabel='Continuous span / valid span',ylabel='Cosine distance ↓',title=f'{mod} compensation')
    axes[0,0].legend(fontsize=6)
    example=root/'runs'/f'F_seed{spec["seeds"][0]}'/'latent_example.npz'
    if not example.exists():raise ValueError('无可用的人工缺失重构示例，不能伪造热力图')
    with np.load(example) as e:
        valid=e['valid'] if 'valid' in e else np.ones(50,dtype=bool)
        stop=int(np.where(valid)[0].max())+1
        compensated=np.where(e['mask'][:,None],e['reconstructed'],e['observed'])
        arrays=[e['reference'],e['observed'],compensated]
        bound=max(.01,float(np.percentile(np.abs(np.concatenate([a[valid].ravel() for a in arrays])),99)))
        for ax,array,title in zip(axes[1],arrays,['Full reference','Missing input','Compensated representation']):
            display=np.ma.array(array[:stop].T,mask=np.broadcast_to(~valid[:stop],(array.shape[1],stop)))
            im=ax.imshow(display,aspect='auto',cmap='RdBu_r',vmin=-bound,vmax=bound)
            slots=np.where(e['mask'])[0]
            for x in (slots.min()-.5,slots.max()+.5):ax.axvline(x,color='black',ls='--',lw=.8)
            ax.set(xlabel='Aligned slot (padding excluded)',ylabel='Latent dimension',title=title)
        fig.colorbar(im,ax=axes[1,2],fraction=.045,label='Latent activation')
    fig.tight_layout();save(fig,4)
    fig=plt.figure(figsize=(12,6));grid=fig.add_gridspec(2,3)
    gates=tables['gate_profile']
    for col,mod in enumerate('TAV'):
        ax=fig.add_subplot(grid[0,col]);ax.axvspan(0,1,color='grey',alpha=.15)
        for variant in ('G','F'):
            for role,style in [('missing modality','-'),('other modality','--')]:
                frame=gates[(gates.variant==variant)&(gates.modalities==mod)&(gates.role==role)]
                curve(ax,frame,'relative_position','gate',variant,style,f'{variant}: {role}')
        ax.set(xlabel='Relative position (missing interval: 0–1)',ylabel='Gate (available mean = 1)',title=f'{mod} missing')
        if col==0:ax.legend(fontsize=6,handlelength=3)
    ax=fig.add_subplot(grid[1,:]);inter=tables['gate_interventions']
    labels=['dynamic','fixed_train_mean','shuffled']
    for j,v in enumerate(('G','F')):
        group=inter[inter.variant==v].groupby(['seed','intervention']).macro_f1.mean().reset_index()
        for i,label in enumerate(labels):
            vals=group.loc[group.intervention==label,'macro_f1'].to_numpy();x=i+(.12 if j else -.12)
            ax.scatter(x+np.linspace(-.025,.025,len(vals)),vals,color=COLORS[v],s=12,alpha=.5)
            ax.errorbar(x,vals.mean(),yerr=vals.std(ddof=1) if len(vals)>1 else None,
                        fmt='o',capsize=4,color=COLORS[v],label=v if i==0 else None)
    ax.set(xticks=range(3),xticklabels=labels,ylabel='Macro-F1 ↑',title='Inference interventions: same checkpoint and inputs')
    ax.legend();fig.tight_layout();save(fig,5)
    lines=['# 图表说明','','主图1–3，辅助机制图4–6。误差区间表示跨训练seed的标准差，不是95%置信区间；单seed不画误差区间。']
    for name,title,contents,caption,goal in CAPTIONS:
        caption=caption.replace('30%',f"{spec['main_rate']:.0%}")
        lines += ['',f'## {name}: {title}',f'图类型/内容：{contents}',f'图注：{caption}',f'论证目标：{goal}',f'![{title}](figures/{name}.png)']
    if spec['smoke']:lines.insert(2,'**本目录全部图仅为流程测试，不能用于论文实证结论。**')
    (root/'图表说明.md').write_text('\n\n'.join(lines)+'\n')
