"""测试/验证共享的中文命名、四指标消融及居中时长图布局。"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

NAMES={'01_full_ablation':'对齐模型完整输入组件消融比较',
       '01_unaligned_ablation':'未对齐模型完整输入组件消融比较',
       '02_duration_accuracy':'连续缺失时长对准确率的影响',
       '02_duration_mae':'连续缺失时长对平均绝对误差的影响',
       '03_type_position':'缺失模态与位置的性能敏感性',
       '04_stability':'局部缺失下的预测稳定性','05_completion':'隐空间补全质量对比',
       '06_confusion':'完整模型的类别错误分布',
       '09_valid_parameter_endpoints':'类别平衡与边界损失的端点对照',
       '10_valid_threshold':'中性判定边界分析'}
COLORS=['#BD5148','#44789A','#8574A4','#528B78','#C39A4B']
MARKERS=['o','s','D','^','v'];VAR=['B0','noC','noQ','noB','noBeta']



def black_text(fig):
    """统一所有图中文字为纯黑，曲线和数据点保持原配色。"""
    from matplotlib.text import Text
    for ax in fig.axes:
        ax.tick_params(axis='both', which='both', labelcolor='black')
    # 实际绘制一次，使刻度与科学计数法偏移文字全部创建。
    fig.canvas.draw()
    for item in fig.findobj(match=Text):
        item.set_color('black')
        item.set_alpha(1.0)


def save_clean(fig,folder,name):
    if fig._suptitle is not None:fig._suptitle.remove();fig._suptitle=None
    # 不加整图总标题；每个子图保留中文标题，统一放在面板上方。
    panel_names={
        '03_type_position':['完整模型','无补全模型','无可靠性模型','无边界损失模型','无类别平衡模型'],
        '04_stability':['类别翻转率','情感强度变化','预测分布差异'],
        '05_completion':['文本模态','音频模态','视觉模态'],
        '06_confusion':['对齐模型','未对齐模型'],
        '09_valid_parameter_endpoints':['对齐：类别平衡强度','对齐：边界损失权重','未对齐：类别平衡强度','未对齐：边界损失权重'],
        '10_valid_threshold':['对齐：验证选模分数','对齐：分类性能','对齐：中性识别','未对齐：验证选模分数','未对齐：分类性能','未对齐：中性识别'],
    }
    if name in panel_names:
        panels=[ax for ax in fig.axes if ax.axison and not hasattr(ax,'_colorbar')]
        for index,(ax,title) in enumerate(zip(panels,panel_names[name])):
            ax.set_title('',loc='left');ax.set_title('',loc='right')
            ax.set_title(f'({chr(97+index)}) {title}',fontsize=12,pad=12)
    black_text(fig)
    for ext in ('png','pdf'):fig.savefig(folder/f'{NAMES.get(name,name)}.{ext}',bbox_inches='tight',pad_inches=.16)
    plt.close(fig)


def ablation(data,layout,folder):
    labels=['完整','去补全','去可靠性','去边界','去平衡'] if layout=='aligned' else ['完整','去局部窗','去时间项','去边界','去平衡']
    fig,axes=plt.subplots(2,2,figsize=(10.4,8.1),layout='constrained');rows=[]
    for col,(ax,(metric,label,scale)) in enumerate(zip(axes.flat,[('accuracy','Accuracy (%)',100),('macro_f1','Macro-F1 (%)',100),('mae','MAE',1),('pearson','Pearson r',1)])):
        groups=[data[data.variant==v][metric].to_numpy()*scale for v in VAR]
        lo=min(min(x.min(),x.mean()-x.std(ddof=1)) for x in groups);hi=max(max(x.max(),x.mean()+x.std(ddof=1)) for x in groups);span=max(hi-lo,1e-6)
        for i,(v,x,c) in enumerate(zip(VAR,groups,COLORS)):
            ax.errorbar(i,x.mean(),yerr=x.std(ddof=1),fmt='none',ecolor=c,capsize=5,elinewidth=1.5,zorder=2)
            ax.scatter(i+np.linspace(-.12,.12,len(x)),x,s=18,color=c,alpha=.3)
            ax.scatter(i,x.mean(),s=70,marker=MARKERS[i],color=c,edgecolors='white',zorder=4)
            ax.text(i,max(x.max(),x.mean()+x.std(ddof=1))+.06*span,f'{x.mean():.2f}' if scale==100 else f'{x.mean():.3f}',ha='center',va='bottom',color=c,fontsize=12)
            rows.append({'layout':layout,'variant':v,'metric':metric,'mean':x.mean()/scale,'std':x.std(ddof=1)/scale})
        letter=('abcd' if layout=='aligned' else 'efgh')[col]
        ax.set(xlim=(-.5,4.5),ylim=(lo-.13*span,hi+.27*span),xticks=range(5),xticklabels=labels,ylabel=label)
        chinese=['准确率','宏平均F1值','平均绝对误差','皮尔逊相关系数'][col]
        ax.set_title(f'({letter}) {chinese}',fontsize=12,pad=12)
        ax.grid(axis='y',alpha=.45);ax.tick_params(axis='x',length=0,pad=7)
    save_clean(fig,folder,'01_full_ablation' if layout=='aligned' else '01_unaligned_ablation')
    return rows


def duration(curves,folder):
    labels=['完整模型','无补全','无可靠性','无边界损失','无类别平衡']
    for metric,ylabel,scale in [('accuracy','Accuracy (%)',100),('mae','MAE',1)]:
        fig=plt.figure(figsize=(12,7.6))
        positions=[(.075,.61),(.375,.61),(.675,.61),(.225,.145),(.525,.145)]
        axes=[fig.add_axes([left,bottom,.25,.32]) for left,bottom in positions]
        base=curves[curves.variant=='B0'].groupby('rate')[metric].mean()*scale
        ys=[]
        for i,(ax,v,c) in enumerate(zip(axes,VAR,COLORS)):
            means=curves[curves.variant==v].groupby('rate')[metric].mean()*scale;ys.extend(means.to_list());x=means.index.to_numpy()*100
            if v!='B0':ax.plot(x,base,color='#747F87',ls='--',lw=1.5,marker='o',ms=7,mfc='white',zorder=2)
            ax.plot(x,means,color=c,lw=1.9,marker=MARKERS[i],ms=4,zorder=3)
            ax.set(xlabel='缺失跨度（%）',ylabel=ylabel,xticks=[10,20,30,40,50]);ax.grid(axis='y',alpha=.5)
            ax.set_title(f'({chr(97+i)}) {labels[i]}',fontsize=12,pad=12)
        span=max(max(ys)-min(ys),1e-6)
        for ax in axes:ax.set_ylim(min(ys)-.22*span,max(ys)+.1*span)
        fig.legend(handles=[Line2D([],[],color='#44789A',marker='s',label='当前模型'),Line2D([],[],color='#747F87',ls='--',marker='o',mfc='white',label='完整模型参照')],loc='lower center',bbox_to_anchor=(.5,.005),ncol=2)
        save_clean(fig,folder,'02_duration_'+metric)


def update_report(text,folder):
    for old,new in NAMES.items():text=text.replace(f'{folder}/{old}.png',f'{folder}/{new}.png')
    old=f'({folder}/对齐模型完整输入组件消融比较.png)'
    if old in text and f'({folder}/未对齐模型完整输入组件消融比较.png)' not in text:
        text=text.replace(old,old+f'\n\n![未对齐完整输入组件消融]({folder}/未对齐模型完整输入组件消融比较.png)')
    text=text.replace(f'({folder}/02_duration.png)',f'({folder}/连续缺失时长对准确率的影响.png)\n\n![连续缺失时长对MAE的影响]({folder}/连续缺失时长对平均绝对误差的影响.png)')
    text=text.replace('两行三列竖向点区间图；行分别对齐/未对齐，横轴为五个实际变体，纵轴分别为Accuracy、Macro-F1、Projected intensity MAE。','对齐、未对齐分别绘制一张2×2点区间图，横轴为五个实际变体，纵轴分别为Accuracy、Macro-F1、MAE和Pearson r。')
    text=text.replace('四面板点区间图。','对齐、未对齐各一张2×2四面板点区间图。')
    text=text.replace('两行五列折线小图，每列一个模型，上行为Accuracy、下行为投影强度MAE，横轴为人工连续缺失跨度10%—50%。','Accuracy与MAE分别绘制一张折线图；每张第一行三个模型、第二行两个模型居中排列，横轴为人工连续缺失跨度10%—50%。')
    text=text.replace('十个小面板，列对应五个模型，行对应Accuracy/MAE；','Accuracy和MAE分为两张图，每张按上三下二居中排列；')
    text=text.replace('三幅面板共用相同纵轴，上方为统一图例','三幅面板共用相同纵轴，下方为统一图例')
    text=text.replace('两行五列折线图；每列对应一个模型，上行为Accuracy，下行为Projected intensity MAE。同一行共用纵轴范围，','Accuracy、MAE各一张折线图，每张第一行三面板、第二行两面板居中，五个面板大小相同、共用纵轴范围；')
    text=text.replace('因此两行不构成同seed配对比较','因此两张图不构成同seed配对比较').replace('六张图','八张图').replace('六张分析图','六类实验分析图')
    return text
