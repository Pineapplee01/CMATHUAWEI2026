"""只读取valid记录并推理valid，补充选模、参数端点和中性阈值诊断。"""
from 论文图表样式 import NAMES, save_clean
import sys,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support
from problem2_retrain_v2.data import load_split,batch,masked_view,sha256
from problem2_retrain_v2.local_missingness import synchronized_view
from problem2_retrain_v2.training import restore

OUT=HERE/'validation_analysis';OUT.mkdir(exist_ok=True)
FIG=HERE/'figures'
SEEDS={'aligned':[2027,2028,2035],'unaligned':[2026,2031,2033]}
NAMES={'aligned':'对齐','unaligned':'未对齐'}
VAR=['B0','noC','noQ','noB','noBeta']
COLORS=['#BD5148','#44789A','#8574A4','#528B78','#C39A4B']
font_manager.fontManager.addfont(HERE/'fonts/simsun.ttc')
plt.rcParams.update({'font.family':['Times New Roman','SimSun'],'font.size':11,
    'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#B8C3CC',
    'axes.linewidth':.65,'grid.color':'#E3E9EC','axes.axisbelow':True,
    'legend.frameon':False,'pdf.fonttype':42,'savefig.dpi':360})

def save(fig,name,title):
    save_clean(fig,FIG,name)

rows=[];histories={};sources=[]
for layout,seeds in SEEDS.items():
    prefix='f1_gate' if layout=='aligned' else 'softalign_gate'
    for seed in seeds:
        suffix='five_seeds' if seed<=2030 else 's2031_2035'
        for variant in VAR:
            folder=ROOT/f'problem2_retrain_v2/results/{layout}/ablations/{prefix}_component_ablation_{suffix}/runs/{variant}_seed{seed}'
            summary=json.loads((folder/'summary.json').read_text());history=json.loads((folder/'history.json').read_text())
            assert summary['seed']==seed
            scores=[.7*h['complete']['accuracy']+.3*h['local30']['accuracy'] for h in history]
            best=int(np.argmax(scores));assert history[best]['epoch']==summary['best_epoch']
            assert abs(scores[best]-summary['selection_score'])<1e-8
            rows.append({'layout':layout,'seed':seed,'variant':variant,'best_epoch':summary['best_epoch'],
                'selection_score':scores[best], 'boundary_weight':summary['boundary_weight'],'balance_beta':summary['balance_beta'],
                'accuracy':summary['complete']['accuracy'],'macro_f1':summary['complete']['macro_f1'],
                'mae':summary['complete']['mae'],'pearson':summary['complete']['pearson'],
                'local30_accuracy':summary['local30']['accuracy'],'checkpoint':summary['checkpoint'],
                'checkpoint_sha256':summary['checkpoint_sha256']})
            histories[(layout,seed,variant)]=history
            for filename in ['summary.json','history.json']:
                p=folder/filename;sources.append({'path':str(p.relative_to(ROOT)),'sha256':sha256(p)})
frame=pd.DataFrame(rows);frame.to_csv(OUT/'valid_candidates.csv',index=False)

# 图9：只有两个真实训练端点，不补造中间点。
sensitivity=[]
fig,axes=plt.subplots(2,2,figsize=(9,6.3),layout='constrained')
for row,layout in enumerate(SEEDS):
    for col,(off,key,label,values) in enumerate([('noBeta','balance_beta','Class-balance beta',[0,.25]),('noB','boundary_weight','Boundary lambda',[0,.1])]):
        ax=axes[row,col];paired=[]
        for seed in SEEDS[layout]:
            subset=frame.query('layout==@layout and seed==@seed').set_index('variant')
            assert subset.loc[off,key]==values[0] and subset.loc['B0',key]==values[1]
            y=subset.loc[[off,'B0'],'selection_score'].to_numpy()*100;paired.append(y)
            ax.plot(values,y,'o-',lw=1,alpha=.4,color='#44789A')
            for x,score in zip(values,y):sensitivity.append({'layout':layout,'seed':seed,'parameter':key,'value':x,'selection_score':score/100})
        arr=np.array(paired)
        ax.errorbar(values,arr.mean(0),yerr=arr.std(0,ddof=1),fmt='o-',lw=2,color='#BD5148',capsize=4)
        ax.set(title=NAMES[layout],xlabel=label,ylabel='Validation score (%)',xticks=values);ax.margins(x=.2);ax.grid(alpha=.4)
pd.DataFrame(sensitivity).to_csv(OUT/'parameter_endpoints.csv',index=False)
save(fig,'09_valid_parameter_endpoints','参数端点对照：关闭与当前取值')

# 只对valid推理，不读取test、附件3或附件4。阈值诊断不更改权重与正式配置。
torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
@torch.inference_mode()
def infer(model,data,device):
    logits=[];raw=[]
    for lo in range(0,len(data['id']),64):
        b=batch(data,torch.arange(lo,min(lo+64,len(data['id']))),device,dtype=torch.float32)
        output=model(b);logits.append(output['logits'].cpu());raw.append(output['intensity'].cpu())
    return torch.cat(logits).numpy(),torch.cat(raw).numpy()

threshold_rows=[];margins=np.linspace(-1,1,81)
for layout,seeds in SEEDS.items():
    data=None
    for seed in seeds:
        row=frame.query("layout==@layout and seed==@seed and variant=='B0'").iloc[0]
        checkpoint=ROOT/row.checkpoint;assert sha256(checkpoint)==row.checkpoint_sha256
        model,cfg=restore(checkpoint,'cuda:3')
        if data is None:
            data=load_split(cfg['data_dir'],'valid',dtype=np.float32)
            assert len(data['id'])==728
            partial=(synchronized_view(data,.3,(0,1,2),'random',4026)[0] if layout=='aligned' else
                     masked_view(data,None,.3,(0,1,2),'random',4026,reencode=False)[0])
        valid_sha=sha256(Path(cfg['data_dir'])/'valid.npz')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['data_identity']['valid']==valid_sha
        del saved
        for viewname,view in [('full',data),('local30',partial)]:
            logits,raw=infer(model,view,'cuda:3');y=view['c'].cpu().numpy()
            np.savez_compressed(OUT/f'{layout}_seed{seed}_{viewname}_valid.npz',logits=logits,true_class=y,raw_intensity=raw,ids=np.asarray(view['id'],dtype=str))
            for tau in margins:
                adjusted=logits.copy();adjusted[:,1]-=tau;pred=adjusted.argmax(-1)
                precision,recall,f1,_=precision_recall_fscore_support(y,pred,labels=[0,1,2],zero_division=0)
                acc=accuracy_score(y,pred)
                if abs(tau)<1e-10:
                    expected=row.accuracy if viewname=='full' else row.local30_accuracy
                    assert abs(acc-expected)<1e-8,(layout,seed,viewname,acc,expected)
                threshold_rows.append({'layout':layout,'seed':seed,'view':viewname,'tau':float(tau),'accuracy':acc,
                    'macro_f1':f1_score(y,pred,labels=[0,1,2],average='macro'),
                    'neutral_precision':precision[1],'neutral_recall':recall[1],'neutral_f1':f1[1]})
        del model;torch.cuda.empty_cache()
        print('valid logits verified',layout,seed,flush=True)
threshold=pd.DataFrame(threshold_rows);threshold.to_csv(OUT/'threshold_scan.csv',index=False)
score=threshold.pivot(index=['layout','seed','tau'],columns='view',values='accuracy').reset_index()
score['score']=.7*score['full']+.3*score['local30']
choices={}
fig,axes=plt.subplots(2,3,figsize=(12,6.4),layout='constrained')
for row,layout in enumerate(SEEDS):
    g=score.query('layout==@layout').groupby('tau').score.mean()
    best=float(sorted(g.index,key=lambda t:(-g.loc[t],abs(t),t))[0])
    choices[layout]={'tau':best,'score':float(g.loc[best]),'default_score':float(g.loc[0.]),'grid_boundary':abs(best)==1}
    ax=axes[row,0];ax.plot(g.index,g.values*100,color='#BD5148');ax.axvline(best,color='#BD5148',ls='--');ax.axvline(0,color='#88959D',ls=':');ax.set(ylabel='Validation score (%)',title=f'{NAMES[layout]} · tau={best:.3f}')
    fullthreshold=threshold.query("layout==@layout and view=='full'").groupby('tau').mean(numeric_only=True)
    for ax,keys in zip(axes[row,1:],[['accuracy','macro_f1'],['neutral_precision','neutral_recall']]):
        for key,color in zip(keys,['#44789A','#528B78']):ax.plot(fullthreshold.index,fullthreshold[key]*100,color=color,label={'accuracy':'Accuracy','macro_f1':'Macro-F1','neutral_precision':'Neutral precision','neutral_recall':'Neutral recall'}[key])
        ax.axvline(best,color='#BD5148',ls='--');ax.axvline(0,color='#88959D',ls=':');ax.legend(fontsize=9);ax.set_ylabel('Validation metric (%)')
    for ax in axes[row]:ax.set_xlabel('Neutral logit margin tau');ax.grid(alpha=.4)
save(fig,'10_valid_threshold','中性判定边界的验证集诊断')
(OUT/'threshold_candidates.json').write_text(json.dumps({'choices':choices,'applied_to_production':False,'criterion':'.7 full valid accuracy + .3 local30 valid accuracy; mean over specified seeds','grid':[-1,1,81]},ensure_ascii=False,indent=2))
(OUT/'provenance.json').write_text(json.dumps({'seeds':SEEDS,'split':'valid','sources':sources,'test_or_special_sets_read':False,'production_modified':False},ensure_ascii=False,indent=2))

lines=['# 验证集选模、参数与决策边界分析','','本补充报告只使用附件2验证集及训练过程保存的验证指标；本次未读取test、附件3、附件4，不修改正式模型。',
'使用既定的对齐2027/2028/2035、未对齐2026/2031/2033。这些种子此前参考过test被指定，因此本次虽仅分析valid，也不能把历史方案选择重新描述为完全未接触test的过程。',
'']
lines+=['','## 3. 参数端点敏感性','',
'![参数端点对照](figures/类别平衡与边界损失的端点对照.png)',
'**图注：** β比较0与0.25（noBeta对B0），边界损失最大λ比较0与0.1（noB对B0）。细蓝线连接同seed结果，红色为均值±样本标准差。各端点来自独立训练并按相同验证规则选取的检查点；λ含训练初期渐增，标的是配置最大值。本图仅支持“关闭/当前值”的局部对照，连线不代表中间参数已实验，不能证明β=0.25或λ=0.1是全局最优。',
'对齐版完整三参数扫描见[超参数敏感性分析](超参数敏感性分析.md)。本图仅保留早期端点对照。',
'', '## 4. 阈值是否需要调整','',
'现有正式代码直接argmax三类logits，没有概率阈值。为分析中性边界，本次引入仅用于诊断的偏置：`pred = argmax([logit_negative, logit_neutral − tau, logit_positive])`。tau=0严格恢复原规则；tau增大使中性判定更严格，tau减小使中性判定更宽松。它是logit间隔阈值，不是情感强度阈值或概率阈值。',
'![验证集阈值分析](figures/中性判定边界分析.png)',
'**图注：** 固定六个B0检查点，仅对验证集完整与局部30%缺失视图推理；tau在[-1,1]按0.025步长扫描。左列按同一70/30规则计算三个指定seed的平均分数，另外两列展示完整验证集的Accuracy/Macro-F1及中性Precision/Recall。灰虚线为tau=0，红虚线为扫描范围内分数最高的候选阈值；并列时优先最接近0，再取较小值。每种布局共用一个候选tau，不为每个seed单独挑值。六个模型两种视图在tau=0的Accuracy均与保存验证结果一致。',
'', '| 布局 | 候选tau | 默认分数 | 候选分数 |','|---|---:|---:|---:|']
for layout,v in choices.items():lines.append(f'| {NAMES[layout]} | {v["tau"]:.3f} | {v["default_score"]:.4f} | {v["score"]:.4f} |')
lines+=['', '候选阈值尚未应用到正式预测，也未据此重新测试test或专项集。本验证集同时用于选轮次及阈值，因此图中的最优分数是校准集结果，不能当作独立泛化提升。若最优位于搜索边缘，只能称该扫描范围内的候选。',
'', '## 复现与范围','',
'在CPMCM根目录执行：`conda run -n CPMCM python "problem2_retrain_v2 copy/验证集选模分析.py"`。阈值诊断调用cuda:3推理验证集，不训练。原始logits、参数端点数据、候选阈值和源文件哈希保存在validation_analysis/。本报告补齐已有证据与新增验证诊断，对齐版多取值参数扫描已单独完成。']
(HERE/'验证集选模与参数分析.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(choices,ensure_ascii=False),flush=True)
