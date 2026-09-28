"""一个文件定义完整架构与目标：逐槽融合、分类、回归和可调中性样本权重。"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class AlignedEmotionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg=cfg
        d=cfg.get('hidden',96)
        self.project=nn.ModuleList([nn.Sequential(nn.Linear(w,d),nn.GELU(),nn.Dropout(.2))
                                    for w in (768,74,35)])
        self.position=nn.Parameter(torch.randn(50,d)*.01)
        self.modality=nn.Parameter(torch.randn(3,d)*.01)
        self.null=nn.Parameter(torch.zeros(1,1,d))
        self.cross_enabled=cfg['architecture']=='cross_attention'
        if cfg['architecture'] not in ('pooled','cross_attention'):raise ValueError('未知架构')
        if self.cross_enabled:
            self.cross=nn.MultiheadAttention(d,4,dropout=.1,batch_first=True)
            self.norm=nn.LayerNorm(d)
            self.ff=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Dropout(.2),nn.Linear(2*d,d))
            self.norm2=nn.LayerNorm(d)
        self.pool_score=nn.ModuleList([nn.Sequential(nn.Linear(d,d//2),nn.Tanh(),nn.Linear(d//2,1))
                                       for _ in range(3)])
        # 每模态同时保留均值池化和可学习注意力池化，缺失率显式输入融合。
        self.fusion=nn.Sequential(nn.Linear(6*d+3,2*d),nn.GELU(),nn.Dropout(.3),nn.LayerNorm(2*d))
        self.text_class=nn.Linear(2*d,3)
        self.class_head=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Dropout(.2),nn.Linear(d,3))
        self.regression_head=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Dropout(.2),nn.Linear(d,1))
        self.class_scale=nn.Parameter(torch.tensor(.1))

    def forward(self, batch):
        M=batch['O'].bool() & batch['P'].bool()
        # 唯一的输入操作是掩码与可训练投影：没有scaler、CLS广播、情感模型或先验logits。
        h=torch.stack([layer(batch[key]*M[:,m,:,None]) for m,(key,layer) in
                       enumerate(zip(('XT','XA','XV'),self.project))],1)
        n,_,_,d=h.shape
        if self.cross_enabled:
            memory=(h+self.position[None,None]+self.modality[None,:,None]).reshape(n,150,d)
            memory=torch.cat((memory,self.null.expand(n,-1,-1)),1)
            available=M.flatten(1).any(1)
            padding=torch.cat((~M.flatten(1),available[:,None]),1)
            # 可见槽用其内容作query，缺失槽仅用位置/模态；所有key均来自可见槽。
            query=(h*M[...,None]+self.position[None,None]+self.modality[None,:,None]).reshape(n,150,d)
            message,_=self.cross(query,memory,memory,key_padding_mask=padding,need_weights=False)
            u=self.norm(query+message)
            h=self.norm2(u+self.ff(u)).reshape(n,3,50,d)
            pooling_mask=batch['P'].bool()
        else:
            pooling_mask=M
        pools=[];weights=[]
        for m,score in enumerate(self.pool_score):
            mask=pooling_mask[:,m]
            logits=score(h[:,m]).squeeze(-1).masked_fill(~mask,-1e4)
            attention=logits.softmax(-1)*mask
            attention=attention/attention.sum(-1,keepdim=True).clamp_min(1e-8)
            mean=(h[:,m]*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
            pools.append(torch.cat((mean,(attention[...,None]*h[:,m]).sum(1)),-1))
            weights.append(attention)
        observed_ratio=M.sum(-1).to(h.dtype)/batch['P'].sum(-1).clamp_min(1).to(h.dtype)
        fused=self.fusion(torch.cat((*pools,observed_ratio),-1))
        logits=self.text_class(pools[0])+self.class_scale*self.class_head(fused)
        raw=3*torch.tanh(self.regression_head(fused).squeeze(-1))
        return {'logits':logits,'intensity':raw,'pool_attention':torch.stack(weights,1)}


class BertAlignedEmotionModel(AlignedEmotionModel):
    """在原槽位上在线编码；缺失词先删除，再进入可微调通用BERT。"""
    def __init__(self, cfg, pretrained=True):
        super().__init__(cfg)
        import numpy as np
        from transformers import AutoConfig, AutoModel
        from shared.common import path
        directory=path('AAAmodel/bert-base-uncased')
        self.bert=(AutoModel.from_pretrained(directory,local_files_only=True) if pretrained
                   else AutoModel.from_config(AutoConfig.from_pretrained(directory,local_files_only=True)))
        depth=len(self.bert.encoder.layer)
        self.trainable_bert_layers=int(cfg.get('bert_trainable_layers',depth))
        if not 1<=self.trainable_bert_layers<=depth:raise ValueError('BERT可训练层数超出范围')
        if self.trainable_bert_layers<depth:
            self.bert.requires_grad_(False)
            for layer in self.bert.encoder.layer[-self.trainable_bert_layers:]:layer.requires_grad_(True)
        # pooler不参与逐槽输出，不为无梯度参数维护优化器状态。
        if getattr(self.bert,'pooler',None) is not None:self.bert.pooler.requires_grad_(False)
        with np.load(path(cfg['data_dir'])/'scaler_params.npz') as z:
            self.register_buffer('text_mu',torch.as_tensor(z['mu_T'],dtype=torch.float32))
            self.register_buffer('text_sigma',torch.as_tensor(z['sigma_T'],dtype=torch.float32))
        if not (self.text_sigma>0).all():raise ValueError('无效的train文本缩放参数')

    def train(self, mode=True):
        super().train(mode)
        # 冻结底层时同时关闭底层dropout，避免给固定表示添加训练噪声。
        if self.trainable_bert_layers<len(self.bert.encoder.layer):
            self.bert.embeddings.eval()
            for layer in self.bert.encoder.layer[:-self.trainable_bert_layers]:layer.eval()
        return self

    def forward(self, batch):
        observed=batch['O'][:,0].bool() & batch['P'][:,0].bool()
        token=batch['I'].clone()
        structural=(token==101)|(token==102)
        attention=observed|structural
        token[~attention]=0
        empty=~attention.any(1)
        token[empty,0]=101;attention[empty,0]=True
        hidden=self.bert(input_ids=token,attention_mask=attention.long()).last_hidden_state
        # 只缩放新产生的原始BERT隐藏态一次，忽略缓存XT，不重处理XA/XV。
        xt=(hidden-self.text_mu)/self.text_sigma
        return super().forward(batch|{'XT':xt*observed[...,None]})


def build_model(cfg, pretrained=True):
    if cfg.get('architecture')=='f3':
        from problem2.model_f3 import F3EmotionModel
        return F3EmotionModel(cfg,pretrained=pretrained)
    if cfg.get('finetune_bert',False):return BertAlignedEmotionModel(cfg,pretrained=pretrained)
    return AlignedEmotionModel(cfg)


def validate_neutral_weight(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('neutral_weight必须为有限正数；1.0表示不增加中性权重')
    return value


def validate_balance_beta(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('balance_beta必须为0到1之间的有限数；0关闭频数平衡，1为逆频数平衡')
    return value


def class_balance_metadata(labels, balance_beta=1.0, neutral_weight=1.0):
    """仅由完整训练集标签计算固定权重；顺序：负向、中性、正向。"""
    beta=validate_balance_beta(balance_beta)
    neutral_weight=validate_neutral_weight(neutral_weight)
    labels=torch.as_tensor(labels).detach().cpu()
    if labels.ndim!=1 or labels.numel()==0 or labels.is_floating_point() or labels.is_complex():
        raise ValueError('训练标签必须是非空一维整数类别')
    if not ((labels>=0)&(labels<=2)).all():raise ValueError('训练类别必须为0、1、2')
    counts=torch.bincount(labels.long(),minlength=3).tolist()
    if min(counts)==0:raise ValueError('训练集缺少类别，不能估计三类平衡权重')
    total=sum(counts)
    weights=[(total/(3*n))**beta for n in counts]
    weights[1]*=neutral_weight
    normalizer=sum(n*w for n,w in zip(counts,weights))/total
    weights=[w/normalizer for w in weights]
    return {'balance_beta':beta,'neutral_weight':neutral_weight,
            'train_class_counts':counts,'class_weights':weights}


def boundary_settings(cfg):
    result={'boundary_weight':float(cfg.get('boundary_weight',0.0)),
            'boundary_margin':float(cfg.get('boundary_margin',0.5)),
            'boundary_warmup_epochs':float(cfg.get('boundary_warmup_epochs',3)),
            'boundary_max_intensity':float(cfg.get('boundary_max_intensity',0.5))}
    if any(not math.isfinite(v) for v in result.values()):raise ValueError('边界训练参数必须有限')
    if result['boundary_weight']<0 or result['boundary_margin']<0:
        raise ValueError('边界权重和间隔不能为负')
    if result['boundary_warmup_epochs']<1 or not result['boundary_warmup_epochs'].is_integer():
        raise ValueError('边界预热轮数必须为正整数')
    if not 0<result['boundary_max_intensity']<=3:raise ValueError('弱极性强度上限须在(0,3]')
    result['boundary_warmup_epochs']=int(result['boundary_warmup_epochs'])
    return result


def neutral_boundary_loss(logits, batch, margin=0.5, max_intensity=0.5):
    """中性与弱极性的批内成对排序；log-odds不受共同logit平移影响。"""
    score=logits[:,1]-torch.logsumexp(logits[:,[0,2]],dim=-1)
    neutral=score[batch['c']==1]
    weak=score[(batch['c']!=1)&(batch['y'].abs()>0)&(batch['y'].abs()<=max_intensity)]
    if neutral.numel()==0 or weak.numel()==0:return logits.sum()*0
    return F.softplus(margin-neutral[:,None]+weak[None,:]).mean()


def objective(output, batch, neutral_weight=1.0, *, class_weights=None,
              boundary_weight=0.0, boundary_margin=0.5, boundary_max_intensity=0.5):
    logits,y=output['logits'],output['intensity']
    neutral_weight=validate_neutral_weight(neutral_weight)
    if class_weights is None:
        class_weights=[1.,neutral_weight,1.]
    else:
        if neutral_weight!=1.0:raise ValueError('class_weights已包含中性倍率，不能重复传入neutral_weight')
        if len(class_weights)!=3:raise ValueError('class_weights必须按负、中、正提供三个权重')
        class_weights=[validate_neutral_weight(w) for w in class_weights]
    if class_weights == [1.,1.,1.]:
        # 默认值严格沿用旧目标，避免改变已有实验的损失及梯度。
        classification=F.cross_entropy(logits,batch['c'],label_smoothing=.03)
    else:
        per_sample=F.cross_entropy(logits,batch['c'],label_smoothing=.03,reduction='none')
        weights=logits.new_tensor(class_weights)[batch['c']]
        # 按真实类别加权整个平滑CE，不额外提高非中性样本的中性平滑目标。
        # 除以权重和，保持分类损失与回归/有序辅助项的尺度可比。
        classification=(per_sample*weights).sum()/weights.sum()
    regression=F.huber_loss(y,batch['y'])
    # 有序类别CDF：负↔正跨越两个边界，比相邻类别错误约束更强。
    target=F.one_hot(batch['c'],3).to(logits.dtype).cumsum(-1)[:,:2]
    ordinal=(logits.softmax(-1).cumsum(-1)[:,:2]-target).square().mean()
    loss=classification+.2*regression+.1*ordinal
    if boundary_weight:
        loss=loss+boundary_weight*neutral_boundary_loss(logits,batch,boundary_margin,boundary_max_intensity)
    return loss
