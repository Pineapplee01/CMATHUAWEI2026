"""唯一模型实现：正式F1、研究消融开关与公共训练目标。"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from shared.common import path


def modality_gate(hm, avail, gate_W, gate_w):
    """hm:(N,3,d)；α=softmax(wᵀtanh(Wh+b))，z=Σα_m h_m。"""
    scores=gate_w(torch.tanh(gate_W(hm))).squeeze(-1)
    scores=scores.masked_fill(~avail,-1e4)
    none=~avail.any(-1)
    if none.any():scores=torch.where(none[:,None],scores.new_zeros(scores.shape),scores)
    alpha=scores.softmax(-1)
    z=(alpha.unsqueeze(-1)*hm).sum(1)
    return z,alpha


class CompletionBlock(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.cross=nn.MultiheadAttention(d,heads,dropout=dropout,batch_first=True)
        self.norm1=nn.LayerNorm(d);self.norm2=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Dropout(dropout),nn.Linear(2*d,d))

    def forward(self, query, memory, padding):
        message,_=self.cross(query,memory,memory,key_padding_mask=padding,need_weights=False)
        h=self.norm1(query+message)
        return self.norm2(h+self.ff(h))


class WindowHead(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(3*(d+3),2*d),nn.GELU(),nn.Dropout(dropout),nn.Linear(2*d,4))

    def forward(self, pooled):
        active=pooled[...,-3]+pooled[...,-2]>0
        weight=active/active.sum(-1,keepdim=True).clamp_min(1)
        return self.mlp((pooled*weight[...,None]).sum(2).flatten(1))


class F1EmotionModel(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__();self.cfg=cfg
        from transformers import AutoConfig,AutoModel
        if not cfg.get('finetune_bert'):raise ValueError('F1要求保留全量BERT微调')
        directory=path('AAAmodel/bert-base-uncased')
        self.bert=(AutoModel.from_pretrained(directory,local_files_only=True) if pretrained else
                   AutoModel.from_config(AutoConfig.from_pretrained(directory,local_files_only=True)))
        if cfg.get('bert_trainable_layers',len(self.bert.encoder.layer))!=len(self.bert.encoder.layer):
            raise ValueError('F1仅支持全量BERT微调，不能静默忽略解冻层数')
        if getattr(self.bert,'pooler',None) is not None:self.bert.pooler.requires_grad_(False)
        with np.load(path(cfg['data_dir'])/'scaler_params.npz') as z:
            self.register_buffer('text_mu',torch.as_tensor(z['mu_T'],dtype=torch.float32))
            self.register_buffer('text_sigma',torch.as_tensor(z['sigma_T'],dtype=torch.float32))
        if not (self.text_sigma>0).all():raise ValueError('文本scaler必须为正')
        d=cfg.get('hidden',96);heads=cfg.get('heads',4);dropout=cfg.get('f3_dropout',.1)
        self.window=cfg.get('f3_window',5)
        if not isinstance(self.window,int) or not 1<=self.window<=50:raise ValueError('窗口必须为1到50的整数')
        # 组件消融：相对完整F1逐项拆除；默认全开即正式模型。
        self.disable_completion=bool(cfg.get('ablation_disable_completion',False))
        self.disable_reliability=bool(cfg.get('ablation_disable_reliability',False))
        if self.disable_completion and self.disable_reliability:
            raise ValueError('关掉补全后可靠性不再独立，勿同时设两个拆除开关')
        self.project=nn.ModuleList([nn.Sequential(nn.Linear(w,d),nn.GELU(),nn.Dropout(.2)) for w in (768,74,35)])
        self.position=nn.Parameter(torch.randn(50,d)*.02);self.modality=nn.Parameter(torch.randn(3,d)*.02)
        self.query=nn.Parameter(torch.randn(3,d)*.02);self.null=nn.Parameter(torch.zeros(1,1,d))
        self.completion=nn.ModuleList([CompletionBlock(d,heads,dropout) for _ in range(2)])
        self.mean=nn.ModuleList([nn.Linear(d,w) for w in (768,74,35)])
        self.logvar=nn.ModuleList([nn.Linear(d,w) for w in (768,74,35)])
        if cfg.get('f3_cross',False) or cfg.get('f3_head','concat')!='concat' or cfg.get('special_tokens','none')!='none':
            raise ValueError('正式入口仅支持F1；历史候选实现已归档')
        # v2：样本级门控 z=Σα_m h_m，α=softmax(wᵀtanh(Wh+b))
        self.gate_W=nn.Linear(d,d)
        self.gate_w=nn.Linear(d,1,bias=False)
        self.cls_head=nn.Linear(d,3)
        self.reg_head=nn.Linear(d,1)
        if self.disable_completion:
            for module in (self.completion,self.mean,self.logvar):
                module.requires_grad_(False)
            for parameter in (self.query,self.null):
                parameter.requires_grad_(False)
        elif self.disable_reliability:
            self.logvar.requires_grad_(False)

    def initialize_shared(self, state):
        """明确仅迁移可比的投影与位置参数；不默默加载不兼容的旧分类头。"""
        own=self.state_dict()
        keys=[k for k in own if k.startswith('project.') or k in ('position','modality')]
        for k in keys:
            if k not in state or own[k].shape!=state[k].shape:raise ValueError(f'共享初始化不兼容: {k}')
        self.load_state_dict({k:state[k] for k in keys},strict=False)
        return keys

    def encode_observed(self, batch):
        P=batch['P'].bool();M=P&batch['O'].bool();token=batch['I'].clone()
        attention=M[:,0]|(token==101)|(token==102);token[~attention]=0
        empty=~attention.any(1);token[empty,0]=101;attention[empty,0]=True
        hidden=self.bert(input_ids=token,attention_mask=attention.long()).last_hidden_state
        xt=(hidden-self.text_mu)/self.text_sigma
        values=[xt*M[:,0,:,None],batch['XA']*M[:,1,:,None],batch['XV']*M[:,2,:,None]]
        observed=torch.stack([layer(x) for layer,x in zip(self.project,values)],1)
        return observed,P,M

    def complete(self, observed, P, M):
        n,_,length,d=observed.shape;offset=self.position[None,None]+self.modality[None,:,None]
        memory=torch.cat([(observed+offset).reshape(n,3*length,d),self.null.expand(n,-1,-1)],1)
        padding=torch.cat([~M.flatten(1),M.flatten(1).any(1)[:,None]],1)
        query=(offset+self.query[None,:,None]).expand(n,-1,-1,-1).reshape(n,3*length,d)
        for layer in self.completion:query=layer(query,memory,padding)
        query=query.reshape(n,3,length,d)
        mu=[layer(query[:,m]) for m,layer in enumerate(self.mean)]
        completed=torch.stack([F.gelu(layer[0](x)) for layer,x in zip(self.project,mu)],1)
        if self.disable_reliability:
            # 保留补全，但去掉方差可靠性门控：缺失槽一律按 q=1 填入。
            q=P.to(observed.dtype)
            h=observed*M[...,None]+completed*(P&~M)[...,None]
        else:
            lv=[layer(query[:,m]).clamp(-6,4) for m,layer in enumerate(self.logvar)]
            reliability=torch.stack([torch.exp(-x.exp().mean(-1)) for x in lv],1)
            q=torch.where(M,torch.ones_like(reliability),reliability)*P
            h=observed*M[...,None]+completed*(P&~M)[...,None]*q[...,None]
        return h,q,completed

    def pool(self, h, P, M, q):
        length=h.shape[2]
        features=torch.cat([h,M[...,None],(P&~M)[...,None],q[...,None]],-1)
        windows=[]
        for lo in range(0,length,self.window):
            mask=P[:,:,lo:lo+self.window,None]
            windows.append((features[:,:,lo:lo+self.window]*mask).sum(2)/mask.sum(2).clamp_min(1))
        return torch.stack(windows,2)

    def forward(self, batch):
        observed,P,M=self.encode_observed(batch)
        if self.disable_completion:
            h=observed*M[...,None]
            q=M.to(observed.dtype)
            estimate=torch.zeros_like(observed)
        else:
            h,q,estimate=self.complete(observed,P,M)
        # 交互/补全后：各模态有效槽均值 → h_T,h_A,h_V ∈ R^d
        avail=P.any(-1)
        denom=P.sum(-1).to(h.dtype).clamp_min(1)
        hm=(h*P[...,None]).sum(2)/denom[...,None]
        hm=hm*avail[...,None].to(h.dtype)
        z,alpha=modality_gate(hm,avail,self.gate_W,self.gate_w)
        logits=self.cls_head(z)
        intensity=3*self.reg_head(z).squeeze(-1).tanh()
        return {'logits':logits,'intensity':intensity,'modality_alpha':alpha,
                'completed':h,'observed_embedding':observed,'fused':h,'q':q,
                'reconstruction':estimate}


class SoftAlignmentModel(nn.Module):
    """未对齐：连续单调局部窗 + 窗内软注意力；样本级模态门控 z=Σα_m h_m。"""
    def __init__(self,cfg,pretrained=True):
        super().__init__();self.cfg=cfg
        from transformers import AutoModel,AutoConfig
        source=path('AAAmodel/bert-base-uncased')
        self.bert=(AutoModel.from_pretrained(source,local_files_only=True) if pretrained else
                   AutoModel.from_config(AutoConfig.from_pretrained(source,local_files_only=True)))
        if not cfg.get('finetune_bert') or cfg.get('bert_trainable_layers',len(self.bert.encoder.layer))!=len(self.bert.encoder.layer):
            raise ValueError('未对齐分支保留全量BERT微调')
        if getattr(self.bert,'pooler',None) is not None:self.bert.pooler.requires_grad_(False)
        with np.load(path(cfg['data_dir'])/'scaler_params.npz') as z:
            self.register_buffer('text_mu',torch.as_tensor(z['mu_T'],dtype=torch.float32))
            self.register_buffer('text_sigma',torch.as_tensor(z['sigma_T'],dtype=torch.float32))
        if not (self.text_sigma>0).all():raise ValueError('文本缩放尺度必须为正')
        d=cfg.get('hidden',96);self.heads=cfg.get('heads',4);drop=cfg.get('dropout',.1)
        if d%self.heads:raise ValueError('hidden必须能被注意力头数整除')
        self.max_half_frac=float(cfg.get('align_max_half_frac',.2))
        self.min_half=float(cfg.get('align_min_half',1.))
        self.time_penalty=float(cfg.get('align_time_penalty',2.))
        if not 0<self.max_half_frac<=.5:raise ValueError('align_max_half_frac须在(0,0.5]')
        if self.min_half<1:raise ValueError('align_min_half至少1帧')
        # 未对齐组件拆除：noC=无连续窗（全局软注意）；noQ=无时间惩罚
        self.disable_span=bool(cfg.get('ablation_disable_span',False))
        self.disable_time_penalty=bool(cfg.get('ablation_disable_time_penalty',False))
        if self.disable_time_penalty:self.time_penalty=0.
        self.project=nn.ModuleList([nn.Sequential(nn.Linear(w,d),nn.GELU(),nn.Dropout(drop)) for w in (768,74,35)])
        self.positions=nn.ParameterList([nn.Parameter(torch.randn(length,d)*.02) for length in (50,500,500)])
        self.query_norm=nn.LayerNorm(d)
        # 内容注意力（手动，便于挂连续窗与单调掩码）
        self.q_proj=nn.Linear(d,d)
        self.k_proj=nn.ModuleList([nn.Linear(d,d) for _ in range(2)])
        self.v_proj=nn.ModuleList([nn.Linear(d,d) for _ in range(2)])
        self.o_proj=nn.ModuleList([nn.Linear(d,d) for _ in range(2)])
        self.attn_drop=nn.Dropout(drop)
        # 每个词槽的正向时间质量（中心）与半宽（可重叠）
        self.span_step=nn.Linear(d,1)
        self.span_half=nn.Linear(d,1)
        self.null=nn.Parameter(torch.zeros(2,1,d))
        self.norm=nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])
        # 样本级模态门控：α=softmax(wᵀtanh(Wh+b))，z=Σα_m h_m
        self.gate_W=nn.Linear(d,d)
        self.gate_w=nn.Linear(d,1,bias=False)
        self.cls_head=nn.Linear(d,3)
        self.reg_head=nn.Linear(d,1)

    def _monotonic_windows(self,query,query_mask,length):
        """词序单调：center_k 非降；每词一段连续[lo,hi]，相邻可重叠。length:(N,)有效帧数。"""
        step=F.softplus(self.span_step(query).squeeze(-1))+1e-3
        step=step*query_mask.to(step.dtype)
        # 无有效查询时退化为均匀铺开，避免除零
        empty=~query_mask.any(-1)
        if empty.any():
            uniform=query_mask.new_ones(query_mask.shape,dtype=step.dtype)/50
            step=torch.where(empty[:,None],uniform,step)
        total=step.sum(-1,keepdim=True).clamp_min(1e-6)
        ends=torch.cumsum(step,-1)/total
        starts=ends-step/total
        span=length.to(step.dtype).clamp_min(1)[:,None]-1
        centers=.5*(starts+ends)*span
        # 半宽：相对有效长度，限制上限，保证是「附近一段」
        half_frac=torch.sigmoid(self.span_half(query).squeeze(-1))*self.max_half_frac
        half=(half_frac*length.to(step.dtype)[:,None]).clamp(min=self.min_half)
        lo=centers-half
        hi=centers+half
        # 强化时序：起点/终点各自非降，使得前一词的窗不会整体跑到后一词之后
        lo=torch.cummax(lo,-1).values
        hi=torch.cummax(hi,-1).values
        hi=torch.maximum(hi,lo+self.min_half)
        L=length.to(lo.dtype)[:,None].clamp_min(1)
        lo=lo.clamp_min(0)
        lo=torch.minimum(lo,L-1e-3)
        hi=torch.maximum(hi,lo+self.min_half)
        hi=torch.minimum(hi,L)
        hi=torch.maximum(hi,lo+self.min_half)
        return centers,lo,hi

    def _windowed_cross(self,query,memory,observed,query_mask,lo,hi,centers,length,null,head_idx,return_attention):
        """窗内软注意力：内容打分 + softmax；窗外屏蔽；窗内再加相对中心的时间惩罚。"""
        n,t,d=query.shape;heads=self.heads;dh=d//heads
        q=self.q_proj(query).view(n,t,heads,dh).transpose(1,2)
        k=self.k_proj[head_idx](memory).view(n,500,heads,dh).transpose(1,2)
        v=self.v_proj[head_idx](memory).view(n,500,heads,dh).transpose(1,2)
        scores=(q@k.transpose(-2,-1))/math.sqrt(dh)  # N,H,50,500
        idx=torch.arange(500,device=query.device,dtype=lo.dtype)[None,None,:]  # 1,1,500
        lo_=lo[:,:,None];hi_=hi[:,:,None];ctr=centers[:,:,None]
        inside=(idx>=lo_)&(idx<=hi_)&observed[:,None,:]  # N,50,500
        denom=length.to(lo.dtype).clamp_min(1)[:,None,None]
        dist=((idx-ctr).abs()/denom).clamp(0,1)
        scores=scores-self.time_penalty*dist[:,None]
        null_key=self.k_proj[head_idx](null).view(n,1,heads,dh).transpose(1,2)
        null_score=(q*null_key).sum(-1,keepdim=True)/math.sqrt(dh)
        empty=~observed.any(-1)
        scores=scores.masked_fill(~inside[:,None],-1e4)
        scores=torch.cat([scores,null_score],-1)
        allow_null=empty[:,None,None,None].expand(-1,heads,t,1)
        scores[...,-1]=scores[...,-1].masked_fill(~allow_null.squeeze(-1),-1e4)
        # 该词窗内无任何观测时，允许null，保证softmax有限
        none=~inside.any(-1)
        scores[...,-1]=torch.where(none[:,None,:]|empty[:,None,None],scores.new_zeros(scores[...,-1].shape),scores[...,-1])
        weights=self.attn_drop(scores.softmax(-1))
        null_v=self.v_proj[head_idx](null).view(n,1,heads,dh).transpose(1,2)
        value=torch.cat([v,null_v],2)
        message=(weights@value).transpose(1,2).contiguous().view(n,t,d)
        message=self.o_proj[head_idx](message)
        real=weights[...,:500]
        if return_attention:return message,real
        return message,None

    def forward(self,batch,return_attention=False):
        P=[batch['P_'+m].bool() for m in 'TAV'];M=[p&batch['O_'+m].bool() for p,m in zip(P,'TAV')]
        token=batch['I'].clone();attention=M[0]|(token==101)|(token==102)
        token[~attention]=0;empty=~attention.any(1);token[empty,0]=101;attention[empty,0]=True
        raw=self.bert(input_ids=token,attention_mask=attention.long()).last_hidden_state
        values=[(raw-self.text_mu)/self.text_sigma,batch['XA'],batch['XV']]
        h=[layer(x*m[...,None])*m[...,None] for layer,x,m in zip(self.project,values,M)]
        query=self.query_norm(h[0]+self.positions[0])
        query_mask=P[0].clone();fallback=~query_mask.any(1)
        any_observed=M[1].any(1)|M[2].any(1)
        query_mask[fallback]=any_observed[fallback,None].expand(-1,50)
        aligned=[];maps={};n=len(token)
        for j in range(2):
            m=j+1
            length=P[m].long().sum(-1).clamp_min(1)
            centers,lo,hi=self._monotonic_windows(query,query_mask,length)
            if self.disable_span:
                L=length.to(lo.dtype).clamp_min(1)
                lo=torch.zeros_like(lo)
                hi=(L-1)[:,None].expand_as(hi)
                centers=.5*(lo+hi)
            memory=h[m]+self.positions[m]
            null=self.null[j:j+1].expand(n,-1,-1)
            message,weights=self._windowed_cross(
                query,memory,M[m],query_mask,lo,hi,centers,length,null,j,return_attention)
            aligned.append(self.norm[j](message)*M[m].any(1)[:,None,None])
            if return_attention:
                maps['attention_'+('audio' if j==0 else 'vision')]=weights*query_mask[:,None,:,None]
                maps['span_lo_'+('audio' if j==0 else 'vision')]=lo
                maps['span_hi_'+('audio' if j==0 else 'vision')]=hi
                maps['span_center_'+('audio' if j==0 else 'vision')]=centers
        # 交互后压成样本级 h_T,h_A,h_V（对有效文本槽做均值）
        denom=query_mask.sum(-1,keepdim=True).to(h[0].dtype).clamp_min(1)
        h_T=(h[0]*query_mask[...,None]).sum(1)/denom
        h_A=(aligned[0]*query_mask[...,None]).sum(1)/denom
        h_V=(aligned[1]*query_mask[...,None]).sum(1)/denom
        avail=torch.stack([query_mask.any(-1),M[1].any(-1),M[2].any(-1)],-1)
        stack=torch.stack([h_T,h_A,h_V],1)*avail[...,None].to(h[0].dtype)
        z,alpha=modality_gate(stack,avail,self.gate_W,self.gate_w)
        logits=self.cls_head(z)
        intensity=3*self.reg_head(z).squeeze(-1).tanh()
        return {'logits':logits,'intensity':intensity,'modality_alpha':alpha,
                'query_mask':query_mask,**maps}


def build_model(cfg, pretrained=True):
    architecture=cfg.get('architecture','f1')
    if architecture in ('f1','f3'):return F1EmotionModel(cfg,pretrained)
    if architecture=='soft_alignment':return SoftAlignmentModel(cfg,pretrained)
    raise ValueError('problem2_retrain_v2仅保留F1、组件拆除开关与未对齐soft_alignment（含样本级模态门控）')


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
