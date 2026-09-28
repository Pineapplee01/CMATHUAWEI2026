"""F0–F3架构族（当前正式采用F1）：观测条件补全→可选跨模态融合→窗口池化→加性/Concat头。

迁移problem2_v4的结构思想，保留主链路数据契约、在线BERT微调和损失。
不导入旧数据处理/裁剪/情感预训练分支；由model.py工厂创建；F1关闭额外融合块并使用Concat头。
"""
import itertools
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from shared.common import path


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


class CrossModalFusion(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.attn=nn.ModuleList([nn.MultiheadAttention(d,heads,dropout=dropout,batch_first=True) for _ in range(3)])
        self.norm=nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        self.alpha=nn.Parameter(torch.full((3,),.1))
        self.null=nn.Parameter(torch.zeros(1,1,d));self.drop=nn.Dropout(dropout)

    def forward(self, h, P):
        outputs=[];n=len(h)
        for m in range(3):
            others=[j for j in range(3) if j!=m]
            memory=torch.cat([h[:,j] for j in others]+[self.null.expand(n,-1,-1)],1)
            padding=torch.cat([~P[:,j] for j in others]+[torch.zeros(n,1,dtype=torch.bool,device=h.device)],1)
            message,_=self.attn[m](h[:,m],memory,memory,key_padding_mask=padding,need_weights=False)
            outputs.append(self.norm[m](h[:,m]+self.drop(self.alpha[m]*message))*P[:,m,:,None])
        return torch.stack(outputs,1)


class WindowHead(nn.Module):
    def __init__(self, d, kind, dropout):
        super().__init__();self.kind=kind;self.pairs=list(itertools.combinations(range(3),2))
        if kind=='concat':
            self.mlp=nn.Sequential(nn.Linear(3*(d+3),2*d),nn.GELU(),nn.Dropout(dropout),nn.Linear(2*d,4))
        elif kind=='additive':
            self.bias=nn.Parameter(torch.zeros(4))
            self.single=nn.ModuleList([nn.Sequential(nn.Linear(d+3,d),nn.Tanh(),nn.Linear(d,4)) for _ in range(3)])
            self.pair=nn.ModuleList([nn.Sequential(nn.Linear(2*(d+3),d),nn.Tanh(),nn.Linear(d,4)) for _ in range(3)])
        else:raise ValueError('F3头必须为additive或concat')

    def forward(self, pooled, special=None):
        active=pooled[...,-3]+pooled[...,-2]>0
        if self.kind=='concat':
            weight=active/active.sum(-1,keepdim=True).clamp_min(1)
            value=(pooled*weight[...,None]).sum(2).flatten(1)
            if special is None:return self.mlp(value)
            # 等价于拼接后的一层Linear；分块参数保留原F1头的初始化。
            hidden=self.mlp[0](value)+self.special_linear(special)
            return self.mlp[3](self.mlp[2](self.mlp[1](hidden)))
        scores=self.bias.expand(len(pooled),-1)
        for m,head in enumerate(self.single):
            value=pooled[:,m];weight=active[:,m]/active[:,m].sum(-1,keepdim=True).clamp_min(1)
            scores=scores+((head(value)-head(torch.zeros_like(value)))*weight[...,None]).sum(1)
        for (a,b),head in zip(self.pairs,self.pair):
            value=torch.cat([pooled[:,a],pooled[:,b]],-1);used=active[:,a]&active[:,b]
            weight=used/used.sum(-1,keepdim=True).clamp_min(1)
            scores=scores+((head(value)-head(torch.zeros_like(value)))*weight[...,None]).sum(1)
        return scores


class F3EmotionModel(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__();self.cfg=cfg
        from transformers import AutoConfig,AutoModel
        if not cfg.get('finetune_bert'):raise ValueError('本轮F3适配要求保留全量BERT微调')
        directory=path('AAAmodel/bert-base-uncased')
        self.bert=(AutoModel.from_pretrained(directory,local_files_only=True) if pretrained else
                   AutoModel.from_config(AutoConfig.from_pretrained(directory,local_files_only=True)))
        if cfg.get('bert_trainable_layers',len(self.bert.encoder.layer))!=len(self.bert.encoder.layer):
            raise ValueError('本轮F3候选仅支持全量BERT微调，不能静默忽略解冻层数')
        if getattr(self.bert,'pooler',None) is not None:self.bert.pooler.requires_grad_(False)
        with np.load(path(cfg['data_dir'])/'scaler_params.npz') as z:
            self.register_buffer('text_mu',torch.as_tensor(z['mu_T'],dtype=torch.float32))
            self.register_buffer('text_sigma',torch.as_tensor(z['sigma_T'],dtype=torch.float32))
        if not (self.text_sigma>0).all():raise ValueError('文本scaler必须为正')
        d=cfg.get('hidden',96);heads=cfg.get('heads',4);dropout=cfg.get('f3_dropout',.1)
        self.window=cfg.get('f3_window',5)
        if not isinstance(self.window,int) or not 1<=self.window<=50:raise ValueError('窗口必须为1到50的整数')
        self.project=nn.ModuleList([nn.Sequential(nn.Linear(w,d),nn.GELU(),nn.Dropout(.2)) for w in (768,74,35)])
        self.position=nn.Parameter(torch.randn(50,d)*.02);self.modality=nn.Parameter(torch.randn(3,d)*.02)
        self.query=nn.Parameter(torch.randn(3,d)*.02);self.null=nn.Parameter(torch.zeros(1,1,d))
        self.completion=nn.ModuleList([CompletionBlock(d,heads,dropout) for _ in range(2)])
        self.mean=nn.ModuleList([nn.Linear(d,w) for w in (768,74,35)])
        self.logvar=nn.ModuleList([nn.Linear(d,w) for w in (768,74,35)])
        # 先构造头再构造融合块，使同类头在cross开关两侧拥有一致初始化。
        self.head=WindowHead(d,cfg.get('f3_head','concat'),dropout)
        self.fusion=CrossModalFusion(d,heads,dropout) if cfg.get('f3_cross',True) else None
        self.special_tokens=cfg.get('special_tokens','none')
        if self.special_tokens not in ('none','cls','cls_sep'):
            raise ValueError('special_tokens必须为none、cls或cls_sep')
        if self.special_tokens!='none':
            if self.head.kind!='concat':raise ValueError('结构词元分支仅支持Concat头')
            self.special_project=nn.Sequential(nn.Linear(self.bert.config.hidden_size,d),nn.LayerNorm(d),nn.GELU())
            count=1 if self.special_tokens=='cls' else 2
            self.head.special_linear=nn.Linear(count*d,2*d,bias=False)

    def initialize_shared(self, state):
        """明确仅迁移可比的投影与位置参数；不默默加载不兼容的旧分类头。"""
        own=self.state_dict()
        keys=[k for k in own if k.startswith('project.') or k in ('position','modality')]
        for k in keys:
            if k not in state or own[k].shape!=state[k].shape:raise ValueError(f'共享初始化不兼容: {k}')
        self.load_state_dict({k:state[k] for k in keys},strict=False)
        return keys

    def forward(self, batch):
        P=batch['P'].bool();M=P&batch['O'].bool();token=batch['I'].clone()
        attention=M[:,0]|(token==101)|(token==102);token[~attention]=0
        empty=~attention.any(1);token[empty,0]=101;attention[empty,0]=True
        hidden=self.bert(input_ids=token,attention_mask=attention.long()).last_hidden_state
        xt=(hidden-self.text_mu)/self.text_sigma
        values=[xt*M[:,0,:,None],batch['XA']*M[:,1,:,None],batch['XV']*M[:,2,:,None]]
        observed=torch.stack([layer(x) for layer,x in zip(self.project,values)],1)
        n,_,length,d=observed.shape;offset=self.position[None,None]+self.modality[None,:,None]
        memory=torch.cat([(observed+offset).reshape(n,3*length,d),self.null.expand(n,-1,-1)],1)
        padding=torch.cat([~M.flatten(1),M.flatten(1).any(1)[:,None]],1)
        query=(offset+self.query[None,:,None]).expand(n,-1,-1,-1).reshape(n,3*length,d)
        for layer in self.completion:query=layer(query,memory,padding)
        query=query.reshape(n,3,length,d)
        mu=[layer(query[:,m]) for m,layer in enumerate(self.mean)]
        lv=[layer(query[:,m]).clamp(-6,4) for m,layer in enumerate(self.logvar)]
        reliability=torch.stack([torch.exp(-x.exp().mean(-1)) for x in lv],1)
        q=torch.where(M,torch.ones_like(reliability),reliability)*P
        completed=torch.stack([F.gelu(layer[0](x)) for layer,x in zip(self.project,mu)],1)
        h=observed*M[...,None]+completed*(P&~M)[...,None]*q[...,None]
        fused=self.fusion(h,P) if self.fusion is not None else h
        features=torch.cat([fused,M[...,None],(P&~M)[...,None],q[...,None]],-1)
        windows=[]
        for lo in range(0,length,self.window):
            mask=P[:,:,lo:lo+self.window,None]
            windows.append((features[:,:,lo:lo+self.window]*mask).sum(2)/mask.sum(2).clamp_min(1))
        special=None
        if self.special_tokens!='none':
            vectors=[]
            for token_id in ([101] if self.special_tokens=='cls' else [101,102]):
                # 读取当前遮蔽视图的BERT输出；不复制到对齐槽，不从完整视图缓存读取。
                mask=batch['I']==token_id
                value=(hidden*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
                vectors.append(self.special_project(value)*mask.any(1)[:,None])
            special=torch.cat(vectors,-1)
        pooled=torch.stack(windows,2);scores=self.head(pooled,special)
        return {'logits':scores[:,:3],'intensity':3*scores[:,3].tanh(),
                'completed':h,'observed_embedding':observed,'fused':fused,'q':q,'pooled':pooled}
