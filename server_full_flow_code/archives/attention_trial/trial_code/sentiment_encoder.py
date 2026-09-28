"""三分类情感迁移；先移除不可观测BERT词元，再重分词，避免文本泄漏。"""
from collections import OrderedDict
import torch
from problem2.token_alignment import retained_text, overlap_map
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from shared.common import path
from problem2.text_encoder import TextEncoder
from problem2.adapters import attach_adapters


class SentimentEncoder(TextEncoder):
    def __init__(self, directory, input_tokenizer):
        nn.Module.__init__(self)
        self.backbone=AutoModelForSequenceClassification.from_pretrained(path(directory),local_files_only=True)
        self.backbone.requires_grad_(False)
        self.source_tokenizer=AutoTokenizer.from_pretrained(path(input_tokenizer),local_files_only=True)
        self.tokenizer=AutoTokenizer.from_pretrained(path(directory),local_files_only=True)
        labels=[self.backbone.config.id2label[i].lower() for i in range(3)]
        if labels!=['negative','neutral','positive']: raise ValueError(f'类别顺序异常: {labels}')
        self.cache=OrderedDict();self.cache_limit=16000
        self.adapters_active=False;self.cls_mix=1.;self.backbone_dropout=False

    def enable_adapters(self, rank, alpha, layers, ffn=False):
        attach_adapters(self.backbone.roberta,rank,alpha,layers,ffn)
        self.adapters_active=True;self.cache.clear()

    def enable_classifier_tuning(self):
        """解冻已有情感分类头以适配转写语域，不新增网络层。"""
        self.backbone.classifier.requires_grad_(True)
        self.adapters_active = True
        self.cache.clear()

    def enable_last_layers(self, count):
        for layer in self.backbone.roberta.encoder.layer[-count:]:layer.requires_grad_(True)
        self.adapters_active=True;self.cache.clear()

    def encode_uncached(self, tokens, observed):
        if getattr(self,'token_level',False):
            return self.encode_token_level(tokens,observed)

        retained=[ids[mask].tolist() for ids,mask in zip(tokens[:,0].cpu(),observed.cpu())]
        sentences=self.source_tokenizer.batch_decode(retained,skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=True)
        encoded=self.tokenizer(sentences,padding=True,truncation=False,return_tensors='pt')
        if encoded['input_ids'].shape[1]>self.backbone.config.max_position_embeddings-2:
            raise ValueError('重分词长度超限，禁止静默截掉保留的内容')
        encoded={k:v.to(tokens.device) for k,v in encoded.items()}
        result=self.backbone(**encoded,output_hidden_states=True)
        feature=F.layer_norm(result.hidden_states[-1][:,0],(768,)).clone()
        # 末三维保留明确的任务先验，后续仍由三模态联合监督学习残差。
        feature[:,-3:]=result.logits
        return feature[:,None].expand(-1,50,-1)*observed[...,None]

    def encode_token_level(self,tokens,observed):
        sentences,spans,slots=[],[],[]
        for ids,mask in zip(tokens[:,0].cpu(),observed.cpu()):
            positions=torch.where(mask)[0].tolist()
            pieces=self.source_tokenizer.convert_ids_to_tokens(ids[mask].tolist())
            text,mapping=retained_text(pieces)
            sentences.append(text);spans.append(mapping);slots.append(positions)
        encoded=self.tokenizer(sentences,padding=True,truncation=False,
                               return_offsets_mapping=True,return_tensors='pt')
        offsets=encoded.pop('offset_mapping').tolist()
        if encoded['input_ids'].shape[1]>self.backbone.config.max_position_embeddings-2:
            raise ValueError('保留文本重分词超限')
        result=self.backbone(**{k:v.to(tokens.device) for k,v in encoded.items()},output_hidden_states=True)
        hidden=result.hidden_states[-1]
        output=hidden.new_zeros((len(tokens),50,768))
        for i,positions in enumerate(slots):
            if not positions:continue
            weights=overlap_map(spans[i],offsets[i],tokens.device)
            local=F.layer_norm(weights@hidden[i],(768,))
            # 保留每个词元不同的语义；句级先验仍由原账本独立计入。
            output[i,positions]=torch.cat([local[:,:765],result.logits[i].expand(len(positions),-1)],-1)
        return output
