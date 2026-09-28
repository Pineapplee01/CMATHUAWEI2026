"""删除不可观测词元后，建立BERT槽到RoBERTa字符区间的映射。"""
import torch


def retained_text(pieces):
    text,spans='',[]
    no_space_before={'.',',','!','?',':',';',"'",')',']','}'}
    no_space_after={"'",'(','[','{'}
    previous=''
    for piece in pieces:
        continuation=piece.startswith('##')
        word=piece[2:] if continuation else piece
        join=continuation or word in no_space_before or previous in no_space_after
        if text and not join:text+=' '
        start=len(text);text+=word;spans.append((start,len(text)));previous=word
    return text,spans


def overlap_map(spans,offsets,device):
    left=torch.tensor(spans,device=device,dtype=torch.float32).reshape(-1,2)
    right=torch.tensor(offsets,device=device,dtype=torch.float32).reshape(-1,2)
    weights=(torch.minimum(left[:,None,1],right[None,:,1])-
             torch.maximum(left[:,None,0],right[None,:,0])).clamp_min(0)
    if len(left) and (weights.sum(1)==0).any():raise ValueError('保留词元无RoBERTa字符支持')
    return weights/weights.sum(1,keepdim=True).clamp_min(1e-12)
