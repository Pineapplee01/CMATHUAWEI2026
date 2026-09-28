"""直接读取已标准化逐槽NPZ；缺失文本重编码只使用通用BERT及保存的scaler。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from shared.common import path, json_write
from problem2_retrain_v2.local_missingness import local_mask

ROOT_DATA = 'AAAdata/Appendix_2/标准化/对齐版本_retrain_20260925'
ALLOWED = ('processed', 'processed_all_zscore', 'processed_float64', 'processed_po')
FEATURES = ('XT', 'XA', 'XV')


def sha256(filename):
    h = hashlib.sha256()
    with Path(filename).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_split(directory, split, dtype=np.float32):
    directory = path(directory)
    if directory.name not in ALLOWED:
        raise ValueError('仅允许审核过的通用BERT逐槽数据版本，禁止情感先验或CLS广播版本')
    with np.load(directory / f'{split}.npz', allow_pickle=True) as z:
        f = {k: z[k] for k in z.files}
    ids = np.asarray(f['id'] if 'id' in f else f['ids']).astype(str).tolist()
    if 'P_T' in f:return load_native_arrays(f,directory,split,dtype)
    P = f['P'].astype(bool)
    if 'O' in f:
        O = f['O'].astype(bool)
    else:
        O = np.stack([f[k] for k in ('mT', 'mA', 'mV')], 1).astype(bool)
    if P.ndim == 2:
        # 旧processed的P表示Padding=True，与processed_po的有效范围P相反。
        if not np.array_equal(P, ~f['Q'].astype(bool)):
            raise ValueError('旧格式P必须等于非attention位置')
        P = np.repeat((~P)[:, None], 3, 1)
    # 排除结构词元作为情感内容，但不通过浮点特征非零来重新推断O。
    P[:, 0] &= ~np.isin(f['I'], [0, 101, 102])
    # 实验：CLS/SEP 槽上音视 P 也置 0（与文本一致剔出有效轴；PAD 本已在 extent 外）。
    # 仅 problem2_retrain_v2；正式 problem2 / 原 NPZ 不改。
    cls_sep = np.isin(f['I'], [101, 102])
    P[:, 1] &= ~cls_sep
    P[:, 2] &= ~cls_sep
    O &= P
    arrays = {k: np.ascontiguousarray(f[k], dtype=dtype) for k in FEATURES}
    n = len(ids)
    for k, width in zip(FEATURES, (768, 74, 35)):
        if arrays[k].shape != (n, 50, width) or not np.isfinite(arrays[k]).all():
            raise ValueError(f'{directory}/{split}: 非法{k}形状或非有限值')
    if O.shape != (n, 3, 50) or not (O <= P).all() or len(set(ids)) != n:
        raise ValueError('无效掩码或重复id')
    y = np.asarray(f['regression_labels'], dtype=dtype).reshape(-1)
    c = np.asarray(f['classification_labels'], dtype=np.int64).reshape(-1)
    if not np.isfinite(y).all() or (abs(y)>3).any() or not np.array_equal(c, np.sign(y).astype(int)+1):
        raise ValueError('极性/强度标签不一致')
    # 数据不减均值、不除标准差、不裁剪；模型只应用有效观测/实验保留掩码。
    return {**{k: torch.from_numpy(v) for k,v in arrays.items()},
            'P': torch.from_numpy(P), 'O': torch.from_numpy(O),
            'I': torch.from_numpy(f['I'].astype(np.int64)),
            'y': torch.from_numpy(y), 'c': torch.from_numpy(c), 'id': ids,
            'directory': str(directory), 'split': split}


def load_native_arrays(f,directory,split,dtype):
    """未对齐50/500/500；保留各自掩码，不插值、不重复标准化。"""
    ids=np.asarray(f['id']).astype(str).tolist();n=len(ids);out={}
    for m,key,length,width in zip('TAV',FEATURES,(50,500,500),(768,74,35)):
        x=np.ascontiguousarray(f[key],dtype=dtype);P=f['P_'+m].astype(bool);O=f['O_'+m].astype(bool)
        if x.shape!=(n,length,width) or P.shape!=(n,length) or O.shape!=P.shape:
            raise ValueError(f'未对齐{m}形状错误')
        if not np.isfinite(x).all() or not (O<=P).all():raise ValueError('未对齐特征/掩码非法')
        if m=='T':P=P&~np.isin(f['I'],[0,101,102]);O=O&P
        out[key]=torch.from_numpy(x);out['P_'+m]=torch.from_numpy(P);out['O_'+m]=torch.from_numpy(O)
    y=np.asarray(f['regression_labels'],dtype=dtype).reshape(-1)
    labels=np.asarray(f['classification_labels']).reshape(-1)
    if len(set(ids))!=n or not np.isfinite(y).all() or (abs(y)>3).any() or not np.array_equal(labels,np.sign(y)+1):
        raise ValueError('未对齐id/标签非法')
    if f['I'].shape!=(n,50):raise ValueError('文本词元形状错误')
    return out|{'I':torch.from_numpy(f['I'].astype(np.int64)),'y':torch.from_numpy(y),
                'c':torch.from_numpy(labels.astype(np.int64)),'id':ids,
                'directory':str(directory),'split':split,'layout':'unaligned_50_500_500'}


def batch(data, indices, device, dtype=None):
    return {k: (v[indices].to(device=device, dtype=dtype if v.is_floating_point() else v.dtype) if torch.is_tensor(v) else [v[i] for i in indices.tolist()])
            for k,v in data.items() if torch.is_tensor(v) or k == 'id'}


def audit_data(output):
    """只用train/valid核对候选来源、逐槽变化、等价版本和划分；不读test。"""
    rows, signatures = [], {}
    for name in ALLOWED:
        directory = path(ROOT_DATA) / name
        signature = hashlib.sha256()
        for split in ('train', 'valid'):
            data = load_split(directory, split)
            for k in (*FEATURES, 'P', 'O', 'I', 'c', 'y'):
                signature.update(data[k].numpy().tobytes())
            signature.update(json.dumps(data['id']).encode())
            if split == 'train': train_ids = data['id']
            else:
                if set(train_ids)&set(data['id']): raise ValueError('划分id重叠')
                if {s.split('$_$')[0] for s in train_ids}&{s.split('$_$')[0] for s in data['id']}:
                    raise ValueError('划分视频重叠')
            valid = data['O'][:,0,2] & data['O'][:,0,3]
            delta = (data['XT'][valid,2]-data['XT'][valid,3]).abs().mean().item()
            if not np.isfinite(delta) or delta < 1e-5: raise ValueError('文本疑似CLS广播')
        digest = signature.hexdigest()
        rows.append({'dataset':name,'status':'eligible','float32_payload_sha256':digest,
                     'equivalent_to':signatures.get(digest), 'text_slot_delta':delta,
                     'scaler_sha256':sha256(directory/'scaler_params.npz')})
        signatures.setdefault(digest,name)
    rows += [{'dataset':name,'status':'excluded','reason':reason} for name,reason in [
        ('processed_sentiment','情感数据集预训练并广播CLS'),
        ('processed_deberta','现有文件为CLS广播，非逐槽表示'),
        ('processed_deberta_float64','同上；float64不改变表示结构')]]
    json_write(output,rows)
    return [r['dataset'] for r in rows if r['status']=='eligible' and r['equivalent_to'] is None]


class MissingTextEncoder:
    """只为改变了文本观测的视图重算逐槽隐藏态，禁止缓存完整上下文泄漏。"""
    def __init__(self, device):
        from transformers import AutoModel
        self.device = device
        self.model = AutoModel.from_pretrained(path('AAAmodel/bert-base-uncased'),
                                               local_files_only=True).to(device).eval()
        self.model.requires_grad_(False)
        self.identity = sha256(path('AAAmodel/bert-base-uncased/model.safetensors'))

    @torch.no_grad()
    def encode(self, ids, observed, scaler):
        with np.load(scaler) as z:
            mu = torch.as_tensor(z['mu_T'],device=self.device,dtype=torch.float64)
            sigma = torch.as_tensor(z['sigma_T'],device=self.device,dtype=torch.float64)
        outputs=[]
        for lo in range(0,len(ids),128):
            token=ids[lo:lo+128].to(self.device).clone()
            obs=observed[lo:lo+128].to(self.device)
            structural=(token==101)|(token==102)
            attention=obs|structural
            token[~attention]=0
            empty=~attention.any(1)
            token[empty,0]=101;attention[empty,0]=True
            hidden=self.model(input_ids=token,attention_mask=attention.long()).last_hidden_state
            # 新生成的原始隐藏态只标准化一次；不再次标准化已有XT/XA/XV。
            scaled=((hidden.double()-mu)/sigma).float()*obs[...,None]
            outputs.append(scaled.cpu())
        return torch.cat(outputs)


def masked_view(data, encoder, rate, modalities, position, seed, cache=True, reencode=True):
    if 'P_T' in data:
        if reencode:raise ValueError('未对齐分支要求在线BERT；使用reencode=False由模型执行遮蔽编码')
        result=dict(data);records=[]
        for m in modalities:
            letter='TAV'[m]
            native={'id':data['id'],'P':data['P_'+letter][:,None].expand(-1,3,-1),
                    'O':data['O_'+letter][:,None].expand(-1,3,-1)}
            keep,part=local_mask(native,rate,(m,),position,seed)
            result['O_'+letter]=data['O_'+letter]&keep[:,m];records.extend(part)
        return result,records
    keep,records=local_mask(data,rate,modalities,position,seed)
    M=data['O'] & keep
    result={**data,'O':M}
    if reencode and 0 in modalities and not torch.equal(M[:,0],data['O'][:,0]):
        # 对文件身份、词元、实际mask与缩放参数全部做哈希，不按名字复用旧向量。
        h=hashlib.sha256()
        for a in (data['I'],M[:,0]):h.update(a.numpy().tobytes())
        scaler=Path(data['directory'])/'scaler_params.npz'
        h.update(sha256(scaler).encode());h.update(encoder.identity.encode())
        dest=path('AAAdata/processed/problem2_aligned_views')/(h.hexdigest()+'.pt')
        if cache and dest.exists():
            result['XT']=torch.load(dest,weights_only=True)
        else:
            result['XT']=encoder.encode(data['I'],M[:,0],scaler)
            if cache:
                dest.parent.mkdir(parents=True,exist_ok=True)
                temp=dest.with_suffix('.tmp');torch.save(result['XT'],temp);temp.replace(dest)
    return result,records
