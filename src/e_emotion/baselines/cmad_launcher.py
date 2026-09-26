#!/usr/bin/env python3
"""CMAD competition adapter: protocol-safe data boundary and smoke launcher.

The upstream CMAD models/losses are intentionally imported unchanged.  This
module only adapts aligned_50.pkl's field-oriented schema to their native
teacher/student tensor interface.
"""
from __future__ import annotations
import argparse, json, os, sys, random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from e_emotion.baselines.workspace import vendor_root

ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = vendor_root() / "CMAD"
DATA = Path("/user_home/gaojianan/CPMCM/AAAdata/Appendix_2/标准化/对齐版本/processed")
BERT = Path.home() / "CPMCM" / "AAAmodel" / "bert-base-uncased"


def approved_data_root(value: str | Path) -> Path:
    protocol_src = str(ROOT.parent / "protocol_src")
    if protocol_src not in sys.path:
        sys.path.insert(0, protocol_src)
    from e_emotion.data.experiment_roots import resolve_experiment_root
    return resolve_experiment_root(value)

def load_split(path: Path, split: str):
    protocol_src = str(ROOT.parent / "protocol_src")
    if protocol_src not in sys.path:
        sys.path.insert(0, protocol_src)
    from e_emotion.npz_adapter import load_native_split
    x = load_native_split(path, split)
    return {
        "id": list(x["id"]),
        "text_embeddings": torch.as_tensor(x["text"], dtype=torch.float32),
        "attention_mask": torch.as_tensor(x["Q"], dtype=torch.long),
        "audio": torch.as_tensor(x["audio"], dtype=torch.float32),
        "vision": torch.as_tensor(x["vision"], dtype=torch.float32),
        "regression": torch.as_tensor(np.asarray(x["regression_labels"]), dtype=torch.float32),
        "classification": torch.as_tensor(np.asarray(x["classification_labels"]), dtype=torch.long),
        "mT": torch.as_tensor(x["mT"], dtype=torch.bool),
        "mA": torch.as_tensor(x["mA"], dtype=torch.bool),
        "mV": torch.as_tensor(x["mV"], dtype=torch.bool),
    }

def model_args():
    return SimpleNamespace(TEXT_DIM=768, ACOUSTIC_DIM=74, VISUAL_DIM=35,
        d_l=96, dropout_prob=.3, num_heads=16, te_layers=2, num_latents=5,
        depth=2, relu_dropout=.3, res_dropout=.3, embed_dropout=.2,
        attn_dropout=.5, p_attn_dropout=0., p_ff_dropout=0.,
        max_seq_length=50, model="bert-base-uncased", alignment="align",
        text_length=50, visual_length=50, acoustic_length=50, p=[0,0,0])

def build_models(device="cpu", config=None):
    sys.path.insert(0, str(VENDOR_ROOT / "CMAD_sentiment" / "Teacher_Model"))
    sys.path.insert(0, str(VENDOR_ROOT / "CMAD_sentiment" / "Student_Model"))
    from transformers import BertConfig
    from teacher_model import TeacherModel
    from student_model import StudentModel
    bert = Path(config['bert']).expanduser() if config else BERT
    cfg = BertConfig.from_pretrained(str(bert), local_files_only=True); cfg.num_labels = 1
    a = model_args()
    # transformers 4.44 fast-init can leave custom, checkpoint-missing Conv1d
    # parameters uninitialized.  The native model's init_weights hook must run.
    teacher = TeacherModel.from_pretrained(str(bert), config=cfg, args=a, _fast_init=False, local_files_only=True).to(device)
    student = StudentModel.from_pretrained(str(bert), config=cfg, args=a, _fast_init=False, local_files_only=True).to(device)
    student.p = a.p  # upstream training script sets this runtime attribute
    return teacher, student

def batches(x, size, shuffle):
    order=torch.randperm(len(x["id"])) if shuffle else torch.arange(len(x["id"]))
    for i in range(0,len(order),size):
        q=order[i:i+size]; yield {k:(v[q] if torch.is_tensor(v) else [v[j] for j in q]) for k,v in x.items()}

def forward(model,b,device):
    z={k:v.to(device) for k,v in b.items() if torch.is_tensor(v)}
    base_attention = z["attention_mask"].bool()
    native_text = z.get("mT", base_attention)
    observed_text = z.get("observed_mT", native_text).bool()
    # Q controls ordinary BERT padding and special tokens. mT/observed_mT
    # control only content positions, so CLS/SEP remain visible to BERT.
    text_mask = base_attention & (~native_text.bool() | observed_text)
    observed_audio = z.get("observed_mA", z.get("mA", torch.ones_like(base_attention))).bool()
    observed_visual = z.get("observed_mV", z.get("mV", torch.ones_like(base_attention))).bool()
    return model(None,z["vision"],z["audio"],z["regression"],
                 attention_mask=text_mask.to(dtype=z["attention_mask"].dtype),
                 inputs_embeds=z["text_embeddings"], text_mask=text_mask,
                 audio_mask=observed_audio, visual_mask=observed_visual),z

@torch.no_grad()
def predictions(model,x,size,device,condition=(1,1,1)):
    mode = model.training
    old_p = getattr(model, 'p', None)
    model.eval(); out=[]; truth=[]
    if old_p is not None: model.p = list(condition)
    try:
        for b in batches(x,size,False):
            y,z=forward(model,b,device)
            out.append(y[0].view(-1).cpu()); truth.append(z['regression'].view(-1).cpu())
    finally:
        model.train(mode)
        if old_p is not None: model.p = old_p
    pred, target = torch.cat(out).numpy(), torch.cat(truth).numpy()
    if not np.isfinite(pred).all(): raise FloatingPointError('nonfinite validation prediction')
    return pred, target

COMBINATIONS = [[1,0,0],[0,1,0],[0,0,1],[1,0,1],[1,1,0],[0,1,1],[1,1,1]]

class MARState:
    """Native cumulative warmup history and one-time begin_epoch freeze."""
    def __init__(self, args, device):
        self.args = args
        self.combinations = torch.tensor(COMBINATIONS, device=device)
        self.weights = torch.ones(7, device=device)
        self.ps, self.teacher_errors, self.student_errors = [], [], []

    def begin(self, epoch):
        from loss import combination_importance
        if epoch > 0 and epoch <= self.args.begin_epoch:
            candidate = combination_importance(torch.cat(self.student_errors),
                torch.cat(self.teacher_errors), torch.cat(self.ps), self.combinations, self.args) * self.args.weights
            # Native pre-transition candidate is computed but not applied.
            if epoch == self.args.begin_epoch:
                if not torch.isfinite(candidate).all():
                    raise FloatingPointError('Native MAR normalization undefined: no positive teacher/student error gap')
                self.weights = candidate
                self.ps, self.teacher_errors, self.student_errors = [], [], []
        return self.weights

    def record(self, ps, teacher_error, student_error):
        # Native weighting is no_grad; retaining graphs cannot affect its result.
        self.ps.append(ps.detach()); self.teacher_errors.append(teacher_error.detach())
        self.student_errors.append(student_error.detach())

def student_objective(pred, hidden, teacher_pred, teacher_hidden, ps, target, state):
    from loss import compute_weighted_mse_loss, regression_loss
    args = state.args
    pred, teacher_pred, target = pred.view(-1), teacher_pred.view(-1), target.view(-1)
    c2fd = compute_weighted_mse_loss(hidden, teacher_hidden, args.tau)[0]
    auxiliary = regression_loss(pred, teacher_pred, args.temperature)
    task = (pred-target).abs()
    state.record(ps, (teacher_pred-target).abs(), task)
    mask = (ps.unsqueeze(1) == state.combinations).all(dim=2).float()
    mar_auxiliary = torch.einsum('i,ij,j->i', auxiliary, mask, state.weights).mean()
    mar_task = torch.einsum('i,ij,j->i', task, mask, state.weights).mean()
    return args.delta*c2fd + mar_task + args.gamma*mar_auxiliary

def metrics(pred, target, interval):
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
    labels = np.sign(target).astype(int)
    classified = np.where(pred < interval[0], -1, np.where(pred > interval[1], 1, 0))
    _, _, f1, support = precision_recall_fscore_support(labels, classified, labels=[-1,0,1], zero_division=0)
    pearson = float(np.corrcoef(pred,target)[0,1]) if len(pred)>1 and np.std(pred)>0 and np.std(target)>0 else None
    return {'accuracy':float(accuracy_score(labels,classified)),
        'macro_f1':float(f1_score(labels,classified,labels=[-1,0,1],average='macro',zero_division=0)),
        'weighted_f1':float(f1_score(labels,classified,labels=[-1,0,1],average='weighted',zero_division=0)),
        'per_class':{name:{'f1':float(f),'support':int(n)} for name,f,n in zip(['Negative','Neutral','Positive'],f1,support)},
        'mae':float(np.abs(pred-target).mean()), 'pearson':pearson}


def write_prediction_csv(path, pred, target, interval):
    """Persist the public intensity/polarity columns for protocol auditing."""
    import pandas as pd
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    polarity = np.where(pred < interval[0], -1, np.where(pred > interval[1], 1, 0))
    frame = pd.DataFrame({
        'sample_id': [str(index) for index in range(len(pred))],
        'raw_intensity': pred,
        'intensity': np.clip(pred, -3.0, 3.0),
        'polarity': polarity,
        'target': target,
    })
    frame.to_csv(path, index=False)

def calibrate(pred, target):
    # Exhaustive asymmetric intervals containing zero, at prediction boundaries.
    # Count-based F1 makes the exact search inexpensive on the 728-example valid split.
    order = np.argsort(pred, kind='stable'); p = pred[order]; y = np.sign(target[order]).astype(int)+1
    counts = np.zeros((len(p)+1,3), dtype=int)
    counts[1:] = np.cumsum(np.eye(3,dtype=int)[y], axis=0)
    totals=counts[-1]; best=(-1.,float('-inf')); interval=[0.,0.]
    for low in np.unique(np.r_[p[p<=0],0.]):
        left = np.searchsorted(p,low,side='left')
        for high in np.unique(np.r_[0.,p[p>=0]]):
            right = np.searchsorted(p,high,side='right')
            predicted = np.array([left,right-left,len(p)-right])
            tp = np.array([counts[left,0],counts[right,1]-counts[left,1],totals[2]-counts[right,2]])
            den=totals+predicted
            score=float(np.divide(2*tp,den,out=np.zeros(3),where=den>0).mean())
            key=(score,-float(high-low))
            if key>best: best=key; interval=[float(low),float(high)]
    return interval

def confined(path):
    path=Path(path).expanduser()
    path=(path if path.is_absolute() else ROOT/path).resolve()
    if not path.is_relative_to((ROOT/'runs').resolve()):
        raise ValueError('all outputs must resolve inside CMAD/runs')
    return path

def load_config(path, data_root: str | Path = DATA):
    cfg=json.loads(Path(path).read_text())
    if (cfg['sequence_length'],cfg['audio_dim'],cfg['vision_dim']) != (50,74,35):
        raise ValueError('competition aligned interface is fixed at 50/74/35')
    if cfg['selection_split']!='valid' or cfg['selection_metric']!='MAE':
        raise ValueError('selection must use clean valid MAE')
    if cfg['primary_f1']!='macro' or set(cfg['additional_f1'])!={'weighted','per_class'}:
        raise ValueError('contract requires macro, weighted and per-class F1')
    cfg['outputs']=str(confined(cfg['outputs']))
    cfg['data']=str(approved_data_root(data_root))
    return cfg

def seed_all(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)

def optimizer_for(model, args, stage):
    from transformers.optimization import AdamW
    no_decay=['bias','LayerNorm.weight']
    groups=[{'params':[p for n,p in model.named_parameters() if not any(s in n for s in no_decay)],
             'weight_decay':args.teacher_weight_decay if stage=='teacher' else args.student_weight_decay},
            {'params':[p for n,p in model.named_parameters() if any(s in n for s in no_decay)],'weight_decay':0.}]
    return AdamW(groups, lr=args.learning_rate, eps=args.adam_epsilon)

def train_stage(stage,path,run_dir,epochs,batch_size,lr,device,teacher_ckpt=None,config=None,limit=None):
    config=config or load_config(ROOT/'competition_config.json')
    args=SimpleNamespace(**config['training']); args.learning_rate=lr
    if args.begin_epoch<1 or epochs<1 or batch_size<2:
        raise ValueError('require begin_epoch>=1, epochs>=1 and batch_size>=2')
    run_dir=confined(run_dir); run_dir.mkdir(parents=True,exist_ok=True)
    seed_all(args.seed)
    tr,va=load_split(path,'train'),load_split(path,'valid')
    if limit:
        tr={k:v[:limit] for k,v in tr.items()}; va={k:v[:limit] for k,v in va.items()}
    teacher,student=build_models(device,config)
    finite_init={name:sum(p.numel() for n,p in model.named_parameters() if not n.startswith('bert.'))
                 for name,model in [('teacher',teacher),('student',student)]}
    for model in (teacher,student):
        for name,p in model.named_parameters():
            if not torch.isfinite(p).all(): raise FloatingPointError('nonfinite initialized parameter '+name)
    if stage=="student":
        if teacher_ckpt is None or not teacher_ckpt.is_file(): raise FileNotFoundError("student stage requires --teacher-checkpoint")
        teacher.load_state_dict(torch.load(teacher_ckpt,map_location=device,weights_only=True)); teacher.eval()
        for p in teacher.parameters(): p.requires_grad=False
        model=student
    else: model=teacher
    opt=optimizer_for(model,args,stage); best=float('inf'); state=MARState(args,device)
    history=[]
    for epoch in range(epochs):
        # The native teacher train_epoch intentionally creates a fresh AdamW each epoch.
        if stage=='teacher': opt=optimizer_for(model,args,stage)
        else: state.begin(epoch); student.p=[0,0,0]
        schedule=args.teacher_schedule if stage=='teacher' else args.student_schedule
        for group in opt.param_groups: group['lr']=lr * (0.1 ** sum(epoch>=m for m in schedule))
        model.train()
        losses=[]
        for b in batches(tr,batch_size,True):
            opt.zero_grad()
            if stage=='student':
                with torch.no_grad(): (tp,th)=forward(teacher,b,device)[0]
            (pred,hid,*extra),z=forward(model,b,device)
            if stage=="teacher": loss=torch.nn.functional.l1_loss(pred.view(-1),z["regression"].view(-1))
            else:
                loss=student_objective(pred,hid,tp,th,extra[0],z['regression'],state)
            if not torch.isfinite(loss): raise FloatingPointError('native loss nonfinite; refusing to alter objective')
            loss.backward()
            if stage=='teacher': torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip)
            opt.step(); losses.append(float(loss.detach()))
        pred,target=predictions(model,va,batch_size,device)
        score=float(np.abs(pred-target).mean())
        if score<best: best=score; torch.save(model.state_dict(),confined(run_dir/f"{stage}_best.pt"))
        row={'stage':stage,'epoch':epoch,'train_loss':float(np.mean(losses)), 'valid_mae':score,
             'mar_weights':state.weights.tolist() if stage=='student' else None}
        history.append(row); print(json.dumps(row),flush=True)
    model.load_state_dict(torch.load(run_dir/f'{stage}_best.pt',map_location=device,weights_only=True))
    pred,target=predictions(model,va,batch_size,device)
    repeat,_=predictions(model,va,batch_size,device)
    if not np.array_equal(pred,repeat): raise AssertionError('clean validation not deterministic')
    interval=calibrate(pred,target)
    (run_dir/'neutral_interval.json').write_text(json.dumps({
        'lower': float(interval[0]), 'upper': float(interval[1]),
        'fit_split': 'valid', 'objective': 'macro_f1',
    }, indent=2) + '\\n')
    write_prediction_csv(run_dir/'valid_predictions.csv', pred, target, interval)
    result={'stage':stage,'seed':args.seed,'smoke_subset':limit,'initialized_finite_parameter_counts':finite_init,
            'checkpoint_reload_mae':float(np.abs(pred-target).mean()),'best_valid_mae':best,
            'neutral_interval':interval,'calibration_split':'valid','condition':[1,1,1],
            'metrics':metrics(pred,target,interval),'history':history,'config':config}
    if stage == 'student':
        test = load_split(path, 'test')
        if limit:
            test={k:v[:limit] for k,v in test.items()}
        test_pred, test_target = predictions(model, test, batch_size, device)
        write_prediction_csv(run_dir/'test_predictions.csv', test_pred, test_target, interval)
        result['test_metrics'] = metrics(test_pred, test_target, interval)
    confined(run_dir/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    np.savez(confined(run_dir/'valid_predictions.npz'),predictions=pred,targets=target)
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=ROOT/'competition_config.json')
    p.add_argument('--data',type=Path); p.add_argument('--stage',choices=('smoke','teacher','student'),default='smoke')
    p.add_argument('--device',default='cpu'); p.add_argument('--run-dir',type=Path)
    p.add_argument('--teacher-checkpoint',type=Path); p.add_argument('--epochs',type=int)
    p.add_argument('--batch-size',type=int); p.add_argument('--learning-rate',type=float); p.add_argument('--seed',type=int)
    a=p.parse_args(); data_root=approved_data_root(a.data or DATA); cfg=load_config(a.config,data_root)
    if a.seed is not None: cfg['training']['seed']=a.seed
    run=confined(a.run_dir or cfg['outputs']); run.mkdir(parents=True,exist_ok=True)
    os.environ['HF_HOME']=str(confined('runs/cache/huggingface'))
    torch.set_num_threads(2)
    kw=dict(path=data_root,epochs=a.epochs or cfg['training']['epochs'],
            batch_size=a.batch_size or cfg['training']['batch_size'],lr=a.learning_rate or cfg['training']['learning_rate'],device=a.device,config=cfg)
    if a.stage=='smoke':
        kw.update(epochs=2,batch_size=2,limit=4)
        cfg['training']['begin_epoch']=1
        train_stage('teacher',run_dir=run/'teacher',**kw)
        result=train_stage('student',run_dir=run/'student',teacher_ckpt=run/'teacher'/'teacher_best.pt',**kw)
    else: result=train_stage(a.stage,run_dir=run,teacher_ckpt=a.teacher_checkpoint,**kw)
    print(json.dumps(result,indent=2,allow_nan=False))
if __name__ == "__main__": main()
