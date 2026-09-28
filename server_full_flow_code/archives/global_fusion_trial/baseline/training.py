"""只使用train拟合、valid选模的联合训练。"""
import numpy as np
import pandas as pd
import torch
import time
from contextlib import nullcontext
from problem2.averaging import ExponentialAverage
from transformers import AutoTokenizer
from shared.common import path, seed_all, to_device, json_write, fingerprint, environment
from shared.data import fit_normalizer, verify_tokenizer, assert_disjoint, block_mask
from problem2.model import EmotionModel
from problem2.training_step import batch_objective
from problem2.sharpness import sharpness_neighborhood
from problem2.runtime import dataset, loader, save_checkpoint
from problem2.validation import evaluate, COMBINATIONS
from problem2.evaluation import metrics



def train(cfg):
    seed_all(cfg['seed'])
    train_data, valid = dataset(cfg, 'train'), dataset(cfg, 'valid')
    assert_disjoint(train_data, valid)
    tokenizer = AutoTokenizer.from_pretrained(path(cfg.get('input_tokenizer_path', cfg['bert_path'])), local_files_only=True)
    check = verify_tokenizer(train_data, tokenizer)
    if cfg.get('input_tokenizer_path'):
        check['input_tokenizer_sha256'] = fingerprint(cfg['input_tokenizer_path'])
    identity = fingerprint(cfg['bert_path'])
    stats = fit_normalizer(train_data)
    model = EmotionModel(cfg, stats).to(cfg['device'])
    if cfg.get('init_checkpoint'):
        initial = torch.load(path(cfg['init_checkpoint']), map_location='cpu', weights_only=False)
        if initial['encoder_sha256'] != identity: raise ValueError('热启动骨干身份不一致')
        missing, extra = model.load_state_dict(initial['model'], strict=False)
        trainable = {n for n,p in model.named_parameters() if p.requires_grad}
        allowed_new = ('head.triple_',)
        if cfg.get('train_text_classifier') and not initial['config'].get('train_text_classifier'):
            # 旧checkpoint未存冻结分类头，保留经encoder哈希核验的预训练原权重。
            allowed_new += ('text.backbone.classifier.',)
        if extra or any(n in trainable and not n.startswith(allowed_new) for n in missing):
            raise ValueError(f'热启动不完整: {missing}, {extra}')
    else:
        # 在 train 上确定固定背景，之后保存到参数文件；不读取 valid/test 背景。
        model.eval()
        background = torch.zeros_like(model.head.background)
        with torch.no_grad():
            for raw in loader(train_data, cfg):
                background += model(to_device(raw, cfg['device']))['pooled'].sum(0)
        model.head.background.copy_(background / len(train_data))
    counts = np.bincount(np.asarray(train_data.fields['classification_labels']).astype(int), minlength=3)
    weights = torch.tensor((len(train_data) / (3 * np.maximum(counts, 1)))**cfg.get('class_weight_power', 1.0), device=cfg['device'], dtype=torch.float32)
    groups = [{'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('text.backbone.')], 'lr':cfg['lr']}]
    adapters = [p for n,p in model.named_parameters() if p.requires_grad and n.startswith('text.backbone.')]
    if adapters: groups.append({'params':adapters,'lr':cfg.get('adapter_lr',cfg['lr'])})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=.5)
    output = path(cfg['output']); output.mkdir(parents=True, exist_ok=True)
    json_write(output / 'run_config.json', {'config': cfg, 'tokenizer': check,
               'encoder_sha256': identity, 'environment': environment(), 'normalizer': stats,
               'selection': cfg.get('selection_metric', 'joint_f1_mae'),
               'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)})
    ema = ExponentialAverage(model, cfg['ema_decay']) if cfg.get('ema_decay') else None
    best, wait, history = float('inf'), 0, []
    rng = np.random.default_rng(cfg['seed'])
    for epoch in range(cfg['epochs']):
        start_time = time.monotonic()
        model.train(); logs = []
        epoch_cfg = dict(cfg)
        curriculum = min(1.0, (epoch+1)/max(1, cfg.get('curriculum_epochs', 1)))
        epoch_cfg['lambda_miss'] *= curriculum
        epoch_cfg['lambda_rec'] *= curriculum
        for raw in loader(train_data, cfg, shuffle=True):
            batch = to_device(raw, cfg['device'])
            rate = float(rng.choice(cfg['missing_rates']))
            modalities = COMBINATIONS[int(rng.integers(len(COMBINATIONS)))]
            B = block_mask(batch, rate, modalities, rng)
            optimizer.zero_grad(set_to_none=True)
            rng_cpu = torch.get_rng_state()
            rng_cuda = torch.cuda.get_rng_state() if cfg['device'].startswith('cuda') else None
            loss, log = batch_objective(model, batch, B, epoch_cfg, weights)
            log['loss'] = float(loss.detach())
            if not torch.isfinite(loss):
                raise FloatingPointError(f'epoch {epoch}: 非有限损失')
            loss.backward()
            if cfg.get('sam_radius', 0):
                with sharpness_neighborhood(model, cfg['sam_radius']):
                    optimizer.zero_grad(set_to_none=True)
                    # 两次前向复用同一随机掩码，避免把Dropout变化当作参数曲率。
                    torch.set_rng_state(rng_cpu)
                    if rng_cuda is not None: torch.cuda.set_rng_state(rng_cuda)
                    neighbor_loss, _ = batch_objective(model, batch, B, epoch_cfg, weights)
                    if not torch.isfinite(neighbor_loss): raise FloatingPointError('SAM目标非有限')
                    neighbor_loss.backward()
                # 在原参数位置使用邻域梯度更新，而非永久保存扰动参数。
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); logs.append(log['loss'])
            if ema is not None: ema.update(model)
        with ema.average_parameters(model) if ema is not None else nullcontext():
            vf, _ = evaluate(model, valid, cfg)
            vm, _ = evaluate(model, valid, cfg, cfg['validation_rate'], reconstruction=False)
            a, b = metrics(vf), metrics(vm)
            score = .5 * (1-a['macro_f1']+a['mae']/3) + .5 * (1-b['macro_f1']+b['mae']/3)
            if cfg.get('selection_metric') == 'accuracy':
                # 完整验证准确率为主，缺失准确率只在相同完整准确率时破同分。
                score = -a['accuracy'] - 1e-4*b['accuracy'] + 1e-6*a['mae']
            history.append({'epoch': epoch, 'train_loss': float(np.mean(logs)), 'selection_score': score,
                            'full_macro_f1': a['macro_f1'], 'missing_macro_f1': b['macro_f1'],
                            'full_accuracy': a['accuracy'], 'missing_accuracy': b['accuracy'],
                            'full_mae': a['mae'], 'missing_mae': b['mae'],
                            'seconds': time.monotonic()-start_time})
            pd.DataFrame(history).to_csv(output / 'training_history.csv', index=False)
            if cfg.get('lr_schedule', False): scheduler.step(score)
            print(history[-1], flush=True)
            if score < best:
                best, wait = score, 0
                save_checkpoint(model, cfg, stats, identity, check, epoch, score)
                if cfg.get('stop_on_target') and a['accuracy'] > cfg.get('target_accuracy', .7):
                    break
            else:
                wait += 1
                if wait >= cfg['patience']:
                    break
