"""级联方案第2步：DeBERTa-v3-large 句级文本先验。

上传的权重是 MLM 预训练底座（无情感分类头），因此采用"冻结骨干 +
train 上按视频分组 K 折训练的岭正则线性头"：每一折的 held-out 概率
都来自未见过该视频的头部，杜绝标签泄漏；test/valid 无标签则使用全
train 拟合的单一头部。输出 problem2/results/cascade/{split}_prior.npz。
"""
import argparse
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from transformers import AutoTokenizer, AutoModel
from shared.common import path, config
from problem2.runtime import dataset
from shared.common import to_device
from problem2.runtime import loader


@torch.no_grad()
def sentence_embeddings(split, model_path, tokenizer, cfg, device):
    ds = dataset(cfg, split)
    model.eval()
    chunks, ids = [], []
    for raw in loader(ds, cfg):
        batch = to_device(raw, device)
        tok, obs = batch['tokens'][:, 0, :], batch['O'][:, 0]
        retained = [ids_[np.asarray(mask) > 0].tolist()
                    for ids_, mask in zip(tok.cpu().numpy(), obs.cpu().numpy())]
        sentences = tokenizer.batch_decode(retained, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=True)
        encoded = tokenizer(sentences, padding=True, truncation=False, return_tensors='pt')
        if encoded['input_ids'].shape[1] > model.config.max_position_embeddings - 2:
            raise ValueError('DeBERTa 重分词长度超限')
        encoded = {k: v.to(device) for k, v in encoded.items()}
        hidden = model(**encoded).last_hidden_state
        mask = encoded['attention_mask'].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        pooled = torch.nn.functional.layer_norm(pooled, pooled.shape[-1:])
        chunks.append(pooled.cpu().numpy())
        ids.extend(batch['id'])
    return np.concatenate(chunks), np.asarray(ds.ids)


def fit_grouped_folds(features, labels, groups, folds, seed, c_grid=(0.05, 0.2, 1.0, 5.0)):
    """返回每折的 held-out 概率与全 train 头部；全部只用 train 标签。"""
    splitter = GroupKFold(n_splits=folds)
    splits = list(splitter.split(features, labels, groups))
    best = None
    for c in c_grid:
        held = np.zeros((len(labels), 3))
        for tr, ho in splits:
            head = LogisticRegression(max_iter=5000, C=c, class_weight='balanced') \
                .fit(features[tr], labels[tr])
            held[ho] = head.predict_proba(features[ho])
        acc = (held.argmax(1) == labels).mean()
        if best is None or acc > best[0]:
            best = (acc, c, held)
    acc, c, held = best
    full_head = LogisticRegression(max_iter=5000, C=c, class_weight='balanced') \
        .fit(features, labels)
    return full_head, c, held, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/final.json')
    parser.add_argument('--model', default='AAAmodel/DeBERTa-v3-large')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'])
    parser.add_argument('--output', default='problem2/results/cascade')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    cfg = config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(path(args.model), local_files_only=True)
    model = AutoModel.from_pretrained(path(args.model), local_files_only=True).to(args.device).eval()
    output = path(args.output); output.mkdir(parents=True, exist_ok=True)

    train = np.load(output / 'train_logits.npz', allow_pickle=True)
    train_labels = train['label']
    train_groups = np.asarray([s.split('$_$')[0] for s in train['ids']])

    head, c, held_probs, held_acc = None, None, None, None
    for split in args.splits:
        npz_path = output / f'{split}_embeddings.npy'
        if npz_path.exists():
            features, ids = np.load(npz_path), None
        else:
            features, ids = sentence_embeddings(split, args.model, tokenizer, cfg, args.device)
            np.save(npz_path, features)
        if split == 'train':
            head, c, held_probs, held_acc = fit_grouped_folds(
                features, train_labels, train_groups, args.folds, cfg['seed'])
            np.savez_compressed(output / 'train_prior.npz', probs=held_probs,
                                label=train_labels, ids=train['ids'])
            print(f'train: grouped-{args.folds}-fold held-out acc={held_acc:.4f} (C={c})')
        else:
            probs = head.predict_proba(features)
            labels = np.load(output / f'{split}_logits.npz', allow_pickle=True)['label']
            np.savez_compressed(output / f'{split}_prior.npz', probs=probs, label=labels,
                                ids=np.load(output / f'{split}_logits.npz', allow_pickle=True)['ids'])
            print(f'{split}: full-train head (C={c}) neutral recall='
                  f'{probs[np.load(output / f"{split}_logits.npz", allow_pickle=True)["label"] == 1, 1].mean():.3f}')
