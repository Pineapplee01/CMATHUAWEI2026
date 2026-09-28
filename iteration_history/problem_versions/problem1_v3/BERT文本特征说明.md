# BERT 文本特征说明（problem1_v3）

> 三版本统一模板：模型 / 槽含义 / 掩码 / 与音视轴关系 / 是否标准化。

## 1. 模型与产出

| 项 | 取值 |
|----|------|
| 模型 | `AAAmodel/bert-base-uncased` |
| 张量 | `(N, 50, 768)` 词级向量（`view50`） |
| 入口 | `extractors.bert_words`（由 `problem1.py` 调用） |
| 配置 | `config_q1.json` → 词槽上限 50 |

## 2. 槽位含义（本版本）

- **50 = MFA 内容词槽**（转写词），**不是**含 `[CLS]`/`[SEP]` 的 BERT `max_length`。
- 每个词内部用 **WordPiece** 拆子词，对该词子词隐状态取均值得到 768 维；特殊符不占对齐槽。
- 词数 \(>50\)：`view50` **截断**保留前 50 词；不足右侧填零。
- MFA 失败：不均分；`alignment=mfa_failed`，区间/特征全零，原因进 `alignment_manifest`。

## 3. 与音视频时间轴的关系

- 三模态共用 MFA 词区间 \(I_k\)；音视源帧先统一到 `common_hop_sec=0.04` 再聚到 \(I_k\)。
- 文本槽 \(k\) 与音视槽 \(k\) **同一词时段**（不是等分秒窗 \(B_k\)）。

## 4. 标准化

- **对词级 BERT 向量做有效位 z-score**（与音/视相同）：有效词槽估参 → 标准化 → 无效/无词槽置零。
- `--standardize` 同时标准化文本 / 音频 / 视觉；`norm_stats.npz` 含 `text_mean` / `text_std`。
