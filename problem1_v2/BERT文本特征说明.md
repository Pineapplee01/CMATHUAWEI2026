# BERT 文本特征说明（problem1_v2）

> 三版本统一模板：模型 / 槽含义 / 掩码 / 与音视轴关系 / 是否标准化。

## 1. 模型与产出

| 项 | 取值 |
|----|------|
| 模型 | `AAAmodel/bert-base-uncased` |
| 分词 | **WordPiece**（`BertTokenizer`） |
| 张量 | `(N, 50, 768)` = `last_hidden_state` |
| 词元 ID | `input_ids` `(N, 50)`：与附件2 `I` 一致，含 `[CLS]=101` / `[SEP]=102` |
| 入口 | `text_quality.run_text_with_mfa` |
| 长度 | `max_length=50` + `add_special_tokens=True`：槽0=`[CLS]`，序列含`[SEP]`，右侧`[PAD]` |

## 2. 槽位含义（本版本）

- **50 = WordPiece 词元槽**（含 `[CLS]` / `[SEP]` / `[PAD]`），不是等分秒窗，也不是转写“词”个数。
- `text_mask` / `text_support` = `attention_mask`（1=有效词元，含 CLS/SEP）。
- `text_content_mask`：排除 CLS/SEP/PAD，对齐附件2 `mT` 语义；仅内容词挂 MFA 时间。

## 3. 与音视频时间轴的关系

- 音、视：先重采样到 `common_hop_sec=0.04`，再聚到等分 50 窗 \(B_k\)（同下标=同一时间片）。
- **文本不参与 \(B_k\)**；词元下标 ≠ 音视窗下标。

## 4. 标准化

- 对 BERT 隐状态做有效位 z-score（与音/视相同规则）。
- `--standardize` 写入 `norm_stats.npz`（`text_mean` / `text_std`）。
