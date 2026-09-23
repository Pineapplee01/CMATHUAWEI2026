# 数据字典

## 附件概览

| Attachment | Content | Expected count |
| --- | --- | --- |
| 附件1 | 100条原始视频、转写和标签 | 100 samples |
| 附件2 | aligned/unaligned 三划分特征 | train/valid/test = 3395/728/727 |
| 附件3 | 局部模态缺失专项特征 | 每个版本30个 PKL |
| 附件4 | 可解释专项视频及完整特征 | 每个版本20个 PKL |

## 附件2主要字段

| Field | Meaning |
| --- | --- |
| `id` | `video_id$_$clip_id` 唯一标识 |
| `raw_text` | 原始英文转写 |
| `text` | `(N,50,768)` 文本特征 |
| `audio` | aligned 为 `(N,50,74)`，unaligned 为 `(N,500,74)` |
| `vision` | aligned 为 `(N,50,35)`，unaligned 为 `(N,500,35)` |
| `text_bert` | 三路 BERT 整数输入，与 `text` 二选一 |
| `classification_labels` | 0/1/2，对应 Negative/Neutral/Positive |
| `regression_labels` | [-3,3] 连续情感强度 |
| `audio_lengths`/`vision_lengths` | 非对齐版本的有效长度 |

附件2最外层是 `train`、`valid`、`test`，字段按样本索引集中保存；不是样本字典列表。
