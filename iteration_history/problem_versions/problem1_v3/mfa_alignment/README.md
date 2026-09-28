# MFA 对齐代码（自 problem3 独立拷贝）

本目录供 **问题一方案 A（词级 MFA）** 使用，不依赖运行 `problem3` 包路径。

| 文件 | 来源 | 作用 |
|------|------|------|
| `word_alignment.py` | `problem3/word_alignment.py` | `mfa align` → 词区间；`content_token_mask` 排除 CLS/SEP |
| `media.py` | `problem3/media.py` | ffprobe / ffmpeg / 抽 `speech.wav` |
| `media_preparation.py` | `problem3/media_preparation.py` | 批量 MFA |
| `aggregate.py` | 旧 `problem1/alignment.py` | 交叠聚合、`view50` 前 50 词 |

## 对齐单位（不含 CLS/SEP）

1. MFA 输入是转写**内容词**（`words_of`），不是 BERT tokenizer 输出。  
2. 对齐轴 $U_k=I_k=[s_k,e_k)$ 对应第 $k$ 个词；**不包含** `[CLS]`、`[SEP]`、`[PAD]`。  
3. 若后续把 BERT 隐状态映到词区间：只用 `content_token_mask==1` 的词元；特殊词元可保留编码，**不参与**时间聚合。  
4. 需要 50 槽时：`view50` 取前 50 **词**区间截断/右侧补零，不是 50 个含特殊符号的 BERT 槽。

配置字段：`mfa_executable`、`dictionary`、`acoustic`。失败时 **不** `uniform_fallback`，只记 `mfa_failed` + 原因。
