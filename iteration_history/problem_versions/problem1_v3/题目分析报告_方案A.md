# 问题一分析报告（方案A：MFA 词级强制对齐）

> 对齐代码独立目录：[mfa_alignment/](mfa_alignment/README.md)（自 problem3 拷贝）。  
> **对齐不考虑 `[CLS]` / `[SEP]`**：轴上每个槽是转写词区间，不是 BERT 特殊词元。

## 1. 对齐定义

$$
\operatorname{MFA}(\text{音频},\text{题给转写})
\longrightarrow
I_k=[s_k,e_k),\quad 0\le s_k<e_k\le D,\quad k=1,\ldots,K.
$$

$$
U_k=I_k.
$$

$K=$ 规范化内容词数（`words_of`）。音、视源帧先重采样到 `common_hop_sec=0.04`，再与文本内容一并按与 $I_k$ 的时间重叠聚合到同一词槽。MFA 失败时**不**均分降级：记录 `mfa_failed` 与失败原因，词区间为空、特征置零，样本保留。

不另建等分时间网格 $B_k$；词边界即三模态共用时间轴。

## 2. 与 BERT 特殊词元的关系

| 对象 | 是否进入 MFA / $U_k$ |
|------|----------------------|
| 转写内容词 | 是 |
| `[CLS]` / `[SEP]` / `[PAD]` | **否** |
| BERT 内容词元 | 可编码；映到词区间时仅 `content_token_mask=1` |

特殊词元可保留在 BERT 序列里做上下文编码，但**不占用**对齐槽、不参与音视时间聚合。

## 3. 重叠聚合与 50 槽

$$
w_{kj}=\operatorname{len}(I_k\cap J_j),\quad
\bar{\mathbf f}_k=
\frac{\sum_{j:q_j=1}w_{kj}\mathbf f_j}{\sum_{j:q_j=1}w_{kj}}\ (D_k>0).
$$

需固定长度 50：保留前 50 词及其区间（超长截断），短序列右侧填零；全长留档。此 50 是**词位数**，不是含 CLS/SEP 的 BERT `max_length`；词内仍用 WordPiece 拆子词后取均值。

## 4. 流程与代码

```
转写 → words_of（无 CLS/SEP）
     → speech.wav + .lab
     → mfa align → TextGrid words 层
     → I_k；校验词序一致
     → aggregate(音/视/内容文本 → I_k)
     → view50（可选）
```

入口：`mfa_alignment` 已接入 `python -m problem1_v3.problem1 extract`（词区间聚合 + view50）。  
质量规则仍可与三模态无效片段文档衔接；本报告定**对齐轴**。

## 5. 配置

`mfa_executable`、`dictionary`、`acoustic`（如 `english_us_arpa`）。`fallback_uniform` 已关闭：失败只记账。边界为估计值，重要样本宜人工抽检。
