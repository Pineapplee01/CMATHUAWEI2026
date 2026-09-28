# 问题一 v3：方案 A（MFA 词级对齐，已接通）

音视**不是**均匀 50 份，而是 MFA 词区间；固定长度 = 前 50 **词**（超长截断）。不含 CLS/SEP。  
音视先 `common_hop=0.04` 再聚到 $I_k$；文本为词级 BERT（词内 WordPiece 均值）。  
产出目录：`problem1_v3/results/`（含 `intervals.npz`：每词槽起止秒）。

```bash
conda activate CPMCM
python -m problem1_v3.problem1 extract --limit 1
python -m problem1_v3.problem1 extract
```

详见 [题目分析报告_方案A](题目分析报告_方案A.md)、[采用说明](采用说明.md)、[mfa_alignment/](mfa_alignment/README.md)。
