# 问题一 v2：WordPiece 槽 + MFA 词时间共享 + 音视加权

| 步骤 | 内容 |
|------|------|
| 文本 | WordPiece 50；MFA 整词时间；同词子词共享 \([s,e)\) |
| 音/视 | 按各词元时间窗重叠加权均值（同窗 → 同特征） |

详见 [采用说明](采用说明.md)。

```bash
conda activate CPMCM
python -m problem1_v2.problem1 extract --limit 1 --keep-tmp
python -m problem1_v2.problem1 extract
```
