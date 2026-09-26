# Tracks

## Active

| ID | Title | Status | Priority |
| --- | --- | --- | --- |
| INIT-001 | 项目初始化与数据契约 | completed | high |
| Q1-001 | 特征提取与时序对齐 | planned | high |
| Q2-001 | 局部缺失鲁棒预测 | planned | high |
| Q3-001 | 可解释情感预测 | planned | high |
| DATA-002 | 附件2预处理 NPZ 统一入口 | in_progress | high |
| Q2-FAIR-001 | Problem 2 公平 Baseline 入口与全部基线迁移 | in_progress | high |
| MEM-001 | Baseline Workspace 记忆层与服务器目录迁移 | in_progress | high |
| PAPER-001 | 论文与提交附件 | planned | high |

## Dependencies

`INIT-001 → Q1/Q2/Q3 → PAPER-001`。问题2和问题3共享附件2适配器、指标和输出契约，但特征版本由各自实验配置显式选择。

## Completed

| ID | Title | Completed |
| --- | --- | --- |
| INIT-001 | 项目初始化与数据契约 | 2026-09-23 |
| AUMDF-001 | [AUMDF独立赛题适配记录](../references/memory/history/aumdf/README.md)：单种子20+20轮与固定测试评估；不代表Q2整体完成 | 2026-09-23 |
| EVAL-001 | [统一预测评分层](../docs/evaluation/protocol.md)、独立需求与AUMDF接入；验收见evaluation/validation.md，不代表专项提交完成 | 2026-09-23 |
| APP3-001 | [附件3专项预测协议](../docs/evaluation/appendix3.md)：无标签30文件覆盖导出；不含附件4解释 | 2026-09-23 |
| PROTO-001 | [公共预处理与对比实验评估协议](../docs/evaluation/common_preprocessing_comparison_protocol.md)：统一数据契约、连续局部缺失矩阵和评分边界；已同步到远程 `Baseline/reference` | 2026-09-24 |
| DATA-002 | 服务器附件2 `train/valid/test.npz` 字段与 SHA-256 已探针确认；协议和配置已切换到 NPZ 来源，方法级 loader 迁移尚未完成 | 2026-09-24 |
