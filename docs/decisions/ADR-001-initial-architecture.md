# ADR-001：共享核心与三任务域

## Status

Accepted

## Decision

采用一个 `e_emotion` 包，使用共享 `config/data/contracts/evaluation/artifacts` 支撑 `alignment/robustness/explainability` 三个任务域。

## Rationale

三项任务共享样本 ID、三模态序列、标签和评价指标；集中契约可以减少数据泄漏、形状不一致和结果格式漂移。
