# ADR-004：Problem 2 公平 Baseline 协议

## Status

Accepted，2026-09-26。

## Decision

问题2公平主表使用 `problem2-fair-v1`。对齐视图固定为
`对齐版本_retrain_20260925/processed_po`，未对齐视图固定为
`未对齐版本/processed_po`。两者均保留物理支持 `P` 和原生观测 `O`。

Q2使用共享 `q2-continuous-local-v2` manifest：complete 加63个连续局部缺失条件。
未对齐窗口以50槽相对时间锚定并映射到500槽音频/视觉。所有方法保留自身网络、损失、
valid选择和极性决策；公共层保留 raw intensity，并通过预测极性的 `project_intensity`
导出最终强度后统一评分。

公平主表只汇总种子1、2、3、相同方法/视图、相同Q2 hash和精确64条件的结果。对齐、
未对齐、未对齐窗口化结果分表。文本缺失后必须按方法自己的编码路径重编码剩余token。

## Consequences

旧 `e-competition-v1` 结果和Q2-v1/历史Q2-v2结果保持原样，不重写也不与公平主表混排。
方法来源、adapter、视图能力和运行状态由 Baseline Workspace 的方法目录统一维护。
CICA保持 `paper_only`。

公共 runner 对未就绪方法和不一致产物显式拒绝；运行产物路径由 ArtifactStore 解析。
