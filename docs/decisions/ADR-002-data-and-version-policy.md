# ADR-002：数据镜像与特征版本策略

## Status

Accepted

## Decision

完整题目附件复制到本地 `data/raw/`，代码只从 `data/` 读取；原始数据和派生结果不进入 Git。`aligned/unaligned` 通过配置显式选择，初始化阶段不指定主线。

## Rationale

题目附件约3.8GB，公开 GitHub 仓库不能承载完整视频和 PKL；显式版本选择能避免两套时序接口混用。
