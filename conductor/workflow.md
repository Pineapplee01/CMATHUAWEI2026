# Workflow

## Methodology

按数据契约驱动开发：先定义可验证的输入/输出，再实现任务域；新增行为先写失败测试，再实现最小代码。

## Data Rules

- `data/` 是唯一输入根目录，原始附件只读。
- 问题2/3的结构选择和阈值只能使用附件2的训练/验证划分。
- 附件3/4只用于最终专项推理，不参与训练或调参。
- 所有运行记录包含配置、随机种子、特征版本和环境信息。

## Git Conventions

- 默认分支：`main`。
- 提交格式：`type(scope): message`。
- `data/`、`artifacts/`、模型、日志和 CodeGraph 索引不提交。
- 远程仓库：`https://github.com/Pineapplee01/CMATHUAWEI2026.git`。

## Quality Gates

| Gate | Requirement |
| --- | --- |
| 契约测试 | `pytest` 全部通过 |
| 数据探针 | 两套特征版本和专项文件数量可核验 |
| CLI | `inspect-data` 和 `validate-data` 可运行 |
| 依赖图 | CodeGraph 已索引且模块边界清晰 |
| 版本同步 | 工作树干净并已推送代码与文档 |
