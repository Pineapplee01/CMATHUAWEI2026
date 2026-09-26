# Workflow

## Methodology

按数据契约驱动开发：先定义可验证的输入/输出，再实现任务域；新增行为先写失败测试，再实现最小代码。

## Data Rules

- `data/` 是唯一输入根目录，原始附件只读。
- 问题2/3的结构选择和阈值只能使用附件2的训练/验证划分。
- 附件3/4只用于最终专项推理，不参与训练或调参。
- 附件3导出使用冻结检查点和valid阈值，逐方法写30行无标签CSV与manifest；不计算专项指标、不生成附件4解释。
- 所有运行记录包含配置、随机种子、特征版本和环境信息。
- Baseline Workspace 的 runtime 统一读取服务器只读预处理 NPZ：
  `/user_home/gaojianan/CPMCM/AAAdata/Appendix_2/标准化/对齐版本/processed/`。
  `train.npz`、`valid.npz`、`test.npz` 的字段契约为 `XT/XA/XV`、`mT/mA/mV`、
  `ids`、`y_regression`、`y_classification`；禁止新运行回退到 `_data/*.pkl`。
- `problem2-fair-v1` 的公平主表固定使用服务器 `processed_po` 双视图、`P/O`、共享
  Q2-v2 64条件 manifest 和种子1、2、3；文本局部缺失必须重编码剩余 token。
  aligned、unaligned 与 unaligned-windowed 分表，不能跨视图排名。
- 方法来源、adapter、运行状态和方法目录以
  `references/memory/catalog/methods.yaml` 为唯一来源；原始运行产物统一写入
  `artifacts/problem2/`。
- 对齐测试252协议已实现但当前暂停执行；恢复时仅对已验证的六个基线 checkpoint
  在 aligned test 划分运行。

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
