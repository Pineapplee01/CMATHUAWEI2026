# CPMCM 迭代过程归档

本目录整理自服务器 `/user_home/gaojianan/CPMCM`，生成时间：2026-09-28 17:54:57 UTC+08:00。

## 归档目的

`AAA提交版代码及结果/` 保存最终提交版；本目录保存建模和实验迭代过程，便于回溯各 problem 版本之间的设计变化、配置变化和主要结果摘要。

## 内容结构

- `project_root/`：项目根目录的说明、环境、依赖、打包脚本和论文文档。
- `problem_versions/`：各问题版本目录的轻量归档，包括源码、配置、说明文档、图表和摘要结果。
- `supporting_code/`：共享代码、配置和测试。
- `MANIFEST.tsv`：每个归档文件的来源、大小和 SHA256。
- `EXCLUDED_SUMMARY.md`：被排除内容的统计与原因。

## 保留规则

- 完整保留源码、配置、Markdown 说明、测试和小型脚本。
- 结果目录仅保留图表、Markdown 报告和聚合摘要指标。
- `problem2_retrain_v2 copy` 以 `problem2_retrain_v2_copy` 名称归档，避免空格目录名影响跨平台使用。

## 排除规则

以下内容未纳入 Git 归档：模型权重、原始/处理后大数据、MFA 临时文件、音频、运行日志、缓存、`__pycache__`、逐样本预测明细、mask 记录、以及超过阈值的大文件。

完整数据、权重和大规模实验产物仍以服务器 `/user_home/gaojianan/CPMCM` 为准。
