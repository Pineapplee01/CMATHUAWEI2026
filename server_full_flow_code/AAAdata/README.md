# AAAdata 依赖说明

本地迁移整理版没有把服务器 `AAAdata/` 的大体积数据整体复制进来。本目录仅作为路径占位，记录运行代码所需的数据结构。

## 服务器来源

服务器路径：

```text
/user_home/gaojianan/CPMCM/AAAdata
```

## 期望目录

```text
AAAdata/Appendix_1/
AAAdata/Appendix_2/对齐版本/
AAAdata/Appendix_2/未对齐版本/
AAAdata/Appendix_3/对齐版本/
AAAdata/Appendix_3/未对齐版本/
AAAdata/Appendix_4/对齐版本/
AAAdata/Appendix_4/未对齐版本/
AAAdata/processed/
```

其中 `Appendix_2/{对齐版本,未对齐版本}` 是最终提交使用的 NPZ 目录，包含 `train.npz`、`valid.npz`、`test.npz`、`scaler_params.npz`、`preprocess_report.json`。

若要重跑附件2预处理，原始 pkl 应另放：

```text
AAAdata/Appendix_2_raw/对齐版本/
AAAdata/Appendix_2_raw/未对齐版本/
```

默认重建输出为：

```text
AAAdata/Appendix_2_regenerated/对齐版本/
AAAdata/Appendix_2_regenerated/未对齐版本/
```

## 本地已有数据

本机已有原始数据/服务器数据审计副本位于：

```text
G:/CMath/2026华为杯/data/raw
G:/CMath/2026华为杯/data/server_original
```

数据对比和哈希报告见：

```text
G:/CMath/2026华为杯/preprocessing_audit/data_comparison_report.md
G:/CMath/2026华为杯/preprocessing_audit/data_file_comparison.tsv
```
