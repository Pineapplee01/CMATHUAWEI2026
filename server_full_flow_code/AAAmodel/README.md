# AAAmodel 依赖说明

本地迁移整理版没有把服务器 `AAAmodel/` 的大模型、外部工具和权重整体复制进来。本目录仅作为路径占位。

## 服务器来源

服务器路径：

```text
/user_home/gaojianan/CPMCM/AAAmodel
```

## 运行所需依赖

```text
AAAmodel/bert-base-uncased/
AAAmodel/OpenFace/build/bin/FeatureExtraction
AAAmodel/covarep/
AAAmodel/mfa/english_us_arpa_v3.0.0.dict
AAAmodel/mfa/english_us_arpa_acoustic_v3.0.0.zip
AAAmodel/checkpoints/selected/problem2/aligned_gate_B0_primary.pt
AAAmodel/checkpoints/selected/problem2/unaligned_gate_B0_primary.pt
AAAmodel/checkpoints/selected/problem2/aligned_init_boundary010.pt
```

系统工具还需要：

```text
ffmpeg
matlab
mfa
```

## 本地已有清单

本机只保留了小体积元数据和清单：

```text
G:/CMath/2026华为杯/preprocessing_models_summary
```

其中包括 BERT tokenizer/config 元数据、MFA 清单、服务器模型目录大小和正式问题二/三权重清单。完整权重与工具仍需按 `AAA提交版代码及结果/模型下载与使用说明.md` 下载或从服务器复制。
