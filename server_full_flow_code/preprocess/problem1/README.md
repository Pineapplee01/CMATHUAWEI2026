# 问题一预处理

本目录来自最终提交的 `问题一/代码`，用于从附件1原始视频和标签生成方法 A、方法 B 的三模态特征。

## 输入与依赖

- 输入：`AAAdata/Appendix_1/` 中的视频与 `label-100.xlsx`。
- 文本：`AAAmodel/bert-base-uncased`。
- 音频：`ffmpeg` 抽取 wav，`AAAmodel/covarep` 生成 74 维 COVAREP 特征。
- 视觉：`AAAmodel/OpenFace/build/bin/FeatureExtraction` 生成 35 维 AU 特征。
- 对齐：`AAAmodel/mfa` 中的字典和声学模型。

## 入口

```bash
python preprocess/problem1/method_a.py extract --config preprocess/problem1/config_a.json
python preprocess/problem1/method_b.py extract --config preprocess/problem1/config_b.json
```

脚本会按配置写入 `AAAdata/processed/` 下的结果目录；`--limit N` 可用于小样本烟测。
