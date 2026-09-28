# 数据预处理脚本统一目录

本目录集中保存竞赛项目中可追溯的数据预处理脚本。模型训练、预测与解释入口仍保留在 `AAA提交版代码及结果/` 和各 `problem*` 目录；本目录用于说明和复现“原始附件/题目给定特征如何进入模型输入”。

## 目录

- `problem1/`：附件1原始视频、文本和标签到三模态特征的完整抽取流程。
- `problem2/`：附件2原始 `.pkl` 特征到 `train/valid/test.npz` 与 `scaler_params.npz` 的恢复版完整生成脚本。
- `problem3/`：附件4题目给定 `.pkl` 特征到问题三解释模型输入的构造逻辑。

## 数据路径约定

- 模型与外部工具默认位于 `AAAmodel/`。
- 附件1原始数据默认位于 `AAAdata/Appendix_1/`。
- 附件2最终提交已使用的数据位于 `AAAdata/Appendix_2/{对齐版本,未对齐版本}/`，其中是 NPZ 产物，不是原始 pkl。
- 若要重跑附件2预处理，请将原始 pkl 放在 `AAAdata/Appendix_2_raw/{对齐版本,未对齐版本}/`，或通过 `--raw` 显式指定。
- 附件3/4预测与解释直接读取题目给定 `.pkl` 特征，不从视频重新提取 OpenFace/COVAREP 特征。

## 快速索引

```bash
# 问题一：附件1视频 -> 方法A/方法B特征
python preprocess/problem1/method_a.py extract --config preprocess/problem1/config_a.json
python preprocess/problem1/method_b.py extract --config preprocess/problem1/config_b.json

# 问题二：附件2原始pkl -> NPZ，默认写到安全的 regenerated 目录
python preprocess/problem2/preprocess_aligned.py --raw AAAdata/Appendix_2_raw/对齐版本 --output AAAdata/Appendix_2_regenerated/对齐版本
python preprocess/problem2/preprocess_unaligned.py --raw AAAdata/Appendix_2_raw/未对齐版本 --output AAAdata/Appendix_2_regenerated/未对齐版本
```

重跑前请确认当前目录下的 README 与对应脚本参数；问题一依赖 `ffmpeg`、`matlab/COVAREP`、`OpenFace`、`MFA` 和 `bert-base-uncased`，问题二依赖 `bert-base-uncased`。
