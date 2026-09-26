# 数据契约

## 输入边界

本地任务输入必须位于 `data/` 下，由 `DataPathPolicy` 校验。服务器上的
`Baseline/reference` 对比实验使用竞赛附件2的只读预处理镜像，唯一允许的
外部输入根目录为：

`/user_home/gaojianan/CPMCM/AAAdata/Appendix_2/标准化/对齐版本/processed/`

方法适配器不得回退到参考仓库自带的 MOSEI/MOSI 数据或
`Baseline/reference/_data/*.pkl`。

该目录固定包含 `train.npz`、`valid.npz`、`test.npz`；运行 manifest 必须记录
三个文件的绝对路径和 SHA-256。

严格对比运行还必须通过 `e_emotion.evaluation.protocol_guard`：它复用共享 NPZ
加载器检查字段 schema、dtype、维度、ID 覆盖和划分大小，并核对三个文件的
SHA-256。`*.pkl`（包括 `Baseline/reference/_data/*.pkl`）在新运行边界直接拒绝，
不能作为兼容回退。manifest 必须记录 `experiment_seed`、`mask_seed`、归一化来源、
检查点与阈值来源、mask manifest hash，以及 `native`、`synthetic`、`observed`
三层掩码语义。所有检查点、阈值、预测和诊断产物都必须位于该方法自己的
`method_root/artifact_root` 下；越界路径会被拒绝。

当前仍读取历史 PKL、或尚未把原生/合成 mask provenance 传入 manifest 的方法，在
`references/memory/catalog/methods.yaml` 中记录方法状态。该记录只约束新的
严格 NPZ 对比，不会重命名、覆盖或重新解释既有历史运行。

## 标准对象

- `SampleId`：`video_id`、`clip_id` 和题目规定的 `$_$` 序列化。
- `ModalitySequence`：二维数值矩阵、有效长度和布尔位置掩码。
- `MultimodalSample/Batch`：三种模态、可选标签和特征版本。
- `PredictionRecord`：极性、强度和可选置信度。
- `EvidenceRecord`：模态、半开区间 `[start,end)`、贡献分数和定位器。
- `FeatureManifest`：样本覆盖、特征维度、有效长度、粒度、工具版本和来源。

## 版本规则

`aligned_50` 与 `unaligned_50` 是互斥输入版本。当前对比实验统一使用
`aligned_50` NPZ：`XT=(N,50,768)`、`XA=(N,50,74)`、`XV=(N,50,35)`，
对应原生可用掩码为 `mT/mA/mV`，样本 ID 为 `ids`，标签为
`y_regression` 和 `y_classification`。训练、验证和专项测试不得混用版本。

`mT/mA/mV` 是预处理阶段保存的**原生可用掩码**（`native_valid_mask`）：它们
同时反映序列内容/有效观测和预处理时已存在的特征缺失，不能只按普通 padding
解释，也不能从标准化后的全零值重新推断。文本的 `Q/P` 仍分别保留 token
attention 与 padding 语义。完整输入时 `observed_mask` 等于原生可用掩码；Q2
局部缺失只在副本上生成独立的 `synthetic_missing_mask`，再计算
`observed_mask = native_valid_mask & ~synthetic_missing_mask` 并将被遮挡区间特征置零。

因此 `complete` 表示“保留附件2原生缺失背景的输入”，不表示人为补全为三模态完整。
若需要无原生缺失的理想上界，必须单独命名为 `oracle_complete`，不能并入主矩阵。
