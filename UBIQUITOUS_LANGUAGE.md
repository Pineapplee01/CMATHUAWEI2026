# Ubiquitous Language

## 核心任务

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **情感预测 / emotion prediction** | 根据文本、语音和视觉输入预测情感极性与连续情感强度。 | 只写“情感识别”、泛化为“情感分析” |
| **情感极性 / polarity** | 三分类输出 Negative、Neutral 或 Positive。 | sentiment class、类别标签 |
| **情感强度 / intensity** | 区间为 [-3, 3] 的连续情感标注或预测值。 | regression label、情绪分数 |
| **模态 / modality** | 文本、语音或视觉中的一种输入通道。 | channel（在领域文档中避免） |
| **多模态样本 / multimodal sample** | 由同一 video_id 和 clip_id 标识的三模态输入及其标签。 | 视频样本（当强调特征契约时） |

## 时序与缺失

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **时序对齐 / temporal alignment** | 将不同模态的时间位置映射到可比较的共同时间组织。 | 对齐（单独使用时含义不清） |
| **对齐版本 / aligned variant** | 附件2/3/4中三模态均按50个位置组织的特征版本。 | aligned data（未说明版本） |
| **非对齐版本 / unaligned variant** | 文本50个位置、语音和视觉最多500个独立时间位置的特征版本。 | raw data |
| **局部模态缺失 / local modality missingness** | 某模态的连续时间区间特征全部不可用或为零。 | 整模态缺失、missing modality（未说明范围） |
| **有效长度 / valid length** | 序列中未被填充且可用于模型计算的位置数量。 | sequence size、长度（未说明有效性） |

## 解释与产物

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **可解释性 / explainability** | 模型输出能够以可量化、可复核的方式关联到输入证据。 | 因果解释 |
| **模态贡献 / modality contribution** | 某模态对当前预测结果的相对作用分数。 | 模态权重（除非确实是模型权重） |
| **关键证据 / salient evidence** | 与当前预测密切相关、可映射到文本、语音或视频位置的局部片段。 | 解释片段、原因（避免因果暗示） |
| **专项测试集 / special test set** | 附件3或附件4中无标签、只用于最终推理的样本集合。 | 测试集（与附件2 test 混用） |
| **数据契约 / data contract** | 规定输入字段、形状、掩码、标签和来源约束的公共接口。 | 数据格式 |

## Relationships

- 一个 **多模态样本** 由一个 `video_id + clip_id` 唯一标识。
- 一个多模态样本包含文本、语音和视觉三个 **模态**，每个模态有序列值、有效长度和位置掩码。
- 一个样本可以同时拥有一个 **情感极性** 和一个 **情感强度**。
- 一个 **专项测试集** 属于问题2或问题3，只在最终推理阶段使用。
- 一个 **关键证据** 属于一个样本和一个模态，并具有起止位置与贡献分数。

## Example dialogue

> **Dev:** 这个样本是 aligned data 还是已经完成 temporal alignment？
>
> **Domain expert:** 前者表示文件版本，后者表示处理步骤；报告中分别写“对齐版本”和“时序对齐”。
>
> **Dev:** 附件3里的零值区间能叫 missing modality 吗？
>
> **Domain expert:** 只能叫局部模态缺失，因为整个模态仍然存在。
>
> **Dev:** 模型给出的高贡献片段就是情感原因吗？
>
> **Domain expert:** 不是。它是可复核的关键证据，不能直接宣称因果关系。

## Flagged ambiguities

- “情感识别”“情感分析”和“情感预测”在资料中混用；项目总称固定为 **情感预测**，文献原名除外。
- 附件2的 `classification_labels` 是数值编码，文档中的用户可读标签固定使用 Negative/Neutral/Positive。
- 附件2的 `test` 与附件3/4的无标签专项测试必须分开称呼，避免把有标签划分和最终推理集混淆。
