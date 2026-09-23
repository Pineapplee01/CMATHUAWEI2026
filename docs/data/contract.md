# 数据契约

## 输入边界

所有输入必须位于 `data/` 下，由 `DataPathPolicy` 校验；任务域不得直接拼接工作区根目录或读取 `E题数据/`。

## 标准对象

- `SampleId`：`video_id`、`clip_id` 和题目规定的 `$_$` 序列化。
- `ModalitySequence`：二维数值矩阵、有效长度和布尔位置掩码。
- `MultimodalSample/Batch`：三种模态、可选标签和特征版本。
- `PredictionRecord`：极性、强度和可选置信度。
- `EvidenceRecord`：模态、半开区间 `[start,end)`、贡献分数和定位器。
- `FeatureManifest`：样本覆盖、特征维度、有效长度、粒度、工具版本和来源。

## 版本规则

`aligned` 与 `unaligned` 是互斥输入版本。问题2和问题3的运行配置必须显式指定版本，训练、验证和专项测试不得混用。
