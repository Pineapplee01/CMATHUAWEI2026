# 问题二：对齐逐槽多模态情感预测

正式方案采用 **F1：观测条件补全 → 5槽窗口池化 → Concat-MLP**，保留全量BERT微调、边界训练λ=0.1、温和类别平衡β=0.25。seed2026的test ACC为72.63%（528/727），中性召回36.08%、F1 44.71%，局部30%缺失ACC为66.44%。

F1不启用额外的池化前跨模态融合块，补全模块本身仍使用注意力。通用BERT只用本题train标签微调，保留50槽对齐；缺失文本先遮蔽再编码，不广播CLS，不使用外部情感预训练模型。XA/XV不重复标准化，在线BERT隐藏态应用一次已有train scaler。正式架构集中在 `model_f3.py`；`model.py` 保留工厂、共同损失和历史架构重载，训练与推理分别在 `training.py`、`inference.py`。

配置中的 `architecture=f3` 是F0–F3共用的架构族标识；正式F1由 `f3_cross=false, f3_head=concat` 指定，并非启用了完整F3。

## 运行

在CPMCM环境、项目根目录执行：

```bash
# 重新训练：seed自由指定，配置预算自动继承
python run_problem2.py --seed 2026 --run-name my_bert_boundary_mild
# 复核正式权重
python -m problem2.problem2 evaluate --split test --frozen-test --device cuda:3
# 使用正式模型预测附件3
python -m problem2.problem2 predict --device cuda:3
```

正式推理配置为 `configs/problem2_aligned.json`，训练默认配置为 `configs/problem2_bert_finetune.json`，均采用相同组合。正式权重位于 `AAAmodel/checkpoints/problem2_aligned.pt`，结果位于 `results/aligned/selected/`。每次重新训练仍独立保存到runs目录，不自动覆盖正式权重。`compare` 是历史冻结特征架构比较入口，会重新训练并替换正式模型，不用于复核本方案。

## 说明与结果

- [当前实验结果](当前实验结果.md)：正式指标、权重来源与核验。
- [建模思路与公式](问题二建模思路说明.md)、[中性识别推导与消融](中性识别改进方案.md)。
- [运行与超参数说明](运行说明.md)。
- [三项组合实验](results/bert_mild_balance_seed2026/实验报告.md)、[BERT微调对照](results/bert_finetune_comparison_seed2026/实验报告.md)。

epoch按valid选择；方案曾经过多轮test开发比较，正式记录如实标注test参与方案选择。当前成绩仅对应seed2026，不代表多seed均值。问题三解释结果必须与当前权重身份对应，不能混用历史模型解释。

上一正式模型（68.91%）已保存为 `AAAmodel/checkpoints/archive/problem2_before_f1_adoption.pt`；配置、文档和结果见[采用前归档](../archives/20260925_problem2_before_f1_adoption/README.md)。历史五种子、float64和冻结BERT比较不代表当前模型。

## F0–F3结构对照

本轮只训练四组，每组seed=2026。F1为72.63%，完整F3为69.46%；额外池化前注意力本轮没有增益，因此按用户要求采用F1。结构与完整指标见[F3架构对照](F3方案对照说明.md)和[实验报告](results/f3_structure_seed2026/实验报告.md)。

“50项测试”是50个独立软件检查（45个问题二及公共流程、5个问题三），覆盖掩码、梯度、权重重载、参数传递等，不是50轮模型训练。

## CLS/SEP输出分支验证

已完成同seed2026的两组对照：F1+CLS test ACC为71.66%，F1+CLS+SEP为71.53%，均未超过原F1的72.63%，因此正式方案保持不变。CLS单分支中性F1提高到50.34%，但总体准确率下降。详见[实验报告](results/special_tokens_seed2026/实验报告.md)。实验开关为 `special_tokens=none/cls/cls_sep`，未指定时为none，历史权重严格兼容。
