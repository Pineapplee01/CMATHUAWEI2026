# 问题二：对齐逐槽多模态情感预测

正式方案为**全量BERT微调＋边界训练λ=0.1＋温和类别平衡β=0.25**。seed2026的test ACC为68.91%，中性召回32.28%、F1 39.08%，局部30%缺失ACC为66.57%。尚未达到70%；与不加权微调相比总体ACC持平，中性识别提高，局部缺失ACC略降。

通用BERT只用本题train标签微调，保留50槽对齐；缺失文本先遮蔽再编码，不广播CLS、不使用外部情感预训练模型。XA/XV不重复标准化，在线生成的BERT原始隐藏态应用一次已有train scaler。全部架构模块集中在 `model.py`，训练与重载在 `training.py`，局部缺失及附件3接口在 `inference.py`。

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

epoch按valid选择；方案曾经过多轮test开发比较，正式记录如实标注test参与方案选择。当前成绩仅对应seed2026，不代表多seed均值。问题三旧加性解释尚未迁移，不能套用于该模型。

旧正式权重保存在 `AAAmodel/checkpoints/archive/problem2_before_bert_adoption.pt`；旧配置、文档、预测和场景图表见[采用前归档](../archives/20260925_problem2_before_bert_adoption/README.md)。历史[五种子实验](五随机种子实验结果.md)和[float64对照](float64精度对照结果.md)属于冻结BERT阶段，不代表当前正式模型。

## 独立F3结构实验（未替换正式方案）

已学习参考F3，保留全量BERT微调、边界λ=0.1与类别平衡β=0.25，完成四组结构消融。seed2026下F1（共同补全＋窗口池化＋Concat-MLP）test ACC为72.63%；完整F3为69.46%。额外池化前注意力本轮未带来增益。正式模型和默认配置保持不动；结构、边界条件和完整指标见[F3架构对照](F3方案对照说明.md)与[实验报告](results/f3_structure_seed2026/实验报告.md)。
