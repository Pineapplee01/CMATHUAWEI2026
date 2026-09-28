# float64精度对照实验（2026-09-25）

**结论：不采用float64，保留现有 `processed_po + cross_attention` float32模型。** 正式权重哈希核对未变化。第一阶段只按验证集比较；随后按用户要求对已冻结的四个候选补测test，结论仍为保留float32。

## 按用户要求补测test

本次不重训、不重新挑epoch，直接评估已保存权重。精度方案的最终取舍以完整观测test Accuracy为主，不再使用验证集综合分替代。test已参与本次方案比较，应按这一使用方式报告。

| 数据 | 计算精度 | 完整test正确数 | 完整test Acc | 局部30%test Acc |
|---|---|---:|---:|---:|
| processed_float64 | float32 | 488/727 | 67.13% | 63.96% |
| processed_float64 | float64 | 489/727 | 67.26% | 64.24% |
| processed_po | float32 | 495/727 | 68.09% | 66.02% |
| processed_po | float64 | 494/727 | 67.95% | 64.92% |

当前PO输入float64比float32少预测正确1条，另一套float64输入虽比同输入float32多正确1条，但没有超过现有495/727。因此按test指标也不替换模型；差距很小，不据此宣称float64一般性无效。70%目标仍未达到。

逐样本预测、权重身份、相同局部掩码和独立核验记录位于 `results/precision/test_comparison/`。

## 实际测试范围

使用相同冻结逐槽特征、架构、种子2026、初始权重、连续缺失掩码、优化器、40轮上限及10轮早停；四组均在12轮停止、最佳epoch均为2。输入特征、预测模型参数、输出、loss、梯度及AdamW一阶/二阶累积量按指定精度计算，已做dtype断言。整数词元、类别和布尔掩码保持其原类型。

BERT提取结果固定不变：原文件最初由float32编码生成，提升至float64不能恢复当时丢失的数值信息。此次比较的是后续预测训练的数值精度，不是用double BERT重新制作另一套特征。float32沿用原方案的matmul precision=high；不同dtype/backend的dropout不保证逐位相同，同种子结果不能当作多种子显著性检验。

`processed_float64`直接以float64读取，未先降精度；`processed_po`的原float32值精确提升到float64。二者各自运行float32对照。

## 验证集结果

| 数据 | 计算精度 | 完整Acc | 局部30%Acc | 综合评分 | 耗时/秒 | 峰值显存/MiB |
|---|---|---:|---:|---:|---:|---:|
| processed_float64 | float32 | 64.42% | 58.93% | 0.6277 | 25.38 | 634.7 |
| processed_float64 | float64 | 64.70% | 58.93% | 0.6297 | 36.90 | 996.6 |
| processed_po | float32 | 64.84% | 59.89% | 0.6335 | 25.20 | 634.0 |
| processed_po | float64 | 64.29% | 60.16% | 0.6305 | 37.52 | 996.6 |

综合评分固定为 `0.7×完整验证Acc + 0.3×局部30%验证Acc`，越大越好。耗时包括训练、逐轮验证及检查点保存，不包括数据解压和冻结BERT视图准备；显存为本进程CUDA峰值分配，包含同进程仍驻留的冻结编码器，不等于整张显卡占用。

原float64数据版本的完整Acc有0.27个百分点提升，但仍未超过当前最佳方案。当前最佳PO输入改为float64，完整Acc下降0.55个百分点、局部Acc提高0.27个百分点，综合评分下降约0.30个百分点；耗时约1.49倍、峰值显存约1.57倍。因此依据预定评分保留float32。

## 保留与归档

- 正式配置与权重保持不变：`configs/problem2_aligned.json`、`AAAmodel/checkpoints/problem2_aligned.pt`。
- 完整比较、每轮日志、逐样本预测和决定记录在 `results/precision/`。
- 试验入口归档到 `archives/20260925_float64/precision_experiment.py`，不增加主流程实验分支。
- 底层数据读取/训练保留少量可验证的dtype支持，默认float32；不引入额外网络。
- 复现：在CPMCM环境和项目根目录运行 `PYTHONPATH=. python archives/20260925_float64/precision_experiment.py`。这会重跑并覆盖精度试验目录，不替换正式模型。

本次并非结论“float64永远无用”，而是这些输入、架构和当前验证协议下，收益不足以替代已选float32模型。
