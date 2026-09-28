# 问题二：对齐逐槽多模态情感预测

**统一入口：** 在根目录运行 `python run_problem2.py --seeds 42 123 2026 --device cuda:3`。seed数量和值可自由指定；每个seed独立保存结果与权重。详见[运行说明](运行说明.md)。

类别不平衡方案已接入：新训练默认 `--balance-beta 1 --neutral-weight 1`，仅按train类数计算逆频数权重；当前约为负1.1703、中1.4930、正0.6776。`--balance-beta 0` 关闭频数平衡，0.5温和平衡；`--neutral-weight` 是额外中性倍率。新实验记录seed、参数、训练类数、实际权重及中性Precision/Recall/F1。

类别平衡已完成一次seed=2026实验：测试ACC为64.92%，低于原68.09%；中性召回25.32%→43.67%，但总体准确率下降，原正式检查点保留。详见[本次与原模型对比](results/runs/20260925_030256_579306/与原模型对比.md)。这只是单seed结果，其他平衡强度尚未比较。

2026-09-25 按新约束重构。完整架构、公式和输入说明见[问题二建模思路说明](问题二建模思路说明.md)。实验汇总见[当前实验结果](当前实验结果.md)。

- 仅使用通用 `AAAmodel/bert-base-uncased` 生成的逐槽表示；没有情感数据集预训练模型、情感先验logits或CLS广播。
- 直接读取用户已有的标准化NPZ，既有XT/XA/XV不重复标准化、不重复拟合scaler、不裁剪。float64派生输入转为float32计算。
- 区分旧格式Padding掩码与 `processed_po` 有效范围掩码。文本缺失发生在通用BERT编码之前，保留原槽位索引，避免预计算上下文泄漏。
- 一种架构的全部神经网络模块集中在 `model.py`；其中提供同维度的池化对照与跨模态注意力版本。
- train训练，valid比较数据/架构及早停；冻结后才检查test。专项无标签样本不报告Accuracy。

## 文件职责

| 文件 | 内容 |
|---|---|
| `model.py` | 完整网络：投影、跨模态注意力、融合池化、分类/回归头、训练损失 |
| `data.py` | NPZ契约、数据候选审计、通用BERT缺失重编码与缓存 |
| `training.py` | 训练、早停、指标评价和检查点加载 |
| `problem2.py` | 数据审核、候选比较、冻结评估和专项预测入口 |
| `inference.py` | 168组局部缺失规律分析、附件3推理 |
| `local_missingness.py` | 保留至少一个观测的连续局部缺失掩码 |
| `evaluation.py` / `output_contract.py` | 指标图表与极性/强度一致输出 |
| `runtime.py` | 跨问题兼容入口，显式阻止旧不合规检查点继续运行 |

## 运行

在项目根目录、CPMCM环境执行。当前GPU比较使用 `cuda:3`，可按空闲设备修改。

```bash
conda activate CPMCM
python -m unittest discover -s tests -v
python -m problem2.problem2 audit
python -m problem2.problem2 compare --device cuda:3 --epochs 40
python -m problem2.problem2 evaluate --device cuda:3 --split valid --scenarios
python -m problem2.problem2 evaluate --device cuda:3 --split test --frozen-test
python -m problem2.problem2 predict --device cuda:3
```

`compare` 会训练候选并更新冻结模型，复核现有结果请使用 `evaluate`。配置为 `configs/problem2_aligned.json`，正式权重为 `AAAmodel/checkpoints/problem2_aligned.pt`；日志与结果在 `results/aligned/`，图表在 `figures/aligned/`，缺失文本缓存位于 `AAAdata/processed/problem2_aligned_views/`。

## 历史归档

旧代码、旧结果、旧图表、旧配置及对应测试已移至 [历史归档](../archives/20260925_problem2_legacy/README.md)。原始数据和用户已有预训练资源保留，但不进入当前主链路。旧70.70%测试结果不能代表新模型。

问题三原来的检查点和加性账本依赖旧网络，现已阻止通过新接口加载。其历史解释仍可查阅，但不能作为新模型的解释；新模型的池化权重也不能直接冒充Shapley或可加贡献账本。

2026-09-25补充：[float64精度实测对照](float64精度对照结果.md)。已完成真正的双精度训练比较，未超过当前最佳模型，因此继续采用float32。

按用户后续要求，已补测四个精度候选的test成绩：PO float32为68.09%，PO float64为67.95%，原float64目录采用double计算为67.26%。最终按test指标仍保留原模型；完整记录见[float64精度对照结果](float64精度对照结果.md)。

五随机种子实验使用2026—2030，只运行float32；配置见 `configs/precision_five_seeds.json`，入口为 `python -m problem2.seed_experiment`。每种输入各5次，2026复用历史核验结果。

float32五种子实验已完成：[逐种子结果与汇总](五随机种子实验结果.md)。PO输入test均值66.91%，样本标准差0.94个百分点；原标准化版本均值66.80%，标准差0.78个百分点。单次最高仍为原seed2026的68.09%，正式模型保持不变。

2026-09-25中性重点测试：[温和平衡与边界训练五组对照](results/neutral_focus_seed2026/实验对比.md)。seed2026下单独边界训练λ=0.1取得test68.64%（499/727），比原模型净多4条；中性召回仍25.32%，未达到70%。温和平衡及叠加方案下降。最佳候选独立保存，正式检查点未覆盖；公式见[中性识别改进方案第8节](中性识别改进方案.md#8-已实现的最小边界训练seed2026重点测试)。

2026-09-25训练集微调BERT：[两组实测报告](results/bert_finetune_comparison_seed2026/实验报告.md)。seed2026全量微调test68.91%（501/727），中性召回28.48%，局部30%缺失66.85%；顶部4层微调68.23%。全量候选及附件3预测独立保留，正式检查点未覆盖。使用通用bert-base-uncased，标签梯度仅来自本题train。

当前总训练入口已采用**全量BERT微调＋边界λ=0.1＋更温和类别平衡β=0.25**，直接执行 `python run_problem2.py --seed 2026 --run-name my_bert_boundary_mild`。本轮seed2026 test ACC为68.91%，与不加权微调持平，中性召回32.28%、F1 39.08%；局部缺失ACC66.57%。原正式权重仍保留，详见[三项组合实验报告](results/bert_mild_balance_seed2026/实验报告.md)。
