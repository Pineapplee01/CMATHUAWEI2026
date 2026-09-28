# 问题二：正式模型与连续缺失研究实验

正式模型采用 **逐槽BERT → 观测条件补全 → 5槽窗口池化 → Concat-MLP**。保留全量BERT微调、边界λ=0.1、类别平衡β=0.25。seed2026的完整test ACC为72.63%（528/727），局部30%缺失test ACC为66.44%。未启用额外池化前跨模态融合或CLS/SEP输出分支。

唯一模型架构文件是 `model.py`，同时提供问题二正式模型与六组研究实验的模块开关。训练在 `training.py`，正式推理在 `inference.py`，数据加载与连续缺失在 `data.py`、`local_missingness.py`。新的研究模块不是已采用正式模型的既有功能，实验完整模型F也不自动替换正式模型。

## 正式入口

```bash
conda activate CPMCM
# 使用已有正式权重核验test
python -m problem2.problem2 evaluate --split test --frozen-test --device cuda:3
# 附件3预测
python -m problem2.problem2 predict --device cuda:3
# 自选seed重新训练正式模型，独立保存，不覆盖正式模型
python run_problem2.py --mode train --seed 2026 --run-name my_problem2
```

正式权重：`AAAmodel/checkpoints/problem2_aligned.pt`；加载接口：`from problem2.training import restore`。推理/训练默认配置分别为 `configs/problem2_aligned.json`、`configs/problem2_bert_finetune.json`，明确使用architecture=problem2；历史正式.pt中architecture=f3仍兼容，文件内容与SHA未改变。问题三须采用同一checkpoint及SHA，不能混用旧解释结果。

## 六组论文研究实验

```bash
# 5个seed，6组显示结果、25次训练，自动生成六张PNG/PDF图
python -m problem2.ablation
# 先检查完整流水线，小样本结果带SMOKE标记
python -m problem2.ablation --smoke --run-name continuous_ablation_check
# 指定seed
python -m problem2.ablation --seeds 2026 --run-name continuous_one_seed
```

配置为 `configs/problem2_ablation.json`；过程、公式、科学比较边界与断点恢复见[消融实验说明](消融实验说明.md)。产物在 `results/ablations/<run_name>/`，权重在 `AAAmodel/checkpoints/ablations/<run_name>/`。训练/分析/绘图分别由 `ablation.py`、`ablation_analysis.py`、`ablation_plots.py`组织，所有模型架构仍集中在model.py。

## 说明与归档

- [问题二正式模型建模推导](问题二建模思路说明.md)、[当前正式结果](当前实验结果.md)、[运行与超参数](运行说明.md)。
- [中性识别与类别平衡](中性识别改进方案.md)。
- 旧模型、F3/CLS候选代码、旧五seed脚本及失效runtime接口已[归档](../archives/20260925_problem2_f1_cleanup/README.md)，正式入口不再导入；已移除会覆盖正式权重的历史compare命令。
- 历史结果与检查点保留用于溯源，尤其共同投影初始化checkpoint仍为训练依赖。历史候选不再由新model.py加载，如需复现实验须使用对应归档版本。

既有test曾用于开发选方案，72.63%是单seed结果，不是五seed均值或独立盲测。新研究实验不能预先宣称达到该成绩。

## 完整运行总入口

```bash
conda activate CPMCM
python run_problem2.py
```

默认串联代码检查、正式权重test/valid及连续缺失评估、附件3预测、五seed六组研究消融与绘图。自选seed、仅评估及断点续跑用法见[总脚本使用说明](完整运行脚本使用说明.md)。

正式test评估默认包含252组连续缺失条件（7种组合×6种跨度×[3种固定位置+3次随机位置]），完整输入和原30%结果另列。test与valid场景分开写入 `test_scenarios/` 和 `valid_scenarios/`。

## 未对齐数据训练与评估

新增原生文本50槽、音频500槽、视觉500槽的软注意力融合，仍在同一个model.py和总入口中实现。使用独立配置与实验目录：

```bash
python run_problem2.py --mode train --config configs/problem2_unaligned.json --seed 2026 --run-name unaligned_soft_attention_seed2026
```

每个seed训练后自动评估完整/局部30%test及252组连续缺失test；不替换现有对齐模型正式权重。结构公式、50×500连接与掩码规则见[未对齐数据软注意力方案](未对齐数据软注意力方案.md)。
