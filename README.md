# 复杂场景下多模态情感识别

采用现有分析报告的竞赛背景。环境为CPMCM（Python 3.11、GPU PyTorch）；模型在AAAmodel，数据在AAAdata，各问代码、说明与结果在problemX。

- 问题一：[三种模态无效片段识别](problem1/三种模态无效片段识别方法.md)。
- 问题二：[问题二正式模型与运行入口](problem2/README.md)。正式权重为 `AAAmodel/checkpoints/problem2_aligned.pt`，seed2026 test ACC=72.63%，不增加CLS/SEP输出分支；唯一架构文件为 `problem2/model.py`。
- 问题三：[代码与配置](problem3/)。加载问题二使用 `problem2.training.restore`，应核对问题三配置中的checkpoint及SHA是否对应问题二正式模型。

## 问题二论文消融实验

```bash
conda activate CPMCM
python -m problem2.ablation
```

默认五个seed，对六组实验输出性能、连续缺失鲁棒性、稳定性、隐空间补偿和门控干预分析，生成六张PNG/PDF论文图。详细设置及小样本检查命令见[消融实验说明](problem2/消融实验说明.md)。研究实验不自动替换问题二正式模型，也不能把其新模块描述为已采用模型的既有功能。

旧README、旧架构与过期入口见[整理归档](archives/20260925_problem2_f1_cleanup/README.md)。保留历史结果与模型用于审计，当前使用以上入口。

## 完整运行总入口

```bash
conda activate CPMCM
python run_problem2.py
```

默认串联代码检查、正式权重test/valid及连续缺失评估、附件3预测、五seed六组研究消融与绘图。自选seed、仅评估及断点续跑用法见[总脚本使用说明](problem2/完整运行脚本使用说明.md)。
