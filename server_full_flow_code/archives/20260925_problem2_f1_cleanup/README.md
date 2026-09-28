# 复杂场景下多模态情感识别

问题二自选seed总入口：`python run_problem2.py --seeds 42 123 2026`，参数及输出位置见[问题二运行说明](problem2/运行说明.md)。

> **2026-09-25 问题二主链路已重构：** 使用通用BERT逐槽特征和已有标准化NPZ。当前入口为[problem2/README](problem2/README.md)，配置 `configs/problem2_aligned.json`。旧情感预训练/CLS广播方案及其测试成绩仅作历史记录，已归档；问题三旧解释不能用于新模型。以下历史命令涉及旧问题二时请以新入口为准。

> 问题二题意补齐：当前专项入口为 `configs/problem2_robust.json`，采用连续局部缺失验证、分类/回归联合选模及一致输出；168组验证和附件3全量30条已完成。详见[局部缺失建模与题意核对](problem2/局部缺失建模与题意核对.md)。下方70.70%/75%记录属于原完整观测准确率路线。

> 最新目标与结果：用户将测试集目标提高到约75%。借鉴第一个参考仓库完成五组实验后，最佳新候选test为70.2889%，低于原模型70.7015%，已归档并恢复原模型；**75%目标尚未达到**。详见[全局融合优化记录](problem2/全局融合优化记录.md)。

依据现有分析报告的中国研究生数学建模竞赛背景实现。使用CPMCM环境：Python3.11、GPU PyTorch；模型放在AAAmodel，数据放在AAAdata。每问代码、说明、结果索引与图表集中在problemX，核心算法按多个Python模块组织。

## 正式方案与阅读入口

| 问题 | 当前实现 | 详细说明 |
|---|---|---|
| 问题一 | 采用用户指定的Data_Preproc/Q1：三模态质量掩码、74维COVAREP音频 | [采用说明](problem1/采用说明.md)、[质量规则推导](problem1/三种模态无效片段识别方法.md) |
| 问题二 | 条件补全、窗口交互融合、严格局部缺失验证、一致极性/强度输出；门控经消融关闭 | [局部缺失建模与题意核对](problem2/局部缺失建模与题意核对.md) |
| 问题三 | 8子集Shapley、实际预测账本、窗口删除、扰动与媒体回看 | [模型方法与公式推导](problem3/模型方法与公式推导.md) |

问题一源目录保持原样，正式入口在problem1；旧25维方案的源码已归档到 `archives/problem1_original/`，旧100条结果仅为历史结果。新Q1代码当前保存掩码及音频特征，不把分析报告中尚待实现的MFA/完整三模态重聚合写成已完成。文本是50个BERT词元槽截断，音视频是50个等时窗，不等于三者已严格词级对齐。

问题二/三继续读取附件2/3/4的官方预处理特征；Q1的100条媒体不并入训练。问题二专项配置为 `configs/problem2_robust.json`，参数为 `AAAmodel/checkpoints/problem2_robust.pt`；问题三及原完整准确率模型仍为 `configs/final.json` / `AAAmodel/checkpoints/main.pt`。

## 环境与命令

```bash
conda activate CPMCM
python -m pip check
python -m unittest discover -s tests -v

# 问题一：专用config_q1.json，需MATLAB/COVAREP/OpenFace/ffmpeg
python problem1/problem1.py extract --limit 1
python problem1/problem1.py extract
python -m problem1.verify_outputs

# 问题二：训练会更新指定权重；只复核指标无需重训
python problem2/problem2.py train --config configs/robust_no_reliability.json --device cuda
python -m problem2.verify_results --config configs/final.json --split valid
python -m problem2.verify_results --config configs/final.json --split test
python problem2/problem2.py predict --config configs/problem2_robust.json --device cuda
python -m problem2.verify_submission --config configs/problem2_robust.json

# 问题三：唯一入口（解释 + MFA 回溯 + 报告）
python -m problem3.problem3 --device cuda:3
```

环境清单见 `environment.yml`、`requirements.txt` 及对应lock文件。OpenFace复用现有本地构建，MATLAB为外部程序及许可；pip清单不能替代这两项。问题一需要OpenCV质量检查，已补入CPMCM环境。

## 输出与指标

- 新Q1数据：`AAAdata/processed/problem1_covarep`；小样本联调为独立 `_smoke_1` 目录；图表在 `problem1/figures`。源目录已有out仅1条，不代表100条全量完成。
- 历史Q1数据：`AAAdata/processed/problem1`（旧25维方案，保留追溯）。
- 问题二：专项CSV在 `problem2/results/robust_selected`；局部缺失规律与消融在 `problem2/results/local_robustness`，图在 `problem2/figures/local_robustness`；`results/main` 保留原完整准确率路线。
- 问题三：`problem3/results/`、`problem3/figures/`；入口为 `python -m problem3.problem3`。

当前完整观测三分类：valid **68.2692%（497/728）**，test **70.7015%（514/727）**。按用户澄清的测试集70%验收目标，当前模型已达标；valid用于选模；不更换分类口径、不删中性样本。详见 [运行与优化报告](运行与优化报告.md)。

本文与各问README为当前入口；`运行与优化报告_第一轮.md`属于历史记录。最终文档包含已启用参数与未启用实验选项的区别，不能把所有候选算法共同写成最终模型。

全量运行更新：problem1已在CPMCM环境处理全部100条，退出码0。98条COVAREP成功；2条严格全零音轨触发提取警告，样本保留、音频掩码和特征均为0。三模态掩码均为100×50，音频特征100×50×74且全部有限；质量表行数为文本100、音频5000、视觉5000，两张图已生成。验证记录见 `AAAdata/processed/problem1_covarep/full_run_validation.json`，日志见 `logs/problem1_full.log`。此结果证明运行和输出契约通过，不代表人工质量标签准确率验收。

本次新增五轮优化已结束：R-Drop、有序类别与频数加权、情感头领域适配、SAM、参数平均，最佳valid仍为68.2692%，未严格超过70%。按用户五轮上限停止，保留原主模型。详见 [五轮优化记录](problem2/五轮优化记录.md)。此前误将valid作为验收门槛；现按用户明确的test目标验收，已达到70%。参考仓库核对见 [对照分析](problem2/参考仓库对照分析.md)。


问题二专项交付：[附件3预测CSV](deliverables/附件3_预测结果.csv)、[问题二模型与预测材料包](deliverables/问题二_局部缺失模型与预测.zip)。生成与校验：`python -m problem2.package_submission`。该包复用现有赛题数据和预训练资源，不是三问完整离线包。
