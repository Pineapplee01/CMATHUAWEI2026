# 问题2：条件补全与情感预测

建议先阅读 [问题二建模思路说明](问题二建模思路说明.md)：包含模型架构图、逐层张量维度、模块连接、公式推导与示例；先从架构图阅读，再核对实验结果。

> 当前问题二采用 `configs/problem2_robust.json` 和 `AAAmodel/checkpoints/problem2_robust.pt`。完整的局部缺失定义、公式、168组验证、同条件消融、输出一致性及附件3提交见[局部缺失建模与题意核对](局部缺失建模与题意核对.md)。附件3正式CSV为 `results/robust_selected/附件3_预测结果.csv`。后面的70.70%为原完整观测模型历史结果。

> 75%目标进展（2026-09-24）：三条冲击路径——级联中性检测、DeBERTa/cardiffnlp先验融合、弱极性软标签重训——全部实测为负结果，均未读test；完整观测主模型保持 main.pt，test 514/727=70.7015%。试验代码已归档至 `archives/cascade_softlabel_trial/`，证据与结论见 [级联试验记录](级联试验记录.md) 与 [优化空间分析与75%路径](优化空间分析与75%路径.md)。

完整建模说明见 [模型方法与公式推导](模型方法与公式推导.md)：包含符号、逐步推导、算法流程、公式与函数对照、检验和适用边界。

## 正式代码（2026-09-24 清理后，仅保留主链路）

| 模块 | 职责 |
|---|---|
| problem2.py | 训练、验证、专项推理任务编排 |
| runtime.py | 数据加载和模型参数保存/恢复 |
| training.py / training_step.py | 联合训练、早停与验证选模；单批次联合目标 |
| text_encoder.py / sentiment_encoder.py | 掩码缓存、保留BERT词元重分词、三分类句级编码；PrecomputedTextEncoder 消费预处理npz固化的XT嵌入 |
| adapters.py / averaging.py | 骨干低秩适配（LoRA）；可选EMA |
| completion.py / temporal.py | 跨模态条件注意力补全；可选时序读出 |
| heads.py | 加性预测头及贡献账本 |
| model.py | 观测、补全、可靠性与预测的完整前向流程 |
| losses.py | 分类、回归、补全重建和一致性损失（含可选有序/混叠/一致性分支） |
| validation.py / evaluation.py | 完整/连续缺失场景及可靠性分组；指标、视频分组置信区间和图表 |
| inference.py | 附件3全量预测、缺失区间审计和提交manifest |
| local_missingness.py / local_evaluation.py | 严格连续局部缺失协议与冻结模型验证 |
| local_report.py | 类型/位置/相对时长规律、随机mask变化和图表 |
| output_contract.py / verify_submission.py | 极性/强度一致输出及独立提交核验 |
| package_submission.py | 专项材料包生成与50MB/身份路径检查 |
| verify_results.py | 对照原始标签逐ID复核已保存预测 |

当前流程：`python problem2/problem2.py predict --config configs/problem2_robust.json`；复核使用 `python -m problem2.verify_submission`。历史完整观测路线训练复现：`python problem2/problem2.py train --config configs/final.json`（`configs/robust_no_reliability.json` 可复现消融训练）。训练参数只来自train，valid用于选择，test只在最终冻结后检验。

## 历史与清理说明

- 历史完整观测配置为 `configs/final.json`：三分类文本先验加多模态残差、低秩适配、条件补全与可靠性门控；未启用Mixup/一致性/EMA/三阶交互（保留为独立实验选项）。完整观测valid 68.2692%，test 70.7015%，超过70%门槛；主模型权重 SHA256 `cb759c0ac080e75888378b404e6ce749b89daea3b24141a7b60439b8a3b039ab`。
- 五轮优化（R-Drop/有序损失/领域头/SAM/参数平均）与五组全局融合试验均未超过主模型；注意力结构对照66.3462%已回退。上述未采用分支及本轮级联/软标签/各探针脚本已从正式代码移除，快照与配置在 `archives/`（five_rounds、global_fusion_trial、attention_trial、cascade_softlabel_trial），历史实验配置移入 `configs/archive/`，负结果记录在 `results/` 与各试验记录文档。
- 验证集偏置拟合（原 calibration.py）不作为独立泛化证据，脚本已归档。逐ID复核：`python -m problem2.verify_results --config configs/final.json --split test`。

## 标准化预处理数据线（2026-09-24 接入）

`AAAdata/Appendix_2/标准化/对齐版本/` 的 npz 预处理产物（XT 冻结BERT嵌入 + z-score 音视频 + 统一掩码）经 `convert_npz_dataset.py` 转换后由 `encoder_type: precomputed` 直接消费。文本骨干两版：bert-base-uncased（`processed/`）与 cardiffnlp 情感模型（`processed_sentiment/`，由 `预处理/encode_sentiment.py` 生成，CLS+末3维情感logits 广播）。test 完整观测 66.99% / **69.33%**（后者），30% 缺失 65.47% / **68.91%**（退化仅 −0.4pt，为全部模型中最稳）；对照原管道 main 70.70% / 65.06%。 configs: `precomputed_main.json`、`precomputed_sentiment.json`。
