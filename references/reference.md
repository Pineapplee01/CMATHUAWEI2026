# E题参考方法与论文汇总

> 整理及核验日期：2026-09-23。面向“复杂场景下多模态情感预测的数学建模与算法设计”。
>
> 本文汇总最初本地保存的4套参考实现，以及本次论文阅读和网络检索所得方法：18篇核心文献逐篇总结，14篇补充线索作摘要级归纳，并记录2018年数据集论文中的经典基线。后续新增源码下载、许可与逐篇可用性见第11节。
>
> **来源、核验深度、性能结论分开记录。** 被赛题列为参考文献不等于被指定为必用模型，也不等于当前所有协议下的SOTA。以下方法未在本项目中完成训练复现。

## 1. 阅读方式与任务边界

- **问题1（Q1）**：附件1的100条视频，完成特征提取、时序对齐和原始素材映射。下列多数模型直接接收特征，不能代替这一整条处理流程。
- **问题2（Q2）**：附件2训练、验证，研究连续时间段的局部模态缺失；附件3仅作最终无标签推理。整模态缺失、独立随机位置缺失、连续区间缺失是三种不同设定。
- **问题3（Q3）**：同时给出极性、强度、主要参考模态、模态贡献和局部关键证据，并能回映原始素材。路由权重、注意力、不确定性和因果效应不能互相替代。
- 统一术语以 [UBIQUITOUS_LANGUAGE.md](../UBIQUITOUS_LANGUAGE.md) 为准。原论文题名中的 sentiment / emotion 保留原写法；具体目标以论文实验为准。
- 公开MOSEI实验常使用约22,856条处理后样本；本题附件2只有4,850条。二分类Acc2、七级Acc7也不同于本题负向／中性／正向三分类，公开分数不能直接写成我们的实验结果。
- “已核验”表示查到论文正文、正式页面或本地代码；“适配建议”是工程判断；“待核验”表示证据不足，不能据此锁定实验设计。

## 2. 来源与本地保存情况

### 2.1 赛题参考文献与本地方法对应

来源为用户提供的E题DOCX末尾“参考文献”。编号沿用题面，不是本文新增编号。

| 题面编号 | 方法／论文 | 本地实现 | 当前认识 |
| --- | --- | --- | --- |
| [2] | Beyond Missing Modalities… | [HyperDiff/](HyperDiff/) | 正式论文摘要将框架称为HyperEF；目录及仓库名为HyperDiff，使用时注明两种名称 |
| [4] | CMAD | [CMAD/](CMAD/) | 官方参考文献中的缺失模态蒸馏方法 |
| [5] | Proxy-Driven Robust Multimodal Sentiment Analysis with Incomplete Data | [P-RMF/](P-RMF/) | 官方参考文献中的不完整输入融合方法 |
| [7] | EMOE | [EMOE/](EMOE/) | 官方参考文献中的动态模态专家方法 |
| [6] | Learning Invariant Modality Representation…（CmIR） | 作者仓库仅占位，未下载 | 本次补充查阅，同时也是题面参考文献 |
| [9] | CaReFlow | [CaReFlow/](CaReFlow/) | 已下载源码，同时也是题面参考文献 |

本地四套实现来自用户按赛题参考索引保存的资料。论文身份与赛题对应已核对，但尚未逐一核对本地快照和上游Git提交的一致性。第三方代码目录按当前Git规则忽略；本汇总文档独立跟踪。表中的目录链接用于本地浏览，远程仓库可能不包含这些目录。

题面其余[1]、[3]、[8]、[10]不在“本地四套实现＋本次已核验核心方法”内。尤其[3]的题名 *Factorize, Reconstruct, Enhance*，不能仅凭名称相近就与TSD、CACR等方法认定为同一篇；其精确出版条目尚待独立核对。

### 2.2 核心方法索引

下表“用途”表示可作为哪类对照的候选，不代表已满足该问题全部要求。

| 编号 | 方法 | 年份／来源 | 机制概括 | 建议用途 | 核验深度 |
| --- | --- | --- | --- | --- | --- |
| R01 | CMAD | ICCV 2025；题面[4] | 教师—学生相关性蒸馏与模态组合加权 | Q2 | 正式摘要、本地代码 |
| R02 | P-RMF | ACL 2025；题面[5] | 高斯潜在表示、不确定性代理融合 | Q2 | 正式摘要、本地代码 |
| R03 | EMOE | CVPR 2025；题面[7] | 动态模态专家与单模态蒸馏 | Q2基础预测、Q3 | 正式摘要、本地代码 |
| R04 | HyperEF／HyperDiff | CVPR 2026；题面[2] | 掩码超图条件扩散、双通道证据融合 | Q2高成本对照 | 正式摘要、本地代码 |
| R05 | Graph-MFN／DFG | ACL 2018；用户提供PDF | 可解释动态图融合与门控记忆 | 经典预测基线、Q3 | 正文、表3及图示 |
| R06 | CICA | CVPR 2026 | 置信度感知预训练与注意力耦合 | Q2、Q3 | 正文、表2 |
| R07 | CaReFlow | CVPR 2026；题面[9] | 循环自适应整流流分布映射 | 融合强基线 | 正文／结果表、作者仓库 |
| R08 | CmIR | ACL 2026；题面[6] | 因果不变表示解耦 | Q2泛化／噪声对照 | 正文、表1、作者仓库 |
| R09 | TSD | CVPR 2026 | 三子空间解耦与子空间感知交叉注意力 | 融合强基线 | 正文、表1 |
| R10 | QA-MoE | ACL 2026 | 可靠性估计引导专家路由 | Q2 | 正式摘要、正文及协议 |
| R11 | DNRNet | CVPR 2026 | 局部—全局嵌套循环补全 | Q2 | 正式摘要 |
| R12 | CACR | CVPR 2026 | 冲突感知自适应交叉重构 | 融合／冲突对照 | 正文、表2 |
| R13 | MMRest | CVPR 2026 | 聚类基础上的全局与局部多度量学习 | 细粒度强度预测 | 正式摘要 |
| R14 | EBMC | CVPR 2026 | 弱模态增强、平衡优化、可信蒸馏 | Q2、Q3 | 正式摘要 |
| R15 | PaP | CVPR 2026 | 情感原型作为LLM软提示 | 大模型预测对照 | 正文、表1 |
| R16 | MoB | arXiv，2026-09 | 极性／强度分解与信息瓶颈专家 | 双目标预测 | 正文、表I |
| R17 | SeRIn | arXiv，2026-07 | 分离单模态精炼和跨模态融合 | 融合强基线 | 正文、表1、代码链接 |
| R18 | MAESTRO | arXiv，2026-08 | 文本路由专家与有序原型优化 | 极性／强度预测 | 正文及方法部分 |

## 3. 本地保存的四个赛题参考方法

### R01. CMAD

- **论文**：*CMAD: Correlation-Aware and Modalities-Aware Distillation for Multimodal Sentiment Analysis with Missing Modalities*，ICCV 2025，题面[4]。
- **链接**：[正式论文](https://openaccess.thecvf.com/content/ICCV2025/html/Zhuang_CMAD_Correlation-Aware_and_Modalities-Aware_Distillation_for_Multimodal_Sentiment_Analysis_with_ICCV_2025_paper.html)；[作者代码](https://github.com/YetZzzzzz/CMAD)；[本地说明](CMAD/README.md)。
- **研究问题**：缺失模态使完整输入与不完整输入的表示不一致，同时不同缺失组合的学习难度不同。
- **方法总结**：完整模态教师指导缺失模态学生。相关性感知特征蒸馏（CAFD）同时保持教师—学生的特征相似性和高阶相关结构；模态感知正则化（MAR）按照模态组合的难度自适应加权，稳定训练过程。
- **与E题关联**：可以比较“完整信息教师能否提升局部缺失预测”，适合Q2的蒸馏路线基线；其蒸馏机制不直接提供Q3要求的时间证据。
- **本地代码事实**：[modality_drop.py](CMAD/CMAD_sentiment/Student_Model/modality_drop.py)采用七种模态可用组合，按整个模态序列置零；[stu_config_mosei.py](CMAD/CMAD_sentiment/Student_Model/stu_config_mosei.py)读取train/dev/test而非本题train/valid/test。
- **适配要求**：重新建立附件2加载器；教师与学生均使用题目允许的数据训练；区分原整模态缺失训练与连续区间缺失适配版；统一中性类别与回归指标。不能将整模态组合权重直接当作局部时间掩码。

### R02. P-RMF

- **论文**：*Proxy-Driven Robust Multimodal Sentiment Analysis with Incomplete Data*，ACL 2025，题面[5]。
- **链接**：[正式论文](https://aclanthology.org/2025.acl-long.1075/)；[检索到的可访问仓库](https://github.com/hawksilent/P-RMF)；[本地说明](P-RMF/README.md)。论文原文声明的GitHub入口为 `aoqzhu/P-RMF`，2026-09-23检查返回404；与当前可访问仓库的迁移或同源关系未核对，不据此认定本地快照的上游提交。
- **研究问题**：缺失程度不确定时，单模态与融合表示容易偏移，直接依赖不可靠输入会损害预测。
- **方法总结**：将单模态输入映射为高斯潜在分布，学习稳定表示并量化内在不确定性；基于不确定性构建联合的代理模态，再通过多层动态跨模态注入增强其信息多样性。本地实现还使用表示重构与KL相关约束。
- **与E题关联**：与Q2“不完整输入下的强度预测”接近，是优先复现候选。
- **本地代码事实**：[dataset.py](P-RMF/core/dataset.py)的generate_m按位置独立随机缺失，而非连续区间；文本缺失使用UNK等标记，不能与数值全零缺失无条件等同。[train_mosei.yaml](P-RMF/configs/train_mosei.yaml)采用非对齐50/500/500序列和单一回归输出；推理路径允许完整输入为None。
- **适配要求**：对连续缺失另建可复现协议；统一文本编码、长度、填充和观测掩码；增加三分类决策规则或明确标注分类头改动；修正测试集选优行为，见第8节。

### R03. EMOE

- **论文**：*EMOE: Modality-Specific Enhanced Dynamic Emotion Experts*，CVPR 2025，题面[7]。
- **链接**：[正式论文](https://openaccess.thecvf.com/content/CVPR2025/html/Fang_EMOE_Modality-Specific_Enhanced_Dynamic_Emotion_Experts_CVPR_2025_paper.html)；[作者代码](https://github.com/fuyyyyy/EMOE)；[本地说明](EMOE/README.md)。
- **研究问题**：不同样本依赖不同模态，固定融合难以平衡各模态，同时融合可能削弱单模态自身的预测能力。
- **方法总结**：模态专家混合为每个样本学习不同的模态权重；单模态蒸馏保留各模态预测信息，使融合表示兼顾共享信息与模态特异性。
- **与E题关联**：可作为完整输入下的动态融合基线，并提供Q3的模态作用分析起点；不能仅凭三个权重完成局部证据定位。
- **本地代码事实**：[emoe.py](EMOE/trains/singleTask/model/emoe.py)返回channel_weight，输出头为单值回归。路由器先展平三模态，再经过大线性层。
- **成本提示**：按本题50步、768/74/35维代入[router.py](EMOE/trains/singleTask/model/router.py)，仅路由器参数量为240,363,777，FP32约916.9 MiB。这是按结构计算的估计，不是实测训练显存。
- **适配要求**：补充三分类评价、缺失掩码、局部归因及时间映射。若改成池化后路由，须命名为适配／轻量版，不能将结构改动后的结果冒充原版EMOE。

### R04. HyperEF（本地目录／代码仓库名：HyperDiff）

- **论文**：*Beyond Missing Modalities: Hypergraph Conditioned Diffusion for Uncertainty-Aware Multimodal Emotion Recognition*，CVPR 2026，题面[2]。
- **链接**：[正式论文](https://openaccess.thecvf.com/content/CVPR2026/html/Qiu_Beyond_Missing_Modalities_Hypergraph_Conditioned_Diffusion_for_Uncertainty-Aware_Multimodal_Emotion_CVPR_2026_paper.html)；[代码仓库](https://github.com/wdqdp/HyperDiff)；[本地目录](HyperDiff/)。
- **名称说明**：正式论文摘要称框架为Hypergraph Diffusion and Evidence Fusion based Emotion Recognition（HyperEF），本地及检索到的仓库名为HyperDiff。这里按完整题名关联，不把二者重复计为两篇。
- **研究问题**：对话中的模态缺失会破坏语义一致性，恢复特征又可能引入新的不确定性。
- **方法总结**：以掩码超图注意力（MHGAT）捕捉可用模态的高阶语义关系，作为扩散模型恢复缺失潜在特征的条件；再通过双通道证据融合（DCEF）分别考虑特征来源与判别层面的不确定性。
- **与E题关联**：可作为Q2生成式恢复的高成本对照，也可参考其不确定性分析；对话／话语层级恢复不等同于片段内部连续缺失恢复。
- **本地代码事实**：[dataloader_cmumosi.py](HyperDiff/dataloader_cmumosi.py)按话语读取独立特征并聚合为对话上下文；[HyperDiff.py](HyperDiff/trains/singleTask/HyperDiff.py)生成每个话语的三模态可用掩码。存在多阶段训练和硬编码路径。
- **适配要求**：重做输入适配与局部掩码，检查无标签推理路径和连续强度输出；统一验证集选优。当前本地README内容不足以支持完整复现说明，不能仅凭目录名认定已复现论文。

## 4. 数据集原论文与经典基线

### R05. Graph-MFN / Dynamic Fusion Graph（DFG）

- **论文**：*Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph*，ACL 2018。
- **链接**：[正式论文](https://aclanthology.org/P18-1208/)；[PDF](https://aclanthology.org/P18-1208.pdf)。本次同时阅读了用户提供的本地PDF。
- **研究问题**：建立大规模多模态情感／情绪数据集，并研究模态之间随时间变化的交互关系。
- **方法总结**：三条LSTM分别建模文本、视觉、语音；DFG用带可学习efficacy的连接组织单模态、双模态和三模态交互，再与MFN中的多视图门控记忆结合，得到Graph-MFN。动态连接可用于可视化模型在不同时间偏重哪些交互。
- **与E题关联**：原始特征及对齐说明对Q1有参考价值；Graph-MFN是经典预测基线，DFG是Q3动态模态交互的直接参考，但连接强度不等于已验证的因果贡献。
- **原文证据**：表3（印刷页2242）报告Graph-MFN的A2=76.9%、F1=77.0%、MAE=0.71、r=0.54；这些是2018年的原始实验口径，不能与更换特征后的现代结果机械比较。
- **特别说明**：原文表中SOTA1和SOTA2指当时各指标最优、次优的已有方法，并非两个方法名。

### 原论文列出的主要经典比较方法

本小节是对R05第2.2节的二级归纳，没有逐篇完成原始方法论文精读，不将它们称为本次发现的新SOTA。

| 方法 | 方法概要 | 可作为本题什么对照 |
| --- | --- | --- |
| EF-LSTM | 每个时间步拼接多模态输入，再交给单个LSTM | 最直接的早期融合基线；并非本次识别的一篇独立新论文 |
| MV-LSTM | 在LSTM内部划分不同视图的表示区域 | 时序联合表示基线 |
| TFN | 通过张量外积显式组织单、双、三模态交互 | 高阶融合基线，需控制参数量 |
| MFN | 模态内LSTM、跨模态变化注意力和多视图门控记忆 | Graph-MFN的基础结构与消融参照 |
| MARN | 用多个注意力系数捕捉多种跨模态动态，并结合时序记忆 | 多重跨模态交互基线 |

R05还列出BC-LSTM、C-MKL、DF、SVM、RF、THMM、SAL-CNN、3D-CNN，以及语言／声学单模态模型。这里只登记其出现于原论文的事实；如实际纳入实验，应继续核对对应原文与实现。简单文本单模态、三模态拼接、晚期融合仍应保留，不应只比较复杂模型。

## 5. 本次核验的近期已发表方法

### R06. CICA

- **论文**：*CICA: Coupling Confidence-Aware Pretraining with Confidence-Informed Attention for Robust Multimodal Sentiment Analysis*，CVPR 2026。
- **链接**：[正式论文](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_CICA_Coupling_Confidence-Aware_Pretraining_with_Confidence-Informed_Attention_for_Robust_Multimodal_CVPR_2026_paper.html)。未核实可用的作者代码入口。
- **方法总结**：先做置信度感知预训练（CAP），让模态编码器同时输出表示与可靠性；再以置信度引导融合注意力（CIF），减弱噪声或冲突输入的影响。融合训练同时保持与各单模态表示的关联，降低模态塌缩。
- **与E题关联**：适合Q2的质量感知鲁棒融合，也可启发Q3模态作用分析；仍需局部归因与证据回映。
- **证据与限制**：已核对正文表2。MOSEI报告MAE=0.489、Corr=0.856、Acc7=55.29%；Acc2为84.72/90.18、F1为85.15/90.16（依次为has-0/non-0）。这是作者结果；不是本项目复现，也不是赛题三分类成绩。
- **适配要求**：核对CAP使用的数据和损坏构造方式，保证仅使用题目允许的训练数据。完整输入的高成绩不能替代连续缺失验证。

### R07. CaReFlow

- **论文**：*CaReFlow: Cyclic Adaptive Rectified Flow for Multimodal Fusion*，CVPR 2026，题面[9]。
- **链接**：[正式论文](https://openaccess.thecvf.com/content/CVPR2026/html/Mai_CaReFlow_Cyclic_Adaptive_Rectified_Flow_for_Multimodal_Fusion_CVPR_2026_paper.html)；[预印本](https://arxiv.org/abs/2602.19140)；[作者代码](https://github.com/TmacMai/CaReFlow)。
- **方法总结**：通过整流流将音频／视觉分布映射到语言表示分布。采用跨样本的一对多映射观察目标分布，对同一样本施加更严格对齐、对其他配对施加放松对齐；再使用反向循环映射降低信息丢失。
- **与E题关联**：可作为强融合对照。这里的“分布对齐”不同于Q1需要的真实时间戳对齐，不能拿其中一个替代另一个。
- **证据与限制**：结果表报告MOSEI Acc2=87.9%、F1=88.0%、MAE=0.504、Corr=0.799；作者仓库使用DeBERTa-v3-base，并有train/dev/test格式假设。
- **适配要求**：统一文本编码与数据接口；若固定为附件2预计算文本特征，需明确这是特征受控的适配版本。

### R08. CmIR

- **论文**：*Learning Invariant Modality Representation for Robust Multimodal Learning from a Causal Inference Perspective*，ACL 2026，题面[6]。
- **链接**：[正式论文](https://aclanthology.org/2026.acl-long.2119/)；[作者占位仓库](https://github.com/TmacMai/CmIR)。2026-09-23检查只有README，无模型源码，未下载为可复现实现。
- **方法总结**：将模态表示解耦为因果不变部分和环境相关的伪相关部分，结合不变性、互信息及重构约束，尝试在分布偏移和噪声条件下保持与标签稳定相关的信息。
- **与E题关联**：适合Q2的噪声／泛化鲁棒性对照；可启发Q3对伪相关的分析，但“采用因果建模”不等于每个解释片段都具有可识别的因果效应。
- **证据与限制**：已核对表1：MOSEI Acc2=87.8%、F1=87.7%、MAE=0.513、Corr=0.793。分布外与噪声实验不能直接代表本题连续区间缺失。
- **适配要求**：核对环境构造与不变性假设，不能在本题训练／验证集外引入额外情感数据。

### R09. TSD

- **论文**：*Tri-Subspaces Disentanglement for Multimodal Sentiment Analysis*，CVPR 2026。
- **链接**：[正式论文](https://openaccess.thecvf.com/content/CVPR2026/html/Meng_Tri-Subspaces_Disentanglement_for_Multimodal_Sentiment_Analysis_CVPR_2026_paper.html)；[预印本](https://arxiv.org/abs/2602.19585)。未核实作者代码入口。
- **方法总结**：将表示分成三类：三模态共同共享、仅在特定模态对之间共享、单模态私有。通过解耦监督与结构正则保持子空间区分，再用子空间感知交叉注意力（SACA）融合。
- **与E题关联**：可比较更细的共享／私有结构是否优于EMOE等方法，也适合研究三模态互补性；不是原生证据定位方案。
- **证据与限制**：表1同时报告aligned/unaligned。MOSEI MAE分别为0.529/0.525，Acc7为54.9/54.6，Acc2为85.8/86.2；不得调换两种设置。
- **适配要求**：增加缺失掩码和统一三分类评价；避免将子空间结构直接解释成可观测时间片段。

### R10. QA-MoE

- **论文**：*QA-MoE: Towards a Continuous Reliability Spectrum with Quality-Aware Mixture of Experts for Robust Multimodal Sentiment Analysis*，ACL 2026。
- **链接**：[正式论文](https://aclanthology.org/2026.acl-long.1461/)；[论文对应仓库](https://github.com/U235-Aurora/QA-MoE)；[已下载源码](QA-MoE/)。README引用完整题名和arXiv编号；不是同名临床多模态仓库。许可说明见第11节。
- **方法总结**：用连续可靠性谱统一描述模态缺失与质量下降，通过自监督的偶然不确定性估计引导专家路由，抑制不可靠信号的误差传播。论文讨论一个检查点覆盖多种退化场景的能力。
- **与E题关联**：与Q2自适应利用残存信息的目标一致，对Q3也提供可靠性分析思路。
- **证据与限制**：正文区分干净数据比较、固定模态缺失与其他退化协议；不能把其中某个协议的优势泛化到所有缺失场景。
- **适配要求**：明确可靠性估计所需输入、训练增强与测试条件，并专门验证连续区间缺失。

### R11. DNRNet

- **论文**：*Active Perceptual Inference: A Corticothalamic-Inspired Dynamic Nested Recurrent Network for Multimodal Sentiment Analysis with Incomplete Data*，CVPR 2026。
- **链接**：[正式论文](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Active_Perceptual_Inference_A_Corticothalamic-Inspired_Dynamic_Nested_Recurrent_Network_for_CVPR_2026_paper.html)。未核实作者代码入口。
- **方法总结**：以动态嵌套循环替代一次性补全。局部循环进行单模态模式补全，全局循环估计模态置信度并整合跨模态校正，再将两类校正用于下一轮输入更新。
- **与E题关联**：可与P-RMF比较一次代理构造和迭代修正的差异，适合Q2恢复路线。
- **证据与限制**：本次核验正式摘要，原文描述的是随机帧级缺失；尚未逐项核对完整结果表与训练代码。
- **适配要求**：增加连续区间测试，控制迭代次数、推理耗时，并确保未缺失区间不被无约束替换。

### R12. CACR

- **论文**：*Conflict-Aware Adaptive Cross-Reconstruction for Multimodal Sentiment Analysis*，CVPR 2026。
- **链接**：[正式论文](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Conflict-Aware_Adaptive_Cross-Reconstruction_for_Multimodal_Sentiment_Analysis_CVPR_2026_paper.html)；[作者声明的官方代码](https://github.com/wwwwwyyyy559/Conflict-Aware-Adaptive-Cross-Reconstruction-for-Multimodal-Sentiment-Analysis)；[已下载源码](CACR/)。
- **方法总结**：显式计算模态情感冲突分数并转换为跨重构权重；用目标模态的私有特征与其他模态的共享特征重构目标，降低冲突输入对共享语义的扭曲；再提取音视频特异的细粒度线索补充文本。
- **与E题关联**：可检验模态相互矛盾时的预测稳定性；与Q2“信息缺失”有关但并非同一问题，对Q3错误归因也有价值。
- **证据与限制**：正文采用BERT 768维、Facet 35维、COVAREP 74维；MOSEI表2报告Acc2=85.37%、MAE=0.532。维度接近附件2，但不能因此断言逐维语义完全一致。
- **适配要求**：单独区分冲突与缺失，不把检测到的冲突直接当作传感器故障或因果解释。

### R13. MMRest

- **论文**：*Multi-Metric Representation Learning Strategy Based on Clustering for Fine-Grained Multimodal Sentiment Analysis*，CVPR 2026。
- **链接**：[正式论文](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Multi-Metric_Representation_Learning_Strategy_Based_on_Clustering_for_Fine-Grained_Multimodal_CVPR_2026_paper.html)；[作者声明的官方代码](https://github.com/hurriedpi/MMRest)；[已下载源码](MMRest/)。
- **方法总结**：针对不同情感类别中心重叠，利用聚类同时学习全局度量与局部度量，让相似情感接近、不同情感远离；投影与决策层融合（PDLF）将度量投影结果和单／多模态融合分数结合，得到最终预测。
- **与E题关联**：可用于连续强度和细粒度情感边界的对照；不直接解决局部缺失或原始素材定位。
- **证据与限制**：本次为正式摘要级核验，不以摘要中的优越性声明认定其在统一协议下排名第一。
- **适配要求**：聚类、度量与任何统计参数只在训练数据拟合，避免验证／测试信息进入表示学习。

### R14. EBMC

- **论文**：*Enhance-then-Balance Modality Collaboration for Robust Multimodal Sentiment Analysis*，CVPR 2026。
- **链接**：[正式论文](https://openaccess.thecvf.com/content/CVPR2026/html/He_Enhance-then-Balance_Modality_Collaboration_for_Robust_Multimodal_Sentiment_Analysis_CVPR_2026_paper.html)；[预印本](https://arxiv.org/abs/2604.12518)；[代码仓库](https://github.com/kangverse/EBMC)；[已下载源码](EBMC/)。使用CC-BY-NC-4.0非商业许可。
- **方法总结**：先通过模态语义解耦（MSD）与跨模态互补增强（CCE）强化弱模态，再用能量引导的模态协调（EMC）缓解训练竞争；实例级可信蒸馏（IMTD）估计样本级可靠性并调节融合。
- **与E题关联**：适合Q2弱模态、噪声和缺失场景，也可与EMOE比较不同动态融合机制。
- **证据与限制**：本次依据正式摘要归纳，缺失生成方式、损失细节和开销尚待正文／代码精读。
- **适配要求**：把“弱模态增强”和“可靠性路由”的作用拆开消融，不能仅展示权重变化作为有效性证据。

### R15. PaP

- **论文**：*Prototype-as-Prompt: Multimodal Sentiment Prototypes Endowing Large Language Models the Capability to Perform Multimodal Sentiment Analysis*，CVPR 2026。
- **链接**：[正式论文](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_Prototype-as-Prompt_Multimodal_Sentiment_Prototypes_Endowing_Large_Language_Models_the_Capability_CVPR_2026_paper.html)。未核实作者代码入口。
- **方法总结**：将音视频等特征压缩为具有明确情感语义的原型，并作为LLM软提示。通过情感监督、跨模态原型对齐和距离加权的多样性约束，使原型既语义一致又彼此可区分。
- **与E题关联**：可作为大语言模型融合路线的对照；原型分布可辅助Q3展示，但不自带词／秒／帧级证据映射。
- **证据与限制**：已核对MOSEI表1。LLaMA-2-7B版本报告Acc2=87.17%、MAE=0.493；不同LLM版本结果不能混为一行。
- **适配要求**：评估GPU和交付体积；参数高效微调不等于总模型小，也不自动满足50MB附件限制。

## 6. 已查阅正文的近期预印本候选

以下发表状态按本次检索证据记录为预印本；没有核验到正式会议信息时，不推断已录用。

### R16. MoB

- **论文**：*Divide and Conquer: Mixture-of-Bottleneck Experts in Informative Ordinal Space for Video-based Multimodal Sentiment Analysis*，arXiv，2026-09。
- **链接**：[论文](https://arxiv.org/abs/2609.18470)；[HTML全文与表I](https://arxiv.org/html/2609.18470v1)。未核实作者代码入口。
- **方法总结**：将情感预测视为有序问题，拆成极性识别与强度预测；为不同模态与子任务构造信息瓶颈专家，保留任务相关信息并去除冗余，再通过瓶颈路由及困难样本挖掘完成融合。
- **与E题关联**：与同时输出极性和强度的要求直接相关，也值得研究任务间信息共享；并非已经证明适用于局部缺失。
- **证据与限制**：表I中DeBERTaV3版本MOSEI Acc2=86.7/88.5（has-0/non-0）、MAE=0.496、Corr=0.821；音视频使用ImageBind提取，不能当成固定附件2特征下的成绩。
- **适配要求**：若改用赛题预计算特征，需说明改动；BERT、RoBERTa、DeBERTaV3结果按骨干分开报告。

### R17. SeRIn

- **论文**：*Segregate, Refine, Integrate: Decomposing Multimodal Fusion for Sentiment Analysis*，arXiv，2026-07。
- **链接**：[论文](https://arxiv.org/abs/2607.12686)；[HTML全文](https://arxiv.org/html/2607.12686v1)；[作者占位仓库](https://github.com/alexisfilippakopoulos/SeRIn-MSA)。2026-09-23检查只有README，无模型源码。
- **方法总结**：让模态特异信息沿相互隔离的路径演化，各自对编码器上下文做精炼；独立跨模态路径收集联合信息，在最终预测阶段进行充分交互，减少过早融合造成的单模态信息污染。
- **与E题关联**：适合完整输入融合和模态干扰的比较，可研究其门控对视觉退化的响应；仍需额外实现时间证据输出。
- **证据与限制**：表1报告MOSEI Acc2=87.79%、F1=87.78%、MAE=0.493、Corr=0.810、Acc7=56.52，论文称五次运行平均。其MOSEI实验使用GPT2-large相关配置，不能与BERT特征方法无条件横比。
- **适配要求**：控制骨干与参数预算，区分“结构化交互的收益”与“模型容量的收益”。

### R18. MAESTRO

- **论文**：*Multimodal Adaptive Expert Selection with Text Routing and Ordinal Prototype Optimization for Sentiment Analysis*，arXiv，2026-08。
- **链接**：[论文](https://arxiv.org/abs/2608.30726)。未核实作者代码入口。
- **方法总结**：用文本语境作为路由信号，动态激活音视频专家；通过有序原型对比学习，在潜在空间保留情感强度的顺序与距离关系，避免普通对比目标忽略强度层级。
- **与E题关联**：可作为动态专家与有序强度建模的补充候选，与EMOE、MoB形成不同机制的对照。
- **证据与限制**：已查阅方法和实验说明，但未完成统一协议审计与代码复现。其“优于SOTA”是作者对所选比较集合的结论，不能提升为全局最优认定。
- **适配要求**：在同一缺失协议下比较；额外检查中性边界与类别不平衡，避免强度排序收益掩盖中性类退化。

## 7. 检索中发现的其他方法与诊断文献

以下14篇也在本次检索中出现。**摘要级核验**：已读取arXiv检索页的题名和摘要，未完成逐篇正文、代码与结果表审计。下面只总结作者描述的机制与潜在用途，不把它们列入已确认的SOTA排名。月份对应arXiv编号月份，不代替正式出版时间；摘要中的录用声明未在此单独认证。

### S01. 语义完整度引导重构

- **论文**：*Robust Multimodal Sentiment Analysis with Incomplete Modalities via Semantic-aware Completeness based Reconstruction*（2026-09）。
- **链接**：[论文](https://arxiv.org/abs/2609.10950)。
- **方法总结**：估计不完整输入保留了多少与情感相关的信息，用这一完整度引导缺失语义重构，并设计训练策略协调完整度估计和情感预测。
- **本题价值／限制**：Q2候选，关注“缺多少语义”而不只是“缺多少位置”；完整度定义与监督来源仍需核对。

### S02. VG-TPT

- **论文**：*Vision-Guided Text Prompt Tuning for Multimodal Sentiment Analysis*（2026-09）。
- **链接**：[论文](https://arxiv.org/abs/2609.06497)；[论文指向的仓库](https://github.com/ma-tubu/VG-TPT)；[已下载源码](VG-TPT/)。
- **方法总结**：冻结BERT，通过视觉引导的逐层自适应提示校准文本表示；路由器依据文本状态和视觉信息组合提示库，减少全量微调。
- **本题价值／限制**：可参考轻量适配和文本—视觉交互；主要是双模态路线，不能直接覆盖三模态要求。少量可训练参数不代表小的完整模型。

### S03. 迭代代理校正

- **论文**：*Robust Incomplete Multimodal Sentiment Analysis via Iterative Proxy Correction*（2026-08）。
- **链接**：[论文](https://arxiv.org/abs/2608.19971)。
- **方法总结**：由非语言模态构造语言代理，通过门控残差逐步修正，再按语言可靠性融合代理和观测文本；完整语言表示仅作为训练时的语义锚点，稳定分阶段校正。
- **本题价值／限制**：可与P-RMF比较一次性代理和迭代代理；需核对无标签推理是否完全不依赖完整输入。

### S04. MRCF

- **论文**：*Rethinking Modality Reliability in Multimodal Sentiment Analysis with Incomplete Observations*（2026-08）。
- **链接**：[论文](https://arxiv.org/abs/2608.03611)。
- **方法总结**：通过模态内质量线索与跨模态一致性估计可靠性，用可靠性调控跨模态信息传递，再结合语义进行校准融合，缓解不可靠信号的传播。
- **本题价值／限制**：与Q2动态缺失适应、Q3模态作用分析相关；可靠性分数仍需干预测试，不能只看相关性。

### S05. SentiLLM

- **论文**：*Semantic-Aligned Structural Abstraction for Multimodal Sentiment Analysis*（2026-07）。
- **链接**：[论文](https://arxiv.org/abs/2607.27790)；[论文指向的仓库](https://github.com/especiallyW/SentiLLM)；[已下载源码](SentiLLM/)。
- **方法总结**：把音视频序列分为情感显著变化的焦点流和稳定背景的环境流，通过两者校准形成语义紧凑的表示，再投射为LLM可使用的token。
- **本题价值／限制**：可启发Q3局部变化与背景的区分；需核对其显著性输出是否能忠实地映射真实时间。

### S06. MRUF

- **论文**：*MRUF: Multi-granularity Routing with Uncertainty-Aware Fusion for Robust Multimodal Sentiment Analysis*（2026-07）。
- **链接**：[论文](https://arxiv.org/abs/2607.10599)；[代码仓库](https://github.com/ICIG/MRUF)；[已下载源码](MRUF/)。
- **方法总结**：在子空间和模态两个层级路由，利用移除模态后的误差增量监督模态重要性，再用模态不确定性的逆方差校准门控，并以对比约束稳定共享表示。
- **本题价值／限制**：与Q2可靠性融合、Q3贡献验证相关；训练阶段的重要性监督与推理阶段可获得信息必须分开。

### S07. SHAP加权跨模态专家融合

- **论文**：*SHAP-Weighted Cross-Modal Expert Fusion for Emotion and Sentiment Recognition: Evidence and Limits*（2026-07）。
- **链接**：[论文](https://arxiv.org/abs/2607.08573)。
- **方法总结**：为树模型的单模态及跨模态专家计算TreeSHAP，用归因幅度生成融合权重；对比mean-abs、median-abs、sum-abs聚合，分析不同特征维数对权重的影响。
- **本题价值／限制**：Q3解释方法的诊断参考。作者摘要强调主要增益可能来自增加跨模态专家，而非复杂的样本级路由；不能据此宣称SHAP路由必然更优。

### S08. 可靠性置换诊断

- **论文**：*When Does Quality-Aware Multimodal Fusion Matter? A Leakage-Safe Diagnostic for Decision-Level Dependence*（2026-06）。
- **链接**：[论文](https://arxiv.org/abs/2606.26473)；[对应代码仓库](https://github.com/jadenmoon27/quality-aware-fusion-diagnostic)；[已下载源码](quality-aware-fusion-diagnostic/)。
- **方法总结**：固定训练后的模型与输入，只跨样本打乱可靠性分数，检查预测是否改变，从而测试模型是否真正使用可靠性信息。
- **本题价值／限制**：适合验证Q3的模态贡献／可靠性展示是否有预测作用；属于诊断协议，不是新的预测SOTA。

### S09. 原生全模态LLM的判别式读出

- **论文**：*Beyond Generative Decoding: Discriminative Hidden-State Readout from a Native Omni-Modal LLM for Multimodal Sentiment Analysis*（2026-06）。
- **链接**：[论文](https://arxiv.org/abs/2606.05713)；[对应代码仓库](https://github.com/GawaineXiukkie/DiscRead-MSA)；[已下载源码](DiscRead-MSA/)。
- **方法总结**：以Qwen2.5-Omni-7B为骨干，从最后有效token的隐藏状态直接回归连续分数，使用QLoRA训练；与生成文本分数的读出方式做受控比较。
- **本题价值／限制**：可参考连续预测头设计；使用原始音视频的大模型流程与附件2固定特征流程不同，计算及交付成本显著。

### S10. MCAF

- **论文**：*Dynamic Interaction-Aware and Causality-Disentangled Framework for Multimodal Sentiment Analysis*（2026-05）。
- **链接**：[论文](https://arxiv.org/abs/2605.30994)。
- **方法总结**：先以信息瓶颈和结构因果建模分离文本偏置，在特征、时间和模态层级动态评估互补／冲突／冗余，再用条件扩散净化融合表示。
- **本题价值／限制**：与Q3动态交互分析相关，也可探索Q2噪声抑制；摘要中的因果与解释性声明仍需检查假设、识别条件和干预证据。

### S11. 双层参考对齐控制决策漂移

- **论文**：*Controlling Decision Drift in Multimodal Sentiment Analysis with Missing Modalities*（2026-05）。
- **链接**：[论文](https://arxiv.org/abs/2605.16889)；[论文脚注入口重定向后的TLRA仓库](https://github.com/liuxy1005/TLRA)；[已下载源码](TLRA/)。
- **方法总结**：在表示层用完整模态样本提供稳定参考，在决策层通过原型检索与投票压制不可靠模态，减少不同缺失组合间的预测漂移。
- **本题价值／限制**：适合Q2表示恢复与决策一致性的对照；必须核对原型库仅来自训练集，不能包含专项测试标签或隐含真值。

### S12. DBR

- **论文**：*Mitigating Shared-Private Branch Imbalance via Dual-Branch Rebalancing for Multimodal Sentiment Analysis*（2026-04）。
- **链接**：[论文](https://arxiv.org/abs/2604.25179)。
- **方法总结**：共享分支分离时间变化与结构依赖，私有分支用锚点引导路由保留模态特异性，再通过双向平衡融合合并两类信息。
- **本题价值／限制**：可与TSD、EMOE比较共享／私有信息组织；缺失机制与解释输出尚待精读。

### S13. ProMMA／Prompt-based Missing Modality Adaptation

- **论文**：*Evaluation Before Generation: A Paradigm for Robust Multimodal Sentiment Analysis with Missing Modalities*（2026-04）。
- **链接**：[论文](https://arxiv.org/abs/2604.05558)；[作者占位仓库](https://github.com/rongfei-chen/ProMMA)。2026-09-23检查只有README、LICENSE与.gitignore，未发布模型源码。
- **方法总结**：先估计缺失模态是否值得生成，再进行共享／私有提示解耦、动态提示加权和多层残差连接，降低低质量补全的干扰。ProMMA为摘要中实现仓库所使用的名称。
- **本题价值／限制**：Q2中“是否补全”的策略参考；伪标签、预训练模型及提示机制的来源要按赛题数据边界审核。

### S14. 模态平衡方法的受控评估

- **论文**：*The Illusion of Balanced Multimodal Sentiment Analysis: Beyond the Limits of Optimization-Based Methods*（2026-09）。
- **链接**：[论文](https://arxiv.org/abs/2609.11247)。
- **方法总结**：在统一框架中评估梯度与损失平衡策略，分析拟合速度与判别贡献被混淆的问题，并主张用留出数据评估模态效用。
- **本题价值／限制**：是实验设计与反证参考，不是预测模型。其摘要结论不能外推为所有动态融合都无效，但提示我们必须保留晚期拼接等简单对照。

## 8. 作为E题基线前必须统一的协议

### 8.1 输入、缺失与数据边界

1. 所有建模数据从项目data/读取。参考仓库中的外部数据下载指令不属于本项目数据授权。
2. 参数学习只使用附件2训练集；验证集选择结构、超参数和决策阈值。附件2test固定方案后评估，附件3/4仅用于专项推理。
3. 保持同一特征版本；比较不同骨干、重新提取特征或额外预训练时必须单列，不能将输入差异归因为融合模块优势。
4. 附件3对齐版只有text_bert/audio/vision，非对齐版只有raw_text/audio/vision；两版都没有附件2的预计算text，非对齐版还没有显式音视频长度字段。必须先核对统一编码、长度与掩码，不能假设原加载器可直接使用。
5. 填充掩码与观测掩码分别记录。全零不一定是局部缺失；text_bert分段编号全零也不表示文本丢失。
6. 在统一的连续区间缺失测试上评估所有模型。原生训练策略与使用连续缺失增强后的适配版分别报告，避免把改动混在原方法名称中。

### 8.2 标签、指标与检查点

- 本题真实极性为负向／中性／正向，强度0仅属于中性。Non0_acc_2、Has0_acc_2和取整得到的Mult_acc_3不是同一评价任务。
- 明确F1的平均方式，建议同时记录Macro-F1、Weighted-F1和各类表现；回归记录MAE与Pearson，Pearson在真值或预测为常数时未定义，应明确标记。
- 所有模型在同一验证协议下选检查点。若分类和回归采用不同检查点，必须明确披露，不能将独立最优值包装成同一个提交模型。
- **本地P-RMF的代码事实**：[train.py](P-RMF/train.py)按test结果保存最优检查点，验证结果不保存最优模型。复现本题时必须改为验证集选优。
- **本地HyperDiff的代码事实**：[HyperDiff.py](HyperDiff/trains/singleTask/HyperDiff.py)的do_train_class更新best_test时比较测试集分数。该行为不能直接用于我们的最终评价。
- 上述是对当前本地快照的检查，不据此断言原论文全部实验采用同一有问题的实现。
- 教师模型与其他任务检查点需要核对训练数据；一般预训练特征工具的可用性，不等于额外情感数据微调权重可不受限制使用。

### 8.3 解释与交付

- 模态路由权重、不确定性、梯度归因和删除贡献是不同量；需命名和定义清楚。
- 至少用删除／保留关键片段或可靠性置换等诊断检查解释的实际作用，不只展示好看的权重图。
- 附件2不含逐词时间戳；附件4现有字段也不能直接提供完整时间映射。必须核对特征位置到词、秒、帧的对应，不能将50个位置假设为50个等长时间格。
- 总附件≤50MB；需提前预算Q1特征和Q2/Q3模型参数。参数高效训练不代表基础模型无需计入复现方案。
- 公开参考代码保留原许可与引用；本题代码、数据转换和实验脚本另行组织，避免无记录地修改第三方快照。

## 9. 阅读与复现优先级建议

以下是基于当前任务边界的建议，不代表已经决定主模型。

| 目的 | 第一批 | 第二批／扩展 | 应验证的核心问题 |
| --- | --- | --- | --- |
| 简单预测基线 | 文本单模态、早期／晚期拼接、EF-LSTM | Graph-MFN、TFN | 跨模态信息是否确实带来收益 |
| Q2局部缺失 | P-RMF、简单模型＋连续缺失增强 | CMAD、QA-MoE、DNRNet、CICA | 收益来自缺失增强、蒸馏、恢复还是动态融合 |
| Q3可解释预测 | Graph-MFN、EMOE＋统一归因 | CICA、MRUF、SHAP／置换诊断 | 权重是否影响决策，证据能否回映并复核 |
| 强融合对照 | CaReFlow、TSD、CmIR | CACR、MMRest、EBMC | 分布对齐、解耦和鲁棒约束分别解决什么 |
| 双目标／有序建模 | MoB | MAESTRO | 极性与强度是否相互促进，是否损害中性类 |
| 高成本路线 | 暂不作为首批 | HyperEF、PaP、SeRIn、SentiLLM、原生全模态LLM读出 | 性能收益能否覆盖训练、推理与交付成本 |

## 10. 维护规则

- 新增文献时补充：完整题名、发表状态、原始链接、方法摘要、任务对应、核验深度与适配限制。
- 新的SOTA判断必须指定数据划分、输入特征／骨干、完整或缺失设定、指标定义和截止日期；没有统一复现实验时使用“作者报告”或“候选”。
- 本文件是文献与方法索引，不代替逐篇精读、引用审计或本题实验结果。
- 本次没有改动四套参考算法，也没有新增模型训练或专项测试预测。

## 10.1 对比方法簇与统一实验协议

本节把对比方法按“回答哪个科学问题”分簇，而不是按论文年份堆叠。它服务于问题2和问题3；问题1的特征提取与时序对齐另按题面附件1要求验收。题面没有指定固定基线名单，以下是团队建议的最小、可解释对比集合。

### 10.1.1 方法簇总览

| 簇 | 方法 | 核心问题 | 输入版本 | 本地状态 | 首轮优先级 |
| --- | --- | --- | --- | --- | --- |
| L0 下界 | Majority/Mean、单模态 Text/Audio/Vision | 多模态收益是否真实存在 | aligned；单模态不受对齐限制 | 需要在共享训练器中实现 | 必做 |
| L1 直接融合 | Concat-MLP、Early Fusion + GRU、EF-LSTM | 仅靠拼接与时序编码能达到什么水平 | aligned；Concat-MLP也可做 pooled-unaligned | EF-LSTM在 `MMSA`；Concat-MLP/GRU待写 | 必做 |
| L2 经典交互 | TFN、MFN、Graph-MFN/DFG | 高阶交互、记忆和动态融合是否带来收益 | TFN/MFN/Graph-MFN按各自可兼容版本；首轮固定aligned | TFN/MFN/Graph-MFN在 `MMSA`；DFG组件在SDK | 建议 |
| L3 非对齐跨模态注意 | MulT | 跨模态注意能否替代显式时序对齐 | unaligned | MMSA `MULT.py`，作者仓库公开 | 必做至少一个 |
| L4 缺失鲁棒 | P-RMF、CMAD、AUMDF、Masked-Train控制组 | 收益来自代理恢复、教师—学生蒸馏还是缺失增强 | P-RMF首选unaligned；CMAD/AUMDF首选aligned | P-RMF/CMAD本地快照；AUMDF已适配并验证 | 必做 |
| L5 可靠性／动态专家扩展 | EMOE、QA-MoE、CICA | 模态可靠性估计和动态路由是否进一步提升鲁棒性 | 先统一到aligned；逐一核对缺失协议 | EMOE/QA-MoE本地源码；CICA未找到可靠源码 | 扩展 |
| L6 可解释性 | Graph-MFN/DFG + permutation/occlusion 归因 | 模态权重是否真的影响预测，证据能否复核 | aligned | DFG/MFN已有组件；归因评估需项目实现 | 问题3必做 |

首轮不建议同时纳入 HyperDiff、CaReFlow、TSD、CACR、MMRest、EBMC、PaP 等高成本或设定差异较大的方法。它们可作为扩展实验，不能用作者在完整MOSEI上的公开数字直接填入本题结果表。

### 10.1.2 最小可交付对比集合

若计算资源有限，保留以下六个模型即可形成完整证据链：

1. `Text-only`：文本单模态下界。
2. `Concat-MLP`：三模态时间池化后拼接的最低复杂度基线。
3. `EF-LSTM`：每个对齐位置拼接三模态，再用单个LSTM编码。
4. `MulT`：非对齐跨模态注意力基线。
5. `CMAD` 或 `P-RMF`：选择一个公开缺失模态方法；若能复现两个则同时保留。
6. `AUMDF/Ours`：当前项目已完成的赛题数据适配基线或最终模型。

问题3至少在 `Concat-MLP`、`EF-LSTM`、`Graph-MFN/DFG` 和主模型上统一做归因；只展示主模型的注意力热图不构成方法对比。

### 10.1.3 方法定义与公平适配

#### L0：下界与单模态

- `Majority` 的极性输出为训练集最多类，强度输出为训练集强度均值；只用于sanity check，不作为论文主模型。
- 三个单模态模型使用相同的序列编码器和训练轮数，只改变输入模态。报告三分类Accuracy/Macro-F1与强度MAE/Pearson。
- 单模态模型不应使用缺失模态的零填充来伪造另一种模型；缺失输入应从接口中移除或显式标记。

#### L1：直接融合

- `Concat-MLP`：对每个模态沿有效时间位置做MeanPool，得到 `x_T, x_A, x_V`，拼接 `z=[x_T,x_A,x_V]`，共享一个MLP同时输出极性和强度。对unaligned数据，先分别池化，不进行伪造的逐位置拼接。
- `EF-LSTM`：只用于三模态已经具有相同时间位置的输入；令 `x_t=[x_t^T,x_t^A,x_t^V]`，用单个LSTM/GRU编码后接任务头。图中的“EF-LSTM”属于结构描述，不是新的独立论文名称；MMSA实现来源为 `rhoposit/MultimodalDNN`。
- GRU版本只作为LSTM的容量近邻控制组，隐藏维度、层数和任务头保持一致。

#### L2：经典交互

- `TFN` 显式建模单、双、三模态外积交互；需报告参数量，否则高阶外积可能把容量差误当作融合收益。
- `MFN` 使用模态内LSTM和多视图门控记忆；`Graph-MFN/DFG` 增加带效能权重的动态交互图，适合同时作为经典性能基线和问题3结构解释参照。
- 这些方法的MOSEI公开成绩来自不同数据规模、特征和指标口径，只能作为背景，不能直接并入本题结果表。

#### L3：MulT

- MulT的核心是六个方向的跨模态Transformer，将源模态序列通过跨模态注意力注入目标模态；它不要求三模态共享时间网格，因此首选 `unaligned_50.pkl`。
- 代码来源：[作者仓库 yaohungt/Multimodal-Transformer](https://github.com/yaohungt/Multimodal-Transformer)，MIT；本地统一框架为 `references/MMSA/src/MMSA/models/singleTask/MULT.py`。
- 必须记录文本50步、语音/视觉500步的有效长度和mask；不能把MulT改成aligned后仍声称测试了“非对齐优势”。

#### L4：局部缺失鲁棒

- `P-RMF`：不确定性高斯模态表示 + 代理模态重构；本地配置默认 `unaligned_50.pkl`，需要改为题目附件2路径、train/valid/test命名和统一评分协议。
- `CMAD`：完整输入教师、缺失输入学生；CAFD保持表示/相关结构，MAR根据模态组合难度调节训练。原代码主要按整模态组合置零，必须新增连续时间区间mask，才满足问题2局部缺失场景。
- `AUMDF`：REM/RPM/DWAM/MMT + CSD/SRD教师—学生流程，当前版本已在aligned赛题特征上完成独立适配；其随机位置、block区间和whole-modality结果要分开标记。
- `Masked-Train`：与主模型相同网络和参数，仅加入训练期连续区间置零；这是判断“鲁棒性是否只是数据增强收益”的必要控制组。

#### L5：可靠性／动态专家扩展

- `EMOE` 用模态专属增强专家和动态专家组合，可作为基础预测和Q3模态贡献扩展；但专家权重仍需用遮挡／置换验证，不直接等同证据。
- `QA-MoE` 用连续可靠性谱驱动专家路由，适合研究局部缺失强度变化；必须重做缺失构造，使训练／验证只使用附件2。
- `CICA` 只有论文结果可核验，当前没有确认的作者源码，不应作为“已复现基线”。

### 10.1.4 Q2统一缺失实验矩阵

所有进入Q2主表的方法必须使用同一组预生成mask和随机种子。每种条件报告Accuracy、Macro-F1、MAE、Pearson，并保存条件、mask种子和样本覆盖。

| 因素 | 最低设置 |
| --- | --- |
| 输入状态 | complete；单模态缺失T/A/V；双模态缺失TA/TV/AV |
| 连续区间位置 | beginning、middle、end |
| 缺失时长 | 有效长度的10%、30%、50%；按有效长度而非padding长度计算 |
| 缺失方式 | 将连续区间特征置零，同时更新observed mask；样本和标签不删除 |
| 训练增强 | 主模型和 `Masked-Train` 记录是否使用连续缺失增强；其他模型不得隐式使用额外数据 |
| 报告方式 | 每个方法×条件单独报告，禁止将random/block/whole混成一个分数 |

附件3无标签，只对冻结后的模型做一次全量推理；不能用附件3选择缺失率、阈值或最佳模型。

### 10.1.5 Q3统一解释实验

每个可解释模型至少输出：`polarity`、`intensity`、`main_modality`、三模态贡献分数、局部证据区间和原始素材定位。比较不只看图，还要做：

- 贡献归一性：三模态贡献是否非负且和为1，或明确其可为负的定义；
- 删除验证：删除Top-1模态/片段后性能下降是否大于随机片段和Bottom-1片段；
- 稳定性：固定样本重复运行或轻微扰动时，主模态和证据位置是否稳定；
- 可回映性：至少一个典型样本回映到文本片段、语音时段和视频关键帧；附件2没有逐词时间戳时，不得伪造精细词级定位。

### 10.1.6 表格与结论规则

- 主表按“方法簇×特征版本×缺失条件”组织；不得把作者报告的MOSEI Acc2/Acc7/F1与本题三分类结果混列。
- 所有方法通过共享 `e-competition-v1` 评分层，最终强度先固定反归一化和clip，再评分；评分层不替方法选择阈值。
- 公开代码、论文原生实现、本题适配实现、完整复现四种状态分开标记。
- 主结论应写成“在本题附件2、指定特征版本和连续区间协议下，本方法优于哪些对照”，不能写成无条件SOTA。

## 11. 代码公开与本地保存核验（2026-09-23）

本节补充并更新前文的代码可用性信息。**公开仓库存在、存在实际源码、具有完整开源许可证、已完成复现是四件不同的事。** 本次没有执行第三方实现。

- 核查R01–R18、S01–S14共32篇：16篇有对应源码或明确标注的组件／复现实现保存在本地，3篇仅有占位仓库，13篇暂未找到可靠实现。
- 新保存14个源码仓库，加上保留的4套原有实现，共18个本地目录。仓库数不同于论文数：Graph-MFN相关组件和复现框架分开保存，另有经典MFN仓库。
- 每个新仓库使用浅克隆与源码稀疏检出，保留提交号。只保存代码、配置、文本说明和许可文件；未检出数据集、模型权重、音视频。下载源码不代表它已适配本题或能直接运行。
- 完整来源、提交号、Python文件数和逐篇状态见 [code_manifest.json](code_manifest.json)。原有快照的commit记为null，避免冒认其上游版本。

### 11.1 已保存源码的目录与许可

| 本地目录 | 来源仓库 | Python文件数 | 许可核验 | 本次操作 |
| --- | --- | ---: | --- | --- |
| [CMAD/](CMAD/) | [YetZzzzzz/CMAD](https://github.com/YetZzzzzz/CMAD) | 18 | MIT | 保留原有快照 |
| [EMOE/](EMOE/) | [fuyyyyy/EMOE](https://github.com/fuyyyyy/EMOE) | 25 | 未发现独立许可证 | 保留原有快照 |
| [HyperDiff/](HyperDiff/) | [wdqdp/HyperDiff](https://github.com/wdqdp/HyperDiff) | 34 | Apache-2.0 | 保留原有快照 |
| [P-RMF/](P-RMF/) | [hawksilent/P-RMF](https://github.com/hawksilent/P-RMF) | 12 | MIT | 保留原有快照 |
| [CaReFlow/](CaReFlow/) | [TmacMai/CaReFlow](https://github.com/TmacMai/CaReFlow) | 8 | 未发现独立许可证 | 新增源码检出 |
| [QA-MoE/](QA-MoE/) | [U235-Aurora/QA-MoE](https://github.com/U235-Aurora/QA-MoE) | 37 | README声明MIT；未附完整许可文件 | 新增源码检出 |
| [CACR/](CACR/) | [wwwwwyyyy559/Conflict-Aware-Adaptive-Cross-Reconstruction-for-Multimodal-Sentiment-Analysis](https://github.com/wwwwwyyyy559/Conflict-Aware-Adaptive-Cross-Reconstruction-for-Multimodal-Sentiment-Analysis) | 25 | MIT | 新增源码检出 |
| [MMRest/](MMRest/) | [hurriedpi/MMRest](https://github.com/hurriedpi/MMRest) | 16 | MIT | 新增源码检出 |
| [EBMC/](EBMC/) | [kangverse/EBMC](https://github.com/kangverse/EBMC) | 15 | CC-BY-NC-4.0（非商业） | 新增源码检出 |
| [MRUF/](MRUF/) | [ICIG/MRUF](https://github.com/ICIG/MRUF) | 28 | GPL-3.0 | 新增源码检出 |
| [quality-aware-fusion-diagnostic/](quality-aware-fusion-diagnostic/) | [jadenmoon27/quality-aware-fusion-diagnostic](https://github.com/jadenmoon27/quality-aware-fusion-diagnostic) | 34 | MIT | 新增源码检出 |
| [VG-TPT/](VG-TPT/) | [ma-tubu/VG-TPT](https://github.com/ma-tubu/VG-TPT) | 12 | 未发现独立许可证 | 新增源码检出 |
| [SentiLLM/](SentiLLM/) | [especiallyW/SentiLLM](https://github.com/especiallyW/SentiLLM) | 13 | 未发现独立许可证 | 新增源码检出 |
| [DiscRead-MSA/](DiscRead-MSA/) | [GawaineXiukkie/DiscRead-MSA](https://github.com/GawaineXiukkie/DiscRead-MSA) | 25 | Apache-2.0 | 新增源码检出 |
| [CMU-MultimodalSDK/](CMU-MultimodalSDK/) | [CMU-MultiComp-Lab/CMU-MultimodalSDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK) | 50 | MIT；另含历史LICENSE.txt使用说明 | 新增源码检出 |
| [MMSA/](MMSA/) | [thuiar/MMSA](https://github.com/thuiar/MMSA) | 70 | MIT | 新增源码检出 |
| [MFN/](MFN/) | [pliang279/MFN](https://github.com/pliang279/MFN) | 3 | MIT | 新增源码检出 |
| [TLRA/](TLRA/) | [liuxy1005/TLRA](https://github.com/liuxy1005/TLRA) | 21 | 未发现独立许可证 | 新增源码检出 |

说明：未发现独立许可证的仓库只能标为“源码公开、许可未明确”。QA-MoE的README写明MIT，但未附完整许可文本。EBMC的CC-BY-NC-4.0带非商业限制。下载资料用于本地查阅不代表已确认可自由修改、分发或用于所有用途。

### 11.2 每篇论文的代码状态

| 编号／方法 | 状态 | 本地路径或判断依据 |
| --- | --- | --- |
| R01 CMAD | 已有源码 | [CMAD/](CMAD/)；本地已保存，保留原快照。 |
| R02 P-RMF | 已有源码 | [P-RMF/](P-RMF/)；保留原快照；论文原aoqzhu/P-RMF入口失效，记录可访问hawksilent仓库，不认定迁移关系。 |
| R03 EMOE | 已有源码 | [EMOE/](EMOE/)；本地已保存，未发现独立许可证。 |
| R04 HyperEF / HyperDiff | 已有源码 | [HyperDiff/](HyperDiff/)；本地已保存，目录名与论文框架名不同。 |
| R05 Graph-MFN / DFG | 已下载源码／组件 | [CMU-MultimodalSDK/](CMU-MultimodalSDK/)、[MMSA/](MMSA/)；SDK提供作者DFG融合组件；MMSA提供完整Graph-MFN的第三方实现，非原论文同一训练脚本。 |
| R06 CICA | 未找到可靠实现 | 正式页面、PDF源码链接扫描、GitHub检索后未找到可确认的作者代码。 |
| R07 CaReFlow | 已下载源码／组件 | [CaReFlow/](CaReFlow/)；作者仓库有实际源码，尚无独立许可证。 |
| R08 CmIR | 占位仓库，无源码 | [仓库](https://github.com/TmacMai/CmIR)；TmacMai/CmIR只有README，无模型源码，未下载为实现。 |
| R09 TSD | 未找到可靠实现 | 未找到可确认仓库；QFSD为不同论文，未替代下载。 |
| R10 QA-MoE | 已下载源码／组件 | [QA-MoE/](QA-MoE/)；U235-Aurora仓库引用完整论文题名及arXiv 2604.05704；排除同名临床仓库。 |
| R11 DNRNet | 未找到可靠实现 | 论文与GitHub检索均未定位可靠代码入口。 |
| R12 CACR | 已下载源码／组件 | [CACR/](CACR/)；README完整题名和CVPR 2026引用对应；选择非fork的声明官方实现。 |
| R13 MMRest | 已下载源码／组件 | [MMRest/](MMRest/)；README完整题名对应，模型位于MMRest/src/。 |
| R14 EBMC | 已下载源码／组件 | [EBMC/](EBMC/)；标题、模块与论文对应；CC-BY-NC-4.0为受限许可，不等同宽松开源许可。 |
| R15 PaP | 未找到可靠实现 | 未定位可确认的对应论文实现。 |
| R16 MoB | 未找到可靠实现 | 未定位可确认的对应论文实现；搜索到的其他Bottleneck专家仓库不是本论文。 |
| R17 SeRIn | 占位仓库，无源码 | [仓库](https://github.com/alexisfilippakopoulos/SeRIn-MSA)；作者链接可达，但只有README，无模型源码。 |
| R18 MAESTRO | 未找到可靠实现 | 未找到与该有序原型论文对应的实现。 |
| S01 语义完整度重构 | 未找到可靠实现 | PDF给出的LNLN、P-RMF、TF-Mamba链接为比较方法，不能当作本论文源码。 |
| S02 VG-TPT | 已下载源码／组件 | [VG-TPT/](VG-TPT/)；论文摘要直接链接仓库，已下载，未发现独立许可证。 |
| S03 迭代代理校正 | 未找到可靠实现 | 未定位作者实现。 |
| S04 MRCF | 未找到可靠实现 | 未定位作者实现。 |
| S05 SentiLLM | 已下载源码／组件 | [SentiLLM/](SentiLLM/)；论文摘要直接链接仓库；仓库标题有不同版本表述，身份按原文链接确认。 |
| S06 MRUF | 已下载源码／组件 | [MRUF/](MRUF/)；仓库名称与README方法对应，GPL-3.0。 |
| S07 SHAP加权跨模态专家 | 未找到可靠实现 | 未找到与该论文对应的源码；排除其他SHAP加权项目。 |
| S08 可靠性置换诊断 | 已下载源码／组件 | [quality-aware-fusion-diagnostic/](quality-aware-fusion-diagnostic/)；README引用完整论文题名，包含MOSEI相关诊断代码。 |
| S09 DiscRead-MSA | 已下载源码／组件 | [DiscRead-MSA/](DiscRead-MSA/)；README引用完整论文题名，Apache-2.0。 |
| S10 MCAF | 未找到可靠实现 | 未定位对应源码；本轮PDF入口返回404，不据此推断论文撤回。 |
| S11 TLRA | 已下载源码／组件 | [TLRA/](TLRA/)；PDF明确给出LiuXY3366/TLRA，GitHub跳转到liuxy1005/TLRA；不以另一个RACA仓库替代。 |
| S12 DBR | 未找到可靠实现 | 检索命中HDER仅在README引用DBR；不同方法，未下载为DBR。 |
| S13 ProMMA | 占位仓库，无源码 | [仓库](https://github.com/rongfei-chen/ProMMA)；README、LICENSE、.gitignore，无模型实现。 |
| S14 模态平衡受控评估 | 未找到可靠实现 | 正文及仓库检索未找到可确认的实现。 |

### 11.3 经典基线代码入口与来源边界

- **DFG作者组件**：[CMU-MultimodalSDK的dynamic_fusion_graph/model.py](CMU-MultimodalSDK/mmsdk/mmmodelsdk/fusion/dynamic_fusion_graph/model.py)。SDK同时提供tensor_fusion、multiple_attention等融合组件；它们不能自动等同完整TFN／MARN训练流程。
- **Graph-MFN第三方复现**：[MMSA的Graph_MFN.py](MMSA/src/MMSA/models/singleTask/Graph_MFN.py)。SDK README指向的原始Theano完整代码位于Google Drive，本次未下载；使用MMSA时按复现版本引用。
- **TFN、MFN、EF-LSTM统一框架实现**：见[MMSA的singleTask目录](MMSA/src/MMSA/models/singleTask/)。MMSA是第三方统一框架，不冒称每个方法的原作者代码。
- **MFN原作者仓库**：[MFN/test_mosi.py](MFN/test_mosi.py)，也包含EF-LSTM。原仓库附带的MOSI数据与预训练检查点未检出，旧Python/PyTorch依赖尚未适配。
- **TLRA真实入口**：论文PDF脚注提供的LiuXY3366/TLRA重定向至[liuxy1005/TLRA](https://github.com/liuxy1005/TLRA)，本次按该来源保存；不把同作者的RACA项目当作同一实现。

### 11.4 校验范围

已验证新增14个目录有Python源码、Git工作树干净、来源与提交可追溯，且检出的文件不含PKL/HDF5/NumPy数据、模型权重或音视频。原有四套目录保留不变。未运行模型、未验证训练效果或跨平台依赖兼容性；没有将任何新数据引入data/。源码子目录仅本地保存，主Git仓库只同步文档和来源清单。

## 12. 新增独立赛题适配实现：AUMDF

- **论文**：王楠、王淇、欧阳丹彤，《基于知识蒸馏与动态调整机制的多模态情感分析模型》，计算机学报，2025，48(8)：1923–1942；DOI：10.11897/SP.J.1016.2025.01923，对应题面参考文献[10]。
- **方法**：Attention-based Uncertain Missing Modality Distillation Framework。采用REM特征增强、RPM卷积与位置编码、DWAM动态门控、MMT跨模态注意力，以及CSD对比样本蒸馏和SRD相似性表示蒸馏。
- **本地实现**：[AUMDF/README.md](AUMDF/README.md)。这是依据用户提供PDF编写的独立实现，不是下载的作者源码，不计入前述32篇公开源码核验或18个第三方仓库统计。
- **适配边界**：用户确认使用赛题附件2的768/74/35维、50步对齐特征与既有划分；原论文使用300/74/1024维特征，且序列长度、回归/分类公式存在歧义。完整实现选择见[AUMDF/REPRODUCTION.md](AUMDF/REPRODUCTION.md)。
- **训练证据**：见[AUMDF/VALIDATION.md](AUMDF/VALIDATION.md)。代码测试、真实数据流程验证、训练结果和严格原论文成绩复现分开陈述。检查点与日志只保存于项目artifacts/aumdf/，不上传数据或权重。
