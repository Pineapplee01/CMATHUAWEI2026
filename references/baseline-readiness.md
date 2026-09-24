# L1–L6 方法簇准备度

核验日期：2026-09-24。本文把“论文/源码存在”“本地可追溯”“已接入赛题数据”“已有训练证据”分开记录。源码存在不等于已经完成赛题复现；作者在完整MOSEI上的结果也不等于本题结果。

## 结论

当前方法簇的文献和源码储备足够启动适配，但还没有达到“L1–L6全部可按统一竞赛协议复现实验”的状态。

- 已有本地源码或组件：EF-LSTM、TFN、MFN、Graph-MFN/DFG、MulT、P-RMF、CMAD、AUMDF、EMOE、QA-MoE。
- 本次新增本地源码：MulT作者仓库 `references/Multimodal-Transformer`，提交 `a670936824ee722c8494fd98d204977a1d663c7a`。
- 只有论文、没有可靠公开源码：CICA。
- 项目自有控制组，不需要寻找第三方论文源码：Concat-MLP、Early Fusion + GRU、Masked-Train、Permutation/Occlusion。
- 远程目标为 SSH 主机 `gaojianan2`，项目目录为 `/user_home/gaojianan/CPMCM/Baseline/reference`；本轮已补同步缺失的 `MFN` 和 MulT作者仓库，并完成文件哈希核对。项目自有控制组按方法名维护：`Concat-MLP/` 与 `Early-Fusion-GRU/`，不使用日期临时目录。

## 逐方法状态

| 簇 | 方法 | 论文/源码准备 | 本地位置 | 赛题适配或训练证据 | 尚未完成 |
|---|---|---|---|---|---|
| L1 | Concat-MLP | 项目自有基线，无需第三方源码 | `src/e_emotion/baselines`；远程 `reference/Concat-MLP/` | 模型、掩码适配器和契约测试已完成 | 训练、验证选优、统一评分产物 |
| L1 | EF-LSTM | MMSA有实现；MFN仓库也包含早期融合参考 | `references/MMSA/.../EF_LSTM.py` | 仅源码就绪 | 附件2适配、统一缺失实验、训练结果 |
| L1 | Early Fusion + GRU | 项目自有控制组 | `src/e_emotion/baselines`；远程 `reference/Early-Fusion-GRU/` | 模型、掩码适配器和契约测试已完成 | 训练、验证选优、统一评分产物 |
| L2 | TFN | MMSA有实现 | `references/MMSA/.../TFN.py` | 仅源码就绪 | 附件2适配和参数量受控训练 |
| L2 | MFN | MMSA及原作者仓库均有实现 | `references/MMSA/.../MFN.py`、`references/MFN/` | 仅源码就绪 | 附件2适配和统一评分 |
| L2 | Graph-MFN/DFG | SDK有DFG组件，MMSA有第三方Graph-MFN实现；不是原作者完整训练脚本 | `references/CMU-MultimodalSDK/`、`references/MMSA/` | 已有训练/检查点烟测记录；不等于完整主实验 | 统一连续缺失矩阵、Q3删除/置换归因 |
| L3 | MulT | 作者公开仓库和MMSA实现均可用 | `references/Multimodal-Transformer/`、`references/MMSA/.../MULT.py` | 本次新仓库已下载，尚未形成本题训练结果 | `unaligned`适配、有效长度/mask、统一评分 |
| L4 | P-RMF | 可访问仓库为 `hawksilent/P-RMF`；论文原入口失效 | `references/P-RMF/` | 已有本题适配/小规模运行记录 | 连续区间缺失协议和正式统一结果 |
| L4 | CMAD | 作者仓库本地已有 | `references/CMAD/` | 已有教师/学生适配烟测与导出链路 | 连续局部缺失完整网格、正式主表 |
| L4 | AUMDF | 项目独立赛题适配实现，不是作者源码 | `references/AUMDF/` | 已完成测试、训练和统一评分接入 | 多种子、完整消融和最终部署体积审计 |
| L4 | Masked-Train | 项目自有控制组 | 待实现 | planned | 与主模型同结构的连续缺失增强控制 |
| L5 | EMOE | 作者源码已本地保存 | `references/EMOE/` | 已有训练/验证和小规模运行记录 | 统一连续缺失矩阵、Q3归因验证 |
| L5 | QA-MoE | 作者关联仓库已本地保存 | `references/QA-MoE/` | 已有训练/验证和小规模运行记录 | 附件2固定特征适配、连续缺失正式结果 |
| L5 | CICA | 论文可核验，未找到可靠作者源码 | 无 | paper-only | 不纳入已复现基线；如需使用只能自行实现并单独命名 |
| L6 | Graph-MFN/DFG归因 | 模型/DFG组件有源码 | `references/MMSA/`、`references/CMU-MultimodalSDK/` | 归因框架部分存在 | 删除、遮挡、置换和稳定性验证 |
| L6 | EMOE归因 | 模型有源码，贡献权重可读 | `references/EMOE/` | 归因接口部分存在 | 证明权重确实影响预测 |
| L6 | Permutation/Occlusion | 项目自有诊断工具 | 待实现 | planned | 公共归因评估器和证据导出 |

## 已下载的公开源码

本次新增：

- [MulT作者仓库](https://github.com/yaohungt/Multimodal-Transformer)
- 本地目录：[Multimodal-Transformer](Multimodal-Transformer/)
- 固定提交：`a670936824ee722c8494fd98d204977a1d663c7a`
- 许可证：MIT
- 检出内容：10个Python文件、README、LICENSE及模型模块；没有数据、权重或音视频。

已有且与L1–L6直接相关的公开源码：

- [MMSA](MMSA/)：EF-LSTM、TFN、MFN、Graph-MFN、MulT的统一框架实现。
- [MFN](MFN/)：MFN原作者仓库，也包含EF-LSTM参考代码。
- [CMU-MultimodalSDK](CMU-MultimodalSDK/)：DFG组件和其他融合组件。
- [P-RMF](P-RMF/)、[CMAD](CMAD/)、[EMOE](EMOE/)、[QA-MoE](QA-MoE/)：对应方法源码。
- [AUMDF](AUMDF/)：项目独立适配实现，不标记为作者源码。

完整提交号、文件统计和许可证记录见 [code_manifest.json](code_manifest.json)。第三方源码目录继续被主仓库 `.gitignore` 排除；主仓库同步的是来源清单、配置和准备度报告。

## 没有源码或不应冒充已有源码的方法

### CICA

截至核验日期，在论文正式页面、论文源码链接和 GitHub 检索中没有找到可确认的作者实现。CICA保持 `paper_only`，不进入“已复现方法”表，不使用作者论文成绩填充本题结果。

### 项目自有控制组

Concat-MLP、Early Fusion + GRU、Masked-Train 和 Permutation/Occlusion不是需要下载的独立论文实现，而是为本题公平比较建立的控制组。它们需要在项目公共数据契约下实现，不能把某个第三方拼接网络直接称为这些控制组。

## 服务器同步状态

| 目标 | 状态 | 说明 |
|---|---|---|
| GitHub `main` | 可同步项目元数据 | 公开源码目录按既有策略不进入主仓库；manifest、配置和本报告可推送 |
| SSH `gaojianan2` | 已同步 | `/user_home/gaojianan/CPMCM/Baseline/reference`；新增 `MFN`、`Multimodal-Transformer`，并将控制组分别同步到 `Concat-MLP/`、`Early-Fusion-GRU/`；来源清单和准备度报告已写入 |
| SSH `GPU` | 未使用 | 不是本轮指定服务器，不影响 `gaojianan2` 同步结果 |

后续在 `gaojianan2` 上继续适配时，应该按以下顺序执行：

1. 在远程 `Baseline/reference` 下确认现有目录和工作树状态；
2. 以本地固定提交检出 `Multimodal-Transformer`，并核对 `MMSA`、`MFN`、`CMAD`、`P-RMF`、`EMOE`、`QA-MoE`提交号；
3. 只同步源码和来源清单，不同步数据、权重、日志或缓存；
4. 在远程运行公共加载器烟测，再开始赛题训练。
