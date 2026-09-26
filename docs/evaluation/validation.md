# 统一评分层验证记录

日期：2026-09-23；协议：`e-competition-v1`。本记录只证明评分迁移与导出链路，不证明完整竞赛交付合规。

## 自动化检查

- `python -m pytest tests -q`：43 passed（Python3.12.3、NumPy1.26.4）。
- `D:\Anaconda\envs\cmath-e2026\python.exe -m pytest tests -q`：43 passed（项目Python3.11环境，无PyTorch也能使用共享评分层）。
- `python -m pytest tests/baselines/aumdf -q`：AUMDF 含合成数据双阶段训练、检查点和评估集成测试。未重新训练真实数据模型。
- TDD：首批共享协议测试19项在API不存在时失败；AUMDF迁移测试4项在旧评分／导出行为下失败；浮点常数Pearson回归测试先失败后修复。
- CLI `--help`、`score --help` 和AUMDF `run.py --help`通过。当前本机两个Python环境尚未安装根包，源码烟测在进程内设置 `PYTHONPATH=src`；不应误称无需安装即可直接调用 `python -m e_emotion`。常规使用按根README执行可编辑安装。
- `codegraph sync/status/files/impact score_predictions`：索引包含48文件，共享评分影响链覆盖records、CLI和AUMDF。索引不提交Git。
- 独立只读代码审查发现并复核修复：NumPy float32 CSV精度损失、领域Polarity枚举转换、Pearson极小／近常数输入不稳定。均补充失败回归测试后修复。审查者报告60组随机结果与sklearn／SciPy一致，最终无未解决的阻断问题。

## 历史预测重算：不修改旧结果

来源：`artifacts/aumdf/reproduction-20260923/evaluation-test.json`。
SHA-256：`6e6a1926e5f6adfc1ad8e8e046537c07f8d3c3329616cc8eae770e5e74310846`。

对15种既有条件各727条预测：

1. 从 `data/raw/附件2-数据集特征文件/aligned_50.pkl` 读取真实test标签与ID，逐条确认历史target一致；不是用预测构造真值。
2. 仅用既有最终强度及历史报告已保存的阈值0.17恢复极性；不重校准、不生成未知raw值。
3. 共享 `score_records` 按ID检查覆盖，重算Accuracy、Macro-F1、Weighted-F1、MAE、Pearson。
4. 与历史同名指标及独立sklearn（accuracy_score、f1_score、mean_absolute_error）／NumPy corrcoef各自核对；最终代码的最大绝对差均为 `3.3306690738754696e-16`，小于 `1e-12`。
5. 核对前后历史文件字节一致。原始网络输出未保存在旧报告，未尝试补造。

这说明已有最终预测的得分未因共享迁移实质改变，不说明修改了模型性能。

## 冻结模型验证集推理与CSV往返

执行命令：

```powershell
python scripts/baselines/run_aumdf.py evaluate --checkpoint artifacts/aumdf/reproduction-20260923/student.pt --split valid --device cuda --output artifacts/aumdf/scoring-v1-accepted-20260923.json
```

产物位于本地artifacts，不进入Git；命令拒绝覆盖，复跑须换新输出名。

- 冻结学生检查点SHA-256：`b2f8642f36fd821112cee472b11c89d7ef7bba5d5777898ced7cdb6a88841a09`。
- 新报告SHA-256：`6e0105c5d857abc8b351ccf3ef1f4e9bfb0876faf704fd9dc60c0d3bdf99542e`。
- 9种条件 × 728条验证样本；每条件独立CSV，包含raw_intensity、intensity、polarity。
- 使用附件2原始valid标签重读全部9个CSV，`score_records`结果与JSON `metrics.competition`完全一致；逐条确认 `intensity=clip(raw_intensity,-3,3)`。
- 阈值直接读取旧检查点0.17，未进行新的阈值选择。

完整输入验证集结果（仅作为链路验收，不是新SOTA或新训练结果）：

| Accuracy | Macro-F1 | Weighted-F1 | MAE | Pearson |
| --- | --- | --- | --- | --- |
| 0.6085164835164835 | 0.5946763361136855 | 0.6157654146594833 | 0.6081408829412904 | 0.6249545887315924 |

## 附件3专项导出验收

2026-09-23 已对服务器冻结检查点执行无标签附件3对齐版导出：

- MMSA、EMOE、CACR、TLRA、AUMDF、CMAD 各30行；文件名 stem 作为 `sample_id`。
- 每个方法独立保存 CSV 与 `manifest.json`，不含标签和监督指标；阈值来自附件2 valid 冻结记录。
- MMSA 和 AUMDF 的文本由本地 BERT 从 `text_bert` 重算，manifest 明确标记一致性风险；不据此宣称严格等价。
- MMRest 标记 `not_ready`，原因是当前适配使用 unaligned 长度字段，而附件3 aligned 文件没有同一长度契约。
- 附件4预测、贡献、证据解释和连续缺失完整网格仍未完成。

## 未完成项

没有生成附件3/4最终预测或解释，没有实现全模型统一缺失实验矩阵、完整消融、问题1特征或问题3解释评分，没有验证最终50MB匿名提交包。未接入的参考模型仍需单独适配本协议。
