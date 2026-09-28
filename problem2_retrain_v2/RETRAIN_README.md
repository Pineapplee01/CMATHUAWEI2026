# problem2_retrain 沙箱

与正式 `problem2/` 隔离的代码副本；数据为 `对齐版本_retrain_20260925/processed_po`（内容与正式 `processed_po` 相同，**不修改**原目录）。

加载时：**CLS/SEP 槽上 \(P_A=P_V=0\)**（与文本一致）；磁盘 NPZ 不改写。

## 入口

| 步骤 | 命令 |
|---|---|
| 多 seed 正式架构重训 | `python run_problem2_retrain.py --seeds 2026 2027 2028 2029 2030 --run-name ...` |
| 六组消融（默认五 seed） | `python -m problem2_retrain.ablation` |
| 消融冒烟 | `python -m problem2_retrain.ablation --smoke --run-name f1_component_ablation_check` |

配置：`configs/problem2_retrain_bert_finetune.json`、`configs/problem2_retrain_ablation.json`。

## 产物（不覆盖正式；对齐 / 未对齐分目录）

| 类型 | 路径 |
|---|---|
| 对齐训练 runs | `problem2_retrain/results/aligned/runs/<run_name>/` |
| 对齐消融 | `problem2_retrain/results/aligned/ablations/<run_name>/` |
| 对齐完整评测 | `problem2_retrain/results/aligned/full_runs/<run_name>/` |
| 未对齐训练 runs | `problem2_retrain/results/unaligned/runs/<run_name>/` |
| 对齐图 | `problem2_retrain/figures/aligned/` |
| 未对齐图 | `problem2_retrain/figures/unaligned/<run_name>/` |
| 训练权重 | `AAAmodel/checkpoints/runs/<run_name>/retrain_*.pt` |
| 消融权重 | `AAAmodel/checkpoints/ablations_retrain/<run_name>/` |

消融细节见 [消融实验说明.md](消融实验说明.md)。

## 未对齐 50/500/500（局部窗 + 软注意力）

原生音视 **不池化**：文本 50、音视 500；每词一段连续单调局部窗，**窗内仍用内容软注意力**。配置与说明见 [未对齐数据软注意力方案.md](未对齐数据软注意力方案.md)。

```bash
python run_problem2_retrain.py --config configs/problem2_retrain_unaligned.json \
  --seed 2026 --run-name retrain_unaligned_soft_attention_seed2026
```

## 正式架构 F1 重训（CLS/SEP 音视 P=0，五 seed）

| seed | valid Acc | test Acc | local30 | 中性F1 |
|---:|---:|---:|---:|---:|
| 2026 | 66.07% | 71.66% | 66.85% | 42.69% |
| 2027 | 64.97% | 69.33% | 66.57% | 37.75% |
| 2028 | 65.11% | 68.36% | 66.44% | 29.31% |
| 2029 | 65.25% | 69.46% | 65.47% | 40.15% |
| 2030 | 64.56% | 69.60% | 64.65% | 46.00% |

均值±标准差（n=5, ddof=1）：test Acc **69.68% ± 1.21%**；local30 **66.00% ± 0.91%**。

## 消融（F1 组件拆除，B0=完整模型）

| 组 | 含义 |
|---|---|
| B0 | 完整 F1 |
| noC | 拆除补全 |
| noQ | 拆除可靠性 q |
| noB | 拆除边界损失 |
| noBeta | 拆除类别平衡 |

```bash
python -m problem2_retrain.ablation --smoke --run-name f1_component_ablation_check
python -m problem2_retrain.ablation
```

详见 [消融实验说明.md](消融实验说明.md)。旧 G/C/R 矩阵结果已删除。
