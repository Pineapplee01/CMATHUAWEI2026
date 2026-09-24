# L1 直接融合控制组

当前已实现两个项目自有控制组模型：`Concat-MLP` 和 `Early Fusion + GRU`。它们不是第三方论文源码，而是用于回答“简单融合本身能达到什么水平”的可控参照。

模型依赖是可选的 PyTorch extra：`python -m pip install -e ".[models,test]"`。基础数据探针和评分层不要求安装 PyTorch。

## 公共输入

模型接收 `ProcessedSplit` 转换后的三路 torch 张量和三路 `observed_mask`：

```text
text   : (B, T, 768)
audio  : (B, T, 74)
vision : (B, T, 35)
mask   : (B, T) bool
```

转换入口为 `e_emotion.baselines.split_to_torch_inputs`。它只读取项目公共 NPZ/`ProcessedSplit` 契约，不从特征数值推断缺失，不读取历史 PKL。

## 模型定义

### Concat-MLP

对每种模态按 `observed_mask` 做均值池化：

\[
z_m = \frac{\sum_t m_{m,t}x_{m,t}}{\max(1,\sum_t m_{m,t})},\qquad
z=[z_T;z_A;z_V].
\]

拼接向量进入一个共享 MLP，并输出：

```text
polarity_logits : (B, 3)
raw_intensity   : (B,)
```

该模型允许三种模态的时间长度不同，因此可作为 pooled 对照；首轮 aligned 实验仍固定使用 `aligned_50`。

### Early Fusion + GRU

要求三种模态拥有同一时间长度，在每个位置直接拼接：

\[
x_t=[x^T_t;x^A_t;x^V_t].
\]

拼接序列输入单个 GRU，再用跨模态有效位置的掩码均值池化输出同样的两个任务头。单个模态在某位置缺失时，该模态位置被置零，但不会把整个时间位置从序列中删除。

## 代码入口

- 模型：[direct_fusion.py](../../src/e_emotion/baselines/direct_fusion.py)
- 公共输入适配：[adapters.py](../../src/e_emotion/baselines/adapters.py)
- 契约测试：[test_direct_fusion.py](../../tests/test_direct_fusion.py)

```python
from e_emotion.baselines import ConcatMLP, EarlyFusionGRU, split_to_torch_inputs

features, masks = split_to_torch_inputs(valid_split)
model = ConcatMLP(hidden_dim=128)
output = model(features, masks)
```

`output.intensity` 是网络原始输出，不能直接作为竞赛提交值。训练/验证后必须调用统一输出适配器完成反归一化、`clip(-3, 3)` 和极性处理，再进入共享评分层。

## 当前边界

本次完成的是模型和公共输入适配，不是完整训练结果。尚未完成：

- train/valid 训练器、检查点和验证选优；
- 统一 `e-competition-v1` CSV/JSON 产物；
- Q2 连续局部缺失完整矩阵；
- 附件3专项推理。

因此注册表状态使用 `implemented_model_not_trained`，不能写成“已复现”或“已完成基线实验”。
