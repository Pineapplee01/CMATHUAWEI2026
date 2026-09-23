# 系统架构

```mermaid
flowchart LR
    R[data/raw 题目附件] --> D[data adapters]
    D --> C[contracts]
    C --> A[alignment]
    C --> B[robustness]
    C --> X[explainability]
    A --> E[evaluation]
    B --> E
    X --> E
    A --> O[artifacts]
    B --> O
    X --> O
    F[configs/*.yaml] --> D
    F --> A
    F --> B
    F --> X
```

## 依赖方向

`config → data → contracts → task domains → evaluation/artifacts`。任务域之间不互相导入，公共接口集中在 `contracts`。

## 任务域职责

- `alignment`：原始视频特征提取、跨模态时间组织和特征清单。
- `robustness`：缺失掩码模拟、鲁棒预测和附件3导出。
- `explainability`：贡献计算、证据定位和附件4导出。

## CodeGraph 验收

骨架完成后运行 `codegraph init .`、`codegraph status` 和 `codegraph files`；后续大改动用 `codegraph impact` 检查公共契约影响范围。

## 预测评分边界

问题2/3及参考基线的输出进入 `evaluation.outputs`（固定反归一化、截断和极性适配），生成最终记录；`evaluation.records`负责CSV和ID覆盖，`evaluation.scoring`只计算最终值。评分模块不依赖模型或PyTorch。题面与团队细则分别记录于 [独立需求](../evaluation/requirements.md)，决策见 [ADR-003](../decisions/ADR-003-unified-scoring.md)。
