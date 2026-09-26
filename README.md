# 2026年中国研究生数学建模竞赛 E题

本项目面向“复杂场景下多模态情感预测的数学建模与算法设计”，采用共享核心与三任务域结构：

1. 问题1：多模态特征提取与时序对齐；
2. 问题2：局部模态缺失下的鲁棒情感预测；
3. 问题3：可解释多模态情感预测。

## 初始化环境

```powershell
conda env create -f environment.yml
conda activate cmath-e2026
python -m pip install -e ".[data,visualization,test]"
```

将题目附件完整复制到 `data/raw/` 后，执行数据探针：

```powershell
e-emotion inspect-data --config configs/base.yaml
e-emotion validate-data --config configs/base.yaml
pytest
```

问题2和问题3的 `dataset.variant` 必须明确填写 `aligned` 或 `unaligned` 后再运行。共享核心提供契约、校验器、评分和任务接口；AUMDF 的历史适配记录位于 [Baseline Workspace](references/memory/history/aumdf/README.md)。

独立的 [竞赛需求清单](docs/evaluation/requirements.md) 区分官方规则与团队约定。所有基线应接入 [统一评分层](docs/evaluation/protocol.md)：输出适配时固定截断，评分时不改预测。使用 `python -m e_emotion score --help` 查看跨模型CSV评分入口。

术语约束见 [UBIQUITOUS_LANGUAGE.md](UBIQUITOUS_LANGUAGE.md)，项目上下文见 [conductor/index.md](conductor/index.md)。
