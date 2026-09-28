# 问题三 · 对齐版可解释

按 [模型建立_对齐版.md](模型建立_对齐版.md)：复用问题二 F1，**正向** Grad×Input（只保留支持当前预测的槽）+ 多上下文模态边际贡献 + 连续证据 + 删除复核 + Comp/Suff/Stab。

```bash
conda activate CPMCM
python -m problem3.problem3 --device cuda:3
python -m problem3.problem3 --smoke --device cuda:3
```

交付产物见 `results/问题三交付说明.md`（覆盖要求 4.(1)–(5)）：

| 文件 | 内容 |
|------|------|
| `问题三交付说明.md` | (1)–(5) 总述 |
| `问题三结果分析.md` | 测试集 + 附件4 读图分析（含图） |
| `典型样本解释卡.md` | 论文式解释卡表（3 张） |
| `附件4_全部解释卡.md` | 全量 20 张解释卡 |
| `附件4_可解释结果.csv` / `解释卡片.json` | 全量 20 条 |
| `测试集预测.csv` / `测试集指标.json` | 测试集性能 |
| `figures/*_importance.pdf` | 局部重要性曲线（01–20） |
| `figures/*_modality.pdf` | 三模态作用对比（01–20） |
| `figures/test_*.pdf` | 测试集混淆与错误率 |
