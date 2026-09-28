# 问题三 · 未对齐版可解释

按 [模型建立_未对齐版.md](模型建立_未对齐版.md)：复用 SoftAlignment（50/500/500，**不**池化到 50），**正向** Grad×Input + 多上下文模态边际贡献 + 连续证据 + soft association + Comp/Suff/Stab。

```bash
conda activate CPMCM
python -m problem3_unaligned.problem3 --device cuda:3
```

交付见 `results/问题三交付说明.md`（同对齐版结构覆盖 4.(1)–(5)）；读图分析见 `results/问题三结果分析.md`。图为 PDF：`results/figures/*_importance.pdf`、`*_modality.pdf`、`test_*.pdf`（预览 PNG 在 `figures/_preview/`）。
