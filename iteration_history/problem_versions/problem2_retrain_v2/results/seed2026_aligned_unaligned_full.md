# problem2_retrain_v2 seed2026 完整输入

| 版本 | 架构 | 完整test ACC | 正确/总数 | best epoch | ᾱ_T | ᾱ_A | ᾱ_V |
|---|---|---:|---:|---:|---:|---:|---:|
| aligned | F1+门控 | 68.78% | 500/727 | 3 | 0.696 | 0.156 | 0.148 |
| unaligned | soft_alignment+门控 | 69.33% | 504/727 | 2 | 0.716 | 0.134 | 0.150 |

样本级门控：α=softmax(wᵀ tanh(Wh+b))，Σα=1；只报告完整输入 test。
