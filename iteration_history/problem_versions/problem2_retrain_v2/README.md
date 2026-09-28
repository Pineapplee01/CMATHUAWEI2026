# problem2_retrain：CLS/SEP 音视 P=0 对照沙箱

与正式 `problem2/` **隔离**。数据副本 + 加载时 CLS/SEP 上 \(P_A=P_V=0\)。不覆盖正式权重与正式消融结果。

主说明见 [RETRAIN_README.md](RETRAIN_README.md)、[消融实验说明.md](消融实验说明.md)。

```bash
conda activate CPMCM
cd /user_home/gaojianan/CPMCM

# 正式架构多 seed 重训
python run_problem2_retrain.py --seeds 2026 2027 2028 --run-name retrain_f1_clssep_avP0

# 六组消融（默认配置 configs/problem2_retrain_ablation.json）
python -m problem2_retrain.ablation --smoke --run-name f1_component_ablation_check
python -m problem2_retrain.ablation
```
