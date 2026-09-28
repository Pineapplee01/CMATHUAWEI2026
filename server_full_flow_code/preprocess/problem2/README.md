# 问题二预处理

本目录保存从服务器历史记录中恢复的附件2完整生成逻辑。当前服务器项目树中，附件2生成脚本已经不再作为普通文件存在；最终提交只保留了生成后的 NPZ 数据。

## 当前最终数据

最终提交直接读取：

```text
AAAdata/Appendix_2/对齐版本/
AAAdata/Appendix_2/未对齐版本/
```

这两个目录中是 `train.npz`、`valid.npz`、`test.npz`、`scaler_params.npz` 和 `preprocess_report.json`，不是原始 `.pkl`。

## 恢复脚本

- `preprocess_aligned.py`：附件2对齐版原始 `.pkl` -> 对齐版 NPZ。
- `preprocess_unaligned.py`：附件2未对齐版原始 `.pkl` -> 未对齐版 NPZ。
- `RECOVERY_PROVENANCE.md`：恢复来源、历史 transcript 行号和恢复文件 SHA256。

两个脚本都会：

- 读取 `train/valid/test.pkl`；
- 构造有效轴 `P` 与原观测 `O`；
- 使用本地 `AAAmodel/bert-base-uncased` 对文本槽位编码；
- 仅用训练集 `O=1` 位置拟合三模态 z-score；
- 在不可观测位置置零；
- 输出 `train/valid/test.npz`、`scaler_params.npz`、`preprocess_report.json`。

## 推荐重跑方式

为避免覆盖最终提交数据，默认输入与输出使用单独目录：

```bash
python preprocess/problem2/preprocess_aligned.py \
  --raw AAAdata/Appendix_2_raw/对齐版本 \
  --output AAAdata/Appendix_2_regenerated/对齐版本

python preprocess/problem2/preprocess_unaligned.py \
  --raw AAAdata/Appendix_2_raw/未对齐版本 \
  --output AAAdata/Appendix_2_regenerated/未对齐版本
```

如确需覆盖最终提交使用的数据目录，应显式把 `--output` 指向 `AAAdata/Appendix_2/{对齐版本,未对齐版本}`，并先备份现有 NPZ。
