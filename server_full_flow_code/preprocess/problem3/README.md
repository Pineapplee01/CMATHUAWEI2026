# 问题三预处理

本目录保存附件4 `.pkl` 到解释模型输入的构造逻辑，来自最终提交的 `问题三/代码`。

## 脚本

- `preprocess_aligned.py`：附件4对齐版 `.pkl` -> 对齐模型 batch。
- `preprocess_unaligned.py`：附件4未对齐版 `.pkl` -> 文本 50、音频/视觉 500 的未对齐模型 batch。
- `q3_utils.py`：路径解析和可信本地 pickle 读取工具。

## 口径

附件4预处理直接读取题目提供的 `.pkl` 中的 `text_bert`、`audio`、`vision`、`raw_text` 等字段，构造 `P/O` 掩码并用问题二训练集的 `scaler_params.npz` 标准化音频/视觉。文本隐状态由问题二/三模型在线 BERT 编码。

这里不会从附件4视频重新运行 OpenFace、COVAREP 或 FFmpeg 抽取特征；视频只用于媒体回看和结果解释展示。
