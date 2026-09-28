# 问题1：特征提取与对齐

| 模块 | 职责 |
|---|---|
| problem1.py | 参数解析、提取/映射/核验任务编排 |
| media.py | 视频时间戳、音轨解码、静音核验、外部命令 |
| extractors.py | BERT、eGeMAPSv02与OpenFace特征 |
| alignment.py | MFA、区间聚合、50位置视图 |
| pipeline.py | 100条全量处理、来源档案、异常清单 |
| validation.py | 人工词边界误差核验 |
| visualization.py | 时间覆盖图 |

在CPMCM根目录执行：`python problem1/problem1.py extract`。生成文件在本目录的results、figures、work。附件4映射逻辑位于problem3/mapping.py，共享问题1媒体与对齐模块。

`preparation.py`批量执行MFA；`verify_outputs.py`核验全量样本、特征和掩码。处理数据实际位于AAAdata/processed/problem1，工作区results/features为其链接。

2026-09-23更新：按用户要求，问题1的50位置视图采用前缀截断。唯一超长样本 `-a55Q6RWvTA$_$3` 为65词，词级视图保留前50词、丢弃后15词；三模态特征、掩码和时间轴同步截断。均匀对照保留前50个箱（不等同于50个词）。原始全文和全长特征留档，BERT特征沿用原全文上下文，并非重新对50词独立编码。截断记录见 `AAAdata/processed/problem1/view50_truncation.csv`。本次不改变问题2/3已有附件输入和冻结指标。
