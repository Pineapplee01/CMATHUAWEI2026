# 统一评分协议 e-competition-v1

适用问题2、问题3的情感极性／强度预测；不是问题1的特征质量评分器，也不是问题3的解释质量评分器。依据与验收状态见 [requirements.md](requirements.md)。本协议落实用户已接受的输出规则。官方后续发布的评分脚本或澄清若不同，须升级协议并重新评分，不能静默改变旧结果。

## 官方要求与团队选择

题面规定三类极性、原始标签量表、Accuracy/F1/MAE/Pearson、数据使用边界及专项CSV。题面**没有规定** F1平均方式、预测截断、阈值搜索网格、CSV列名或综合总分。下述细化规则为团队统一约定，不是官方隐藏评分器的复刻。

## 预测和评分的唯一边界

`网络原始输出 → 固定反归一化 → clip(-3,3) → 最终预测记录 → CSV／纯评分函数`

1. 原始输出 `raw_intensity` 保留供诊断。若标签训练时变换为 `(y-offset)/scale`，输出适配器用训练阶段确定的正数 `scale` 和 `offset` 恢复原始量纲。没有标签变换时使用1和0。不得基于当前批次做Min-Max放缩。
2. 最终强度 `intensity = clip(scale * raw_intensity + offset, -3, 3)`。真实标签不截断、不归一化、不取整。非有限值及反归一化溢出直接报错。
3. 极性由模型的分类头，或已冻结的强度转极性规则给出；必须显式写入记录。不能在评分时重新生成类别、选阈值或覆盖分类头结果。
4. 对仅有回归头的AUMDF：`p < -τ → Negative`；`-τ ≤ p ≤ τ → Neutral`；`p > τ → Positive`。默认τ=0；可用附件2验证集从0到0.5、步长0.01选择Macro-F1最大者，同分取较小值，保存到检查点。不是所有模型必须使用回归头。
5. 真实标签严格按 `y<0 / y=0 / y>0` 映射为0/1/2。预测的中性阈值不能用于重标真实标签。若给定真实极性列与强度矛盾，拒绝评分。
6. 评分使用最终记录中的强度和极性，**不修改预测**。空集合、重复ID、缺失／多余ID、非有限值、非法类别、最终强度越界均报错，不删除样本或自动修复。记录评分按ID连接，不依赖行序。

例：原始 `[-4,0,4]` 经适配生成最终 `[-3,0,3]`，真值为 `[-3,0,3]`，最终MAE=0。底层诊断函数 `mae(y, raw)` 仍为2/3；统一竞赛评分入口收到未适配的越界值会报错，而不会返回冒充合法提交的成绩。

## 指标定义

| 字段 | 统一口径 |
| --- | --- |
| accuracy | 三分类正确样本数/N，所有样本含中性，值域[0,1] |
| macro_f1 | 固定Negative/Neutral/Positive三个类别，逐类 `2TP/(2TP+FP+FN)` 等权平均；分母为0记0，包括缺席类 |
| weighted_f1 | 相同逐类F1，按真实类别支持数加权；补充报告，不替代Macro-F1 |
| mae | `mean(abs(y-intensity))`，原始[-3,3]量纲，不四舍五入 |
| pearson | 同一组最终强度与真值的Pearson相关系数；N<2或任一序列常数返回JSON null，另给原因；不以0代替未定义 |
| class_support / confusion_matrix | 固定类序Negative、Neutral、Positive；混淆矩阵行为真值、列为预测 |

各条件单独评分，不将缺失率、模态组合或不同特征版本的记录混成一个指标。多种子先逐种子报告；本层不设官方未提供的综合总分、排名或跨条件平均分。

论文常见的排除零值二分类、七级分类只能作为 `paper_diagnostics`，不能代替赛题三分类。AUMDF旧字段保留为兼容别名，标准报告位于 `metrics.competition`。

## 数据与实验纪律

- 模型参数只在附件2 train学习；结构、超参数、决策阈值在valid选择；附件3/4只最终推理，不计算虚构的监督指标。
- 附件2 test仅在方案冻结后作为补充留出评估；不能替代题面要求的验证集基础性能，也不能调参。这是团队对保留test划分的使用约定。
- `select_neutral_threshold(..., split="valid")` 显式约束调用。该参数不是数据来源的密码学证明；调用者仍需保存真实划分、样本ID及配置。数组评分函数无法辨认传入标签的真实来源。
- 比较模型时固定特征版本、划分、样本集合、缺失掩码／种子和条件。Q2须分析连续局部缺失的模态类型、位置、时长／缺失率；独立随机位置和整模态缺失仅为补充。
- 当前统一的是预测评分，尚未实现全模型共用的缺失实验生成矩阵；不得将AUMDF当前条件集说成官方完整验收。

## 公共接口与CSV

实现位于 `src/e_emotion/evaluation/`，不依赖PyTorch：

- `adapt_output(raw_intensity, scale=..., offset=..., threshold=..., predicted_polarity=...)`
- `prediction_rows(ids, adapted_output)`：保留原始与最终值；适配元数据另存入运行报告。
- `score_predictions(y, final_intensity, final_polarity, truth_polarity=...)`：严格一维数组。
- `score_records(truth_rows, prediction_rows)`：按ID校验全量覆盖并评分；跨模型推荐入口。
- `write_predictions_csv / read_records_csv`：UTF-8、完整浮点精度、不覆盖现有文件。

团队CSV基础列：`id,intensity,polarity`；`polarity`使用Negative/Neutral/Positive；诊断列可带 `raw_intensity`。标签CSV为 `id,intensity`，可附真实 `polarity`。ID保留数据提供的 `video_id$_$clip_id`。API也接受数值类别0/1/2；CSV极性使用英文名，避免数字字符串歧义。

问题3的解释字段是另一交付契约，本基础预测CSV不等于完整附件4提交文件。评分层不负责重建关键证据。

通用评分（先按README安装共享包；标签必须在data内，预测文件可位于artifacts）：

```powershell
python -m e_emotion score --truth data/derived/valid-labels.csv --predictions artifacts/BASELINE/valid.csv --output artifacts/BASELINE/score-valid.json
```

输出包含协议版本，不推测输入文件属于哪个模型／划分；调用者必须随运行记录保存这些来源。JSON与CSV既有路径拒绝覆盖。该命令不会训练、校准或访问测试真值调参。

## 迁移与验证

AUMDF `run.py evaluate` 新生成JSON及同名目录下每个条件的CSV，CSV与 `metrics.competition` 基于同一组记录。旧报告和权重保持原样，不补造旧的原始网络输出。历史记录只有最终强度，迁移核对时只能用其已记录的冻结阈值补出极性，并明确这是从旧记录恢复的输出。

测试入口：`tests/test_scoring_protocol.py`、`references/AUMDF/tests/test_data_metrics.py`、`references/AUMDF/tests/test_training.py`。验证证据见 [validation.md](validation.md)。
