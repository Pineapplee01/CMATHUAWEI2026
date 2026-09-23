# AUMDF：赛题数据适配复现

本目录是《基于知识蒸馏与动态调整机制的多模态情感分析模型》（王楠、王淇、欧阳丹彤，计算机学报，2025）的**独立实现**，不是作者发布的源码。用户已确认使用E题数据适配，不能将本目录结果称为原论文完整数据集成绩复现。

论文DOI：10.11897/SP.J.1016.2025.01923。原文、公式定位、实现选择及未报告参数见 [REPRODUCTION.md](REPRODUCTION.md)；已完成的运行证据见 [VALIDATION.md](VALIDATION.md)。

## 已实现

- REM：双BiLSTM全局／过滤分支、时序标准化、阈值门控与统计补偿。
- RPM：一维卷积与正弦位置编码。
- DWAM：成对ReLU门控、拼接及目标模态投影相乘。
- MMT：三路跨模态Q/K/V、局部时间窗口、可靠的填充屏蔽、残差和FFN。
- CSD／SRD：按真实极性构造跨样本正负对；教师—学生双向跨模态对比；投影后余弦表示蒸馏。
- 两阶段训练：完整输入教师，冻结教师后训练缺失输入学生；验证集选择检查点。
- 随机位置、连续区间、整模态三种**分别标记**的评估协议。
- 默认连续强度回归；可选七分类CE。赛题三分类采用验证集校准的中性区间。

不包含附件1特征提取、不使用外部情感训练数据，也不假设附件3/4与附件2字段完全一致。本版本训练和评估接口针对附件2；专项测试集文本接口尚需单独适配。

## 环境

最低Python 3.11。实际验证环境为本机Conda base：Python 3.12.3、PyTorch 2.5.1、CUDA 11.8、NumPy 1.26.4、RTX 4060 Laptop 8GB。没有修改base环境，也没有向缺少PyTorch的cmath-e2026环境安装大型依赖。

在已有兼容环境中可直接使用；若另建环境，按显卡驱动安装相应PyTorch，再安装requirements.txt中的依赖。Windows原生PyTorch 1.8.2不是本次使用的版本。

以下命令均在项目根目录执行：

```powershell
python -m pytest -c references/AUMDF/pyproject.toml references/AUMDF/tests -q
python references/AUMDF/run.py --help
```

无需安装本包；run.py以本目录的aumdf包为入口。

## 数据与输出

默认读取 `data/raw/附件2-数据集特征文件/aligned_50.pkl`，保留原train/valid/test=3395/728/727。模型输入为768/74/35维，均为50步。

只允许从项目data/加载建模数据，只允许向artifacts/写训练输出。配置中的相对数据路径相对项目根目录解析，不受当前工作目录影响。输入PKL必须来自可信的题目材料。

## 训练

先验证流程：

```powershell
python references/AUMDF/run.py train --smoke --device cuda
```

smoke只使用前64条训练、32条验证，每阶段1轮，输出明确标为smoke_only；不能作为精度或复现结论。

完整赛题子集训练：

```powershell
python references/AUMDF/run.py train --config references/AUMDF/configs/adapted.yaml --device cuda
```

每阶段最多20轮。输出默认创建唯一时间戳目录，也可用 `--output artifacts/aumdf/my-run` 指定**尚不存在**的目录；已有目录会拒绝覆盖。

连续区间缺失是额外适配协议，应独立训练并使用不同目录：

```powershell
python references/AUMDF/run.py train --missing-mode block --device cuda --output artifacts/aumdf/block-run
```

默认配置使用论文RPRM式独立随机位置缺失。本次默认训练结果不能冒称以连续区间缺失训练的结果。

## 固定检查点评估

将RUN替换为真实运行目录名：

```powershell
python references/AUMDF/run.py evaluate --checkpoint artifacts/aumdf/RUN/student.pt --split valid --device cuda --whole-modalities --output artifacts/aumdf/RUN/evaluation-valid.json
```

最后只对固定方案使用测试集：

```powershell
python references/AUMDF/run.py evaluate --checkpoint artifacts/aumdf/RUN/student.pt --split test --device cuda --whole-modalities --output artifacts/aumdf/RUN/evaluation-test.json
```

默认评估完整输入及0.1/0.3/0.5/0.7随机位置、连续区间缺失；`--whole-modalities`另加六种非完整模态组合。中性阈值从检查点载入，评估不重新调阈值。报告中的预测保留样本ID，按条件分组，避免混用协议。

## 产物与判断边界

每次训练输出：

- `config.json`、`environment.json`、`data_audit.json`：配置、实际环境、数据划分与SHA-256。
- `teacher.pt`、`student.pt`：验证集选出的模型、缩放参数、中性阈值和损失投影状态。
- `history.jsonl`：每轮训练损失、验证指标、选择分数与耗时。
- `summary.json`、`status.json`：运行结果和完成／失败状态。

CPU单元测试、CUDA小规模训练和完整训练是不同验证层级。一次随机种子的结果不能证明论文提升，也不能替代消融实验。训练checkpoint包含蒸馏投影状态，最终竞赛打包需另做纯学生推理导出并结合Q1特征检查50MB限制。

## 代码结构

```text
aumdf/
  model.py        REM / RPM / DWAM / MMT / readout
  losses.py       CSD / SRD / QKV regularization
  missingness.py  random / block / whole protocols
  data.py         restricted input / masks / train-only scaler
  metrics.py      MAE / Pearson / 3-class / paper-style metrics
  engine.py       teacher-student training / frozen evaluation
configs/adapted.yaml
tests/
run.py
```
