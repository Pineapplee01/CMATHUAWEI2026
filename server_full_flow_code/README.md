# 本地迁移整理说明

本目录是从服务器 `/user_home/gaojianan/CPMCM` 迁移到本地的全流程代码整理版，生成时间为 2026-09-28。

已迁移内容：

- `AAA提交版代码及结果/`：最终提交代码、运行说明、环境文件和主要结果小文件。
- `preprocess/`：统一预处理脚本目录，覆盖问题一原始视频特征抽取、问题二附件2 pkl 到 NPZ 恢复逻辑、问题三附件4 pkl 到解释输入构造。
- `shared/`、`tests/`、`configs/`：共享工具、测试和配置。
- `problem1_v2/`、`problem1_v3/`、`problem2/`、`problem2_retrain_v2/`、`problem3/`、`problem3_unaligned/`、`archives/`：迁移源码、配置、说明文档和小型 CSV 索引；未迁移大型中间特征、图像、缓存和训练结果目录。

未迁移内容：

- `AAAdata/`：大体积附件数据与中间数据未放入本目录，见 `AAAdata/README.md`。
- `AAAmodel/`：BERT、OpenFace、COVAREP、MFA、模型权重等未放入本目录，见 `AAAmodel/README.md`。
- `AAAcheckpoints/`：下载包/检查点集合未放入本目录，见 `AAAcheckpoints/README.md`。

本目录目标是保存“代码链路和依赖说明”，不是完整离线运行包。若要本地完整运行，需要按各 README 把数据、模型和权重放回相同相对路径。

## 校验状态

- 本地迁移目录共 574 个文件，约 20 MB。
- `preprocess/`、`AAA提交版代码及结果/`、`shared/`、`tests/` 等迁移源码已做 Python 语法编译检查。
- 唯一语法检查失败文件是 `archives/reference_repositories/CodeREWorld_MSA/create_data.py`，它来自外部参考仓库，使用 Python 2 风格 `print`，不是最终提交运行链路。
- 服务器原始 README 中仍保留 `run_problem2.py` 的历史入口说明；迁移时服务器根目录没有该文件。最终可用入口以 `AAA提交版代码及结果/问题一/运行说明.md`、`AAA提交版代码及结果/问题二/运行说明.md`、`AAA提交版代码及结果/问题三/运行说明.md` 和 `preprocess/README.md` 为准。

# 复杂场景下多模态情感识别

采用现有分析报告的竞赛背景。环境为CPMCM（Python 3.11、GPU PyTorch）；模型在AAAmodel，数据在AAAdata，各问代码、说明与结果在problemX。

- 问题一：[三种模态无效片段识别](problem1/三种模态无效片段识别方法.md)。
- 问题二：[问题二正式模型与运行入口](problem2/README.md)。正式权重为 `AAAmodel/checkpoints/problem2_aligned.pt`，seed2026 test ACC=72.63%，不增加CLS/SEP输出分支；唯一架构文件为 `problem2/model.py`。
- 问题三：[代码与配置](problem3/)。加载问题二使用 `problem2.training.restore`，应核对问题三配置中的checkpoint及SHA是否对应问题二正式模型。
- 数据预处理脚本统一归档见 [`preprocess/`](preprocess/README.md)：问题一保留从视频抽取三模态特征的完整流程，问题二保留从历史记录恢复的附件2 pkl 到 NPZ 生成逻辑，问题三保留附件4 pkl 到解释输入的构造逻辑。

## 问题二论文消融实验

```bash
conda activate CPMCM
python -m problem2.ablation
```

默认五个seed，对六组实验输出性能、连续缺失鲁棒性、稳定性、隐空间补偿和门控干预分析，生成六张PNG/PDF论文图。详细设置及小样本检查命令见[消融实验说明](problem2/消融实验说明.md)。研究实验不自动替换问题二正式模型，也不能把其新模块描述为已采用模型的既有功能。

旧README、旧架构与过期入口见[整理归档](archives/20260925_problem2_f1_cleanup/README.md)。保留历史结果与模型用于审计，当前使用以上入口。

## 完整运行总入口

```bash
conda activate CPMCM
python run_problem2.py
```

默认串联代码检查、正式权重test/valid及连续缺失评估、附件3预测、五seed六组研究消融与绘图。自选seed、仅评估及断点续跑用法见[总脚本使用说明](problem2/完整运行脚本使用说明.md)。
