# Tech Stack

## Languages & Frameworks

| Technology | Version | Purpose |
| --- | --- | --- |
| Python | 3.11 | 数据探针、建模和导出 |
| NumPy | >=1.24 | 数组契约和轻量指标 |
| PyYAML | >=6.0 | YAML 配置读取 |
| OpenPyXL | >=3.1 | 题目标签表读取 |
| Matplotlib | >=3.8 | 后续论文图表 |
| PyTorch/Transformers | deferred | 后续模型轨道按需要加入 |

## Project Tooling

| Tool | Purpose |
| --- | --- |
| Conda | 固定 Python 3.11 和科学计算环境 |
| pytest | 自动化测试 |
| CodeGraph | 骨架完成后的依赖与影响检查 |
| Git/GitHub | 版本控制和远程同步 |

依赖声明位于 `environment.yml` 与 `pyproject.toml`；大模型权重、视频和 PKL 不进入仓库。

独立基线AUMDF的历史依赖记录位于 `references/memory/history/aumdf/requirements.txt`：已在现有Conda base的Python 3.12.3、PyTorch 2.5.1、CUDA 11.8上验证。项目cmath-e2026环境仍未安装PyTorch，不能将该次验证视为共享Python 3.11环境的训练验证。
