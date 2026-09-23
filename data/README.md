# 本地数据镜像

本目录是项目唯一允许的数据输入根目录。初始化时应将题目提供的 `E题数据/` 完整复制到 `data/raw/`，保持附件目录和文件名不变。

原始数据、PKL 特征、视频和派生模型不会提交到 GitHub；代码通过配置文件中的 `dataset.data_root` 读取它们。复制完成后运行：

```powershell
$env:PYTHONPATH = "src"
python -m e_emotion inspect-data --config configs/base.yaml
python -m e_emotion validate-data --config configs/base.yaml
```
