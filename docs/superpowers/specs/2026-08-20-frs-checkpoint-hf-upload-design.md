# FRS Checkpoint Hugging Face 上传脚本设计

## 目标

在 `scripts/upload_frs_ckpt.sh` 提供一个可复用的命令，将用户指定的 FRS checkpoint 目录上传到 Hugging Face public model 仓库，并把训练图像上传到同一仓库的 `figures/` 目录。

## 命令行接口

```bash
bash scripts/upload_frs_ckpt.sh OWNER/REPO /path/to/checkpoint \
  [--figures-dir /path/to/training-output]
```

- `OWNER/REPO`：目标 Hugging Face model 仓库。
- `/path/to/checkpoint`：需要上传的 checkpoint 目录，可为 `best`、`last` 或任意兼容目录。
- `--figures-dir`：可选。未指定时使用 checkpoint 的父目录。
- `-h`、`--help`：显示帮助，不执行网络操作。

## 上传结构

checkpoint 目录中的文件直接上传到仓库根目录；图像目录顶层的 `*.png` 上传到 `figures/`：

```text
checkpoint.json
params-*.npz
opt_state-*.npz
opt_state-*.treedef.pkl
figures/
  training_curves.png
  training_overview.png
  bimanual_behavior.png
  gate_diagnostics.png
  bimanual_action_examples.png
```

脚本不要求 optimizer 文件存在，因为部署 checkpoint 只需 `checkpoint.json` 和其引用的参数文件；如果用户指定目录包含 optimizer 文件，则随整个目录一起上传。

## 执行流程

1. 校验参数数量、仓库 ID 格式和未知选项。
2. 将 checkpoint 与图像目录解析为绝对路径。
3. 校验 checkpoint 目录不是符号链接，包含 `checkpoint.json`，且其中 `params_file` 指向目录内存在的普通文件；拒绝绝对路径和目录穿越引用。
4. 校验图像目录存在且不是符号链接，并列出其顶层 PNG。没有 PNG 时中止，避免用户误以为图像已经上传。
5. 查找 `uv`，调用 `uv run --no-sync hf auth whoami` 确认登录。
6. 使用 `hf repo create ... --repo-type model --exist-ok` 创建或复用 public 仓库，不传 `--private`。
7. 上传 checkpoint 目录到仓库根目录。
8. 使用 `--include "*.png"` 将图像目录上传至 `figures/`。
9. 打印仓库网页地址和已上传内容摘要。

任一检查或 HF 命令失败时立即退出非零状态。脚本不删除本地文件，也不修改已存在的 HF 仓库可见性。

## 测试

使用临时目录和伪造的 `uv` 可执行文件运行真实 Bash 脚本，覆盖：

- help 和参数错误；
- 非法仓库 ID；
- 用户指定 checkpoint 与 `--figures-dir` 的参数传递；
- 默认图像目录为 checkpoint 父目录；
- 缺失 checkpoint、参数文件或 PNG 时拒绝上传；
- 符号链接及 `params_file` 路径穿越被拒绝；
- public repo 创建命令不含 `--private`；
- checkpoint 上传到根目录，PNG 上传到 `figures/`。
