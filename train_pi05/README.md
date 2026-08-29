# 纯视觉 pi0.5 JAX 训练

这是独立的纯视觉 pi0.5 微调项目，目录结构与 `train_pi05_frs` 对齐：

- `configs/train_pi05.yaml`：数据集、模型档位和训练参数。
- `../scripts/setup_env.sh --pi05_train`：通过仓库统一入口创建 Python 3.11 训练环境和 Python 3.12 数据转换环境。
- `scripts/start_pi05_train.sh`：配置预检、tmux 和正式训练入口。
- `scripts/start_pi05_right_train.sh`：固定使用右手单臂配置的一键启动入口。
- `train.py`：将 YAML 转成 openpi JAX 训练配置。
- `tools/`：底层训练循环、数据检查、归一化统计和 smoke test。
- `src/openpi`：pi0.5 模型和训练循环。
- `src/lerobot`：项目私有的 LeRobot v3 只读数据运行时。

AnyTouch、触觉模型分支、触觉权重加载器以及 vitac 训练配置均已删除。

## 多数据集

在 YAML 的 `datasets` 中按顺序增加数据集。训练端逐项打开后使用
`ConcatDataset` 组成一个数据流；启用训练 shuffle 时会在合并后的所有帧间打乱。
每个数据集可以有独立的 `repo_id`、本地 `root`、`revision`、`episodes` 和
`action_key`，但必须遵守所选 profile 相同的相机、state 和 action 维度契约。

```yaml
datasets:
  - repo_id: KaiyueChen/pick_tube_05
    root: /workspace/lerobot_v30/KaiyueChen/pick_tube_05
    action_key: action
  - repo_id: KaiyueChen/pick_tube_06
    root: /workspace/lerobot_v30/KaiyueChen/pick_tube_06
    action_key: action
```

下载/转换多个数据集时可重复传参：

```bash
bash scripts/download_data.sh \
  --dataset pick_tube_05 \
  --dataset pick_tube_06
```

本仓库的 v2.1 → v3.0 转换器会在首次生成 Parquet 和统计文件时直接把旧字段
`actions` 写成 LeRobot 标准字段 `action`，转换后只做只读校验，不再逐分片二次 rewrite。

## 环境与训练

建议使用 Linux/WSL2、Python 3.11 和 NVIDIA GPU：

```bash
bash scripts/setup_env.sh --pi05_train
# 训练入口会自动读取仓库根目录的 env_path，无需手动激活虚拟环境。
bash train_pi05/scripts/start_pi05_train.sh --check
bash train_pi05/scripts/start_pi05_train.sh
```

Pi0.5 JAX 训练固定使用 `/workspace/venvs/pi05_train`（Python 3.11）；
`download_data.sh` 固定使用 `/workspace/venvs/lerobot_data_tools`（Python 3.12）。
这是因为当前 LeRobot 转换器包含 Python 3.12 语法，两者不能共用解释器。

### 新服务器单张 RTX 4090 完整链路测试

新服务器需要上述两套环境，但只需要执行一个入口。下面的测试固定使用
`insert_01`，检查 7D state、10D action 和 `camera1`，抽样 512 帧生成测试专用
归一化统计，然后执行 2 步真实 JAX 前向、反向和参数更新，最后检查 checkpoint：

```bash
cd /workspace/FRS_Tact
bash train_pi05/scripts/test_pi05_4090_insert01.sh --setup
```

如果两套环境和转换后的数据已经存在，可以跳过安装与下载：

```bash
bash train_pi05/scripts/test_pi05_4090_insert01.sh --skip-download
```

首次运行还会下载 `pi05_base`，并进行耗时较长的 XLA 编译。测试输出位于
`/workspace/outputs/pi05_4090_insert01_smoke`。抽样统计只用于验证链路，正式训练
前仍应使用正式 YAML 对完整训练数据重新运行 `compute_norm_stats.py`。

如果服务器访问 PyPI 较慢，可以仅在安装时指定镜像；PyTorch CPU wheel 仍从
PyTorch 官方索引获取，JAX CUDA wheel 和其他包从所选 PyPI 镜像获取：

```bash
FRS_PYPI_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple \
  bash scripts/setup_env.sh --pi05_train
```

不要把 `FRS_PYTORCH_INDEX` 设置为阿里云的 `pytorch-wheels/cpu` 地址；该索引
可能缺少本项目固定的 `torch==2.7.1`。不设置时会使用项目中配置的 PyTorch
官方 CPU wheel 索引。

Pi0.5 使用 JAX 访问 GPU。环境中的 PyTorch 只用于读取 LeRobot 数据，因而固定使用
CPU wheel，避免重复下载 PyTorch CUDA/NVIDIA 运行库。项目已经内置所需的 LeRobot v3
只读运行时代码，不需要在安装时访问 GitHub 克隆另一份 LeRobot。

### 右手单臂训练

右手训练使用 `configs/train_pi05_right.yaml` 和 `pi05_single` 档位，固定合同为
7D `observation.state`、10D `actions` 和有效右手视觉图像
`observation.images.camera1`。`insert_02` 的 `camera0` 是黑色占位图，训练不会读取它。
训练变换会把 `camera1` 放入模型的
`right_wrist_0_rgb` 槽位，与右手部署配置一致。

配置中的数据必须已经是右手单臂数据。脚本不会自动从双臂 20D state/action
中猜测右手切片；如果手里只有双臂数据，应先按机器人的真实字段定义转换为
7D/10D 单臂数据，再计算对应的归一化统计。

```bash
# 先修改 train_pi05/configs/train_pi05_right.yaml 中的数据路径和 norm_stats。
source env_path
"$TRAIN_PI05_PYTHON" train_pi05/tools/compute_norm_stats.py train_pi05/configs/train_pi05_right.yaml
bash train_pi05/scripts/start_pi05_right_train.sh --check
bash train_pi05/scripts/start_pi05_right_train.sh
```

`source env_path` 后也可以使用不歧义的快捷命令：`hf`、`wandb`、
`data-python`、`pi05-python`、`pi05-deploy-python` 和 `smolvla-python`。

当前右手配置按顺序合并 `insert_01` 和 `insert_02`，随后在 563,414 帧组成的
统一数据流上 shuffle。两个数据集按帧数自然采样，约为 45% 与 55%；归一化统计
必须通过上面的 YAML 命令重新计算，不能复用任一单数据集的 stats。

辅助工具需要时单独调用，例如：

```bash
source env_path
"$TRAIN_PI05_PYTHON" train_pi05/tools/smoke_test.py
"$TRAIN_PI05_PYTHON" train_pi05/tools/check_dataset.py --config-name pi05_bi
"$TRAIN_PI05_PYTHON" train_pi05/tools/compute_norm_stats.py pi05_bi
```

`training.overwrite` 与 `training.resume` 不能同时为 `true`。正式训练前应确保
`norm_stats` 与所有训练数据的 state/action 定义一致；多数据集共用同一套归一化统计。
