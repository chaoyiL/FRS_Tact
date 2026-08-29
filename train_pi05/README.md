# 纯视觉 pi0.5 JAX 训练

这是独立的纯视觉 pi0.5 微调项目，目录结构与 `train_pi05_frs` 对齐：

- `configs/train_pi05.yaml`：数据集、模型档位和训练参数。
- `../scripts/setup_env.sh --pi05_train`：通过仓库统一入口创建本目录专用 `.venv`。
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

## 环境与训练

建议使用 Linux/WSL2、Python 3.11 和 NVIDIA GPU：

```bash
bash scripts/setup_env.sh --pi05_train
bash train_pi05/scripts/start_pi05_train.sh --check
bash train_pi05/scripts/start_pi05_train.sh
```

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
cd train_pi05
uv run python tools/compute_norm_stats.py configs/train_pi05_right.yaml
cd ..
bash train_pi05/scripts/start_pi05_right_train.sh --check
bash train_pi05/scripts/start_pi05_right_train.sh
```

当前右手配置按顺序合并 `insert_01` 和 `insert_02`，随后在 563,414 帧组成的
统一数据流上 shuffle。两个数据集按帧数自然采样，约为 45% 与 55%；归一化统计
必须通过上面的 YAML 命令重新计算，不能复用任一单数据集的 stats。

辅助工具需要时单独调用，例如：

```bash
cd train_pi05
uv run python tools/smoke_test.py
uv run python tools/check_dataset.py --config-name pi05_bi
uv run python tools/compute_norm_stats.py pi05_bi
```

`training.overwrite` 与 `training.resume` 不能同时为 `true`。正式训练前应确保
`norm_stats` 与所有训练数据的 state/action 定义一致；多数据集共用同一套归一化统计。
