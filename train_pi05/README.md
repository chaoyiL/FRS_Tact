# 纯视觉 pi0.5 JAX 训练

这是独立的纯视觉 pi0.5 微调项目，目录结构与 `train_pi05_frs` 对齐：

- `configs/train_pi05.yaml`：数据集、模型档位和训练参数。
- `scripts/setup_env.sh`：创建本目录专用 `.venv`。
- `scripts/start_pi05_train.sh`：配置预检、tmux 和正式训练入口。
- `train.py`：将 YAML 转成 openpi JAX 训练配置。
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
bash train_pi05/scripts/setup_env.sh
bash train_pi05/scripts/start_pi05_train.sh --check
bash train_pi05/scripts/start_pi05_train.sh
```

`training.overwrite` 与 `training.resume` 不能同时为 `true`。正式训练前应确保
`norm_stats` 与所有训练数据的 state/action 定义一致；多数据集共用同一套归一化统计。

