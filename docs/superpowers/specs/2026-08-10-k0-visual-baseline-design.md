# K0 纯视觉公平基线设计

## 目标

新增一个不输入触觉的 SmolVLA K0 基线，在相同六数据集、数据划分、normalization、模型训练模式和 80k 训练预算下，与 K1/K4/K21 比较。

## 配置

新增 `configs/train_vtsmolvla_jax_visual_k0.yaml`。它以 K1 配置为公平性来源，仅改实验身份和触觉开关：`use_tactile_encoder: false`、不训练 `tactile_proj`、不配置 tactile encoder/keys、关闭 tactile embedding cache；离线 frozen vision/connector cache 保持启用。训练使用全局 `batch_size: 192`，单个 JAX 进程在两张 GPU 上进行 data parallel，每卡 96；`steps: 80000`、`save_freq: 10000`、`scheduler_decay_steps: 80000`。

现有 `configs/train_smolvla_jax.yaml` 仅作为纯视觉字段语义参考，不直接修改或作为实验配置，因为它的数据集和训练协议与 VT 实验不同。

## 启动链路

扩展通用 cache/launcher：当 `model.use_tactile_encoder=false` 时跳过 tactile cache，并调用 `tools/train_smolvla_jax.py`；触觉配置继续调用 `tools/train_vtsmolvla_jax.py`，行为不变。新增 `scripts/start_smolvla_k0_train.sh`，将 K0 config 和全部 CLI 参数转发给现有两卡 launcher。

## 失败与验证

K0 仍使用现有 GPU、数据、output 和 checkpoint preflight，任何失败均非零退出。按用户要求不运行测试套件，只进行 YAML 解析、两卡 batch 整除、Bash 语法、入口选择和 `git diff --check` 静态验证。
