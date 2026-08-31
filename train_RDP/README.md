# train_RDP

独立的 Pick Tube Reactive Diffusion Policy 训练子项目。服务器使用的代码目录是
`/home/ljl/FRS_Tact/train_RDP`。内部顶层包名继续使用
`reactive_diffusion_policy`，以兼容 Hydra `_target_` 和 checkpoint。

现有默认双臂合同是 PCA30、20D 动作、AT→LDP 两阶段训练，并带 dataset、PCA、
normalizer、AT/LDP 的 artifact identity 校验。详细说明见
`TRAIN_PICK_TUBE_SINGLE_GPU.md`。

## 环境

```bash
bash train_RDP/scripts/install_pick_tube_training_env.sh
```

训练环境固定 Python 3.12、PyTorch 2.10/CUDA 13、NumPy 1.26 和 Zarr 2，不与仓库
根环境混用。触觉预计算继续使用独立的 `.venv-jax`。

## 推荐工作流

```bash
cd train_RDP

# 先查看由 profile + recipe 生成的完整执行计划
python3 scripts/rdpctl.py plan --run-id picktube6-p30-at20-ldp20-v1

# 执行或恢复
python3 scripts/rdpctl.py run --run-id picktube6-p30-at20-ldp20-v1
python3 scripts/rdpctl.py resume --run-id picktube6-p30-at20-ldp20-v1
python3 scripts/rdpctl.py status --run-id picktube6-p30-at20-ldp20-v1
```

单卡 AT→LDP 入口：

```bash
DRY_RUN=1 bash scripts/train_pick_tube_single_gpu.sh all
bash scripts/train_pick_tube_single_gpu.sh all
```

没有实测 baseline 时 checkpoint 会保留为 `non-deployable`；不得将 smoke 或
non-deployable checkpoint 用于真机。


## `/home/ljl` 服务器：单右臂 Insert / Press

服务器脚本已经固定以下路径：

- RDP 代码：`/home/ljl/FRS_Tact/train_RDP`
- 原始数据：`/DATA/ljl/substage/lerobot_v21/KaiyueChen`
- encoder、中间数据和训练输出：`/DATA/ljl/substage/rdp_single_right`

本服务器脚本使用 PyTorch 2.10/CUDA 12.8 和 JAX 0.8.3/CUDA 12。训练分为两个互相隔离的环境：`.venv-jax` 只读取四路触觉图像，使用
`KaiyueChen/encoder_ckpt_0824` 生成 `[N, 4, 512]` embedding；`.venv` 完成
PCA30、Zarr 转换以及 PyTorch AT -> LDP 训练。不要把两个环境合并。

```bash
cd /home/ljl/FRS_Tact/train_RDP

# 只需首次执行：配置两个环境并从 HF 镜像下载 encoder_ckpt_0824
bash scripts/server_ljl_single_right.sh setup

# 正式运行前检查 GPU、encoder 以及三个数据集的 shape/fps 合同
bash scripts/server_ljl_single_right.sh doctor both

# 两个模型顺序训练：insert_01+insert_02 一个，press_01+press_02 一个
bash scripts/server_ljl_single_right.sh all both
```

也可以分别执行，断点重跑会复用已有 embedding、PCA、Zarr 和 checkpoint：

```bash
bash scripts/server_ljl_single_right.sh all insert
bash scripts/server_ljl_single_right.sh all press
```

`prepare` 会自动完成触觉 embedding、独立 PCA30 和 Zarr 转换；`train` 只训练已有
Zarr。默认使用 GPU 0；触觉预计算和 AT 的 batch size 是512，LDP物理 batch size
是128，workers都是32；训练使用bf16、AT 20 epoch、LDP 10 epoch。需要时仍可覆盖。

## `/home/ljl` 服务器：双臂 Bread

`server_ljl_bread_dual.sh` 会把 `bread_01`、`bread_02`、`bread_03` 合并成一个
双臂数据集，使用 `dual-arm-20x20` 合同（20D state、20D action）训练一个模型。
它固定复用 `/home/ljl/RDP_vitamin/.venv`、`.venv-jax` 和
`data/encoder_ckpt_0824`；`all` 不会创建或下载环境。

```bash
cd /home/ljl/FRS_Tact/train_RDP

# 先检查三个数据集、GPU、现有环境和 encoder
GPU_ID=1 bash scripts/server_ljl_bread_dual.sh doctor

# 生成/续算 embedding、PCA30、RDP Zarr，然后训练 AT -> LDP
GPU_ID=1 EXPERIMENT_ID=bread_v1 \
  bash scripts/server_ljl_bread_dual.sh all
```

中间数据位于 `/DATA/ljl/substage/rdp_bread_dual`，权重分别保存在其
`outputs/bread_01_02_03/at_*` 和 `outputs/bread_01_02_03/ldp_*` 下的
`checkpoints/latest.ckpt`。默认预计算 batch/workers 为512/32，AT batch为512，
LDP物理 batch为64，训练 workers为32；可以用同名环境变量覆盖。
