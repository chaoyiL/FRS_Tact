# train_RDP

独立的 Pick Tube Reactive Diffusion Policy 训练子项目。代码来自
`RDP_vitamin` 的 `agent/rdp-pick-tube-deployment` 分支，精确 revision 记录在
`SOURCE_REVISION`。内部顶层包名继续使用 `reactive_diffusion_policy`，以兼容 Hydra
`_target_` 和 checkpoint。

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


## 单右臂 RDP（Insert 01 + 02）

`train_deco` 中 Insert 数据的合同已移植为独立 RDP profile：
`single-right-arm-7x10`。其中 state 是右臂 7D，action 是右臂 10D
（3D 相对位移、6D 旋转、夹爪宽度）；现有 `dual-arm-20x20` 流程保持不变。
RGB 仍使用 camera0/camera1，触觉仍使用四传感器并压缩为 PCA30。

```bash
cd train_RDP

# 安装训练环境并登录（私有数据集才需要 token）
bash scripts/install_pick_tube_training_env.sh
.venv/bin/hf auth login

# 下载固定 revision 的 insert_01/insert_02 和 RDP 已验证的 0809 编码器
bash scripts/setup_pick_tube_single_right_data.sh datasets
bash scripts/setup_pick_tube_single_right_data.sh encoder

# JAX 环境生成触觉特征；随后拟合 PCA30
JAX_PYTHON=/absolute/path/to/jax/bin/python \
  bash scripts/setup_pick_tube_single_right_data.sh precompute
bash scripts/setup_pick_tube_single_right_data.sh pca

# 建议先转换每个数据集 1 个 episode，再转换全量
bash scripts/setup_pick_tube_single_right_data.sh smoke
bash scripts/setup_pick_tube_single_right_data.sh convert

# AT -> LDP 单卡训练；同一个 RUN_ID 可断点续训
RUN_ID=insert_single_right_v1 \
  bash scripts/train_pick_tube_single_right_gpu.sh all
```

服务器数据不在 `/home/hillbot/datasets` 时，通过 `LEROBOT_ROOT` 指定；输出路径可用
`DATASET_PATH`、`TACTILE_CACHE_ROOT`、`TACTILE_PCA_PATH` 和 `OUTPUT_ROOT` 覆盖。
若提供正式 baseline JSON，单手配置读取
`val_active_right_translation_mae_mm` 和 `val_active_right_rotation_mae_deg`。
