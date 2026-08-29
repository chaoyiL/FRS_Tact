# train_RDP

独立的 Pick Tube Reactive Diffusion Policy 训练子项目。代码来自
`RDP_vitamin` 的 `agent/rdp-pick-tube-deployment` 分支，精确 revision 记录在
`SOURCE_REVISION`。内部顶层包名继续使用 `reactive_diffusion_policy`，以兼容 Hydra
`_target_` 和 checkpoint。

本版本的正式合同是 PCA30、20D 双臂动作、AT→LDP 两阶段训练，并带 dataset、PCA、
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
