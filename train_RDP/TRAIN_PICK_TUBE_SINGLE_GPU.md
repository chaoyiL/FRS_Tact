# Pick-tube PCA30 单卡训练

这套入口面向单张 NVIDIA RTX PRO 6000，训练六个数据集合并后的
PCA30 数据。AT 和 LDP 使用当前项目配置：20 维动作、30 维触觉、
`n_latent_dims=16`、`conv_latent_dims=32`、`rnn_latent_dims=64`、
`n_embed=32`。

## 推荐：统一自动化入口

服务器路径集中写在 `profiles/rtxpro6000x4.yaml`，实验语义集中写在
`recipes/pick_tube_six30.yaml`。默认 profile 中的 `/home/hillbot/datasets`
是路径模板；首次运行前必须按服务器实际目录修改并检查全部 `paths`。
先打印完整计划，再执行：

```bash
python3 scripts/rdpctl.py plan --run-id picktube6-p30-at20-ldp20-v1
python3 scripts/rdpctl.py run --run-id picktube6-p30-at20-ldp20-v1
```

`run` 会执行环境和 CUDA 预检、四卡触觉预计算、PCA30、Zarr 转换、
数据验证以及 AT→LDP 训练。中断后使用同一个 run ID：

```bash
python3 scripts/rdpctl.py resume --run-id picktube6-p30-at20-ldp20-v1
python3 scripts/rdpctl.py status --run-id picktube6-p30-at20-ldp20-v1
```

正式模型选择仍应提供实测 baseline：

```bash
python3 scripts/rdpctl.py run \
  --run-id picktube6-p30-at20-ldp20-v1 \
  --baseline-json /absolute/path/to/frozen_v1_validation_metrics.json
```

默认转换使用 32 个 RGB 解码线程、64-frame RGB chunk，并将分阶段耗时写入
`conversion_metrics.json`。这些参数应根据服务器的本地 NVMe/NFS 基准结果在
recipe 中调整。

## 1. 安装环境

```bash
cd /path/to/FRS_Tact/train_RDP
bash scripts/install_pick_tube_training_env.sh
source .venv/bin/activate
```

默认安装 Python 3.12、PyTorch 2.10 CUDA 13.0 和
`requirements-rdp-training.txt`。可通过环境变量调整安装位置或镜像：

```bash
VENV_DIR=/path/to/.venv \
PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
bash scripts/install_pick_tube_training_env.sh
```

脚本安装的是 `hf` CLI；需要访问私有 Hugging Face 数据时使用：

```bash
.venv/bin/hf auth login
```

## 2. 准备六数据集 PCA30 Zarr

已有 `data/pick_tube_01_06_pca30_rdp_zarr/replay_buffer.zarr` 时跳过转换。
现有 `tactile_pca_2x15.npz` 已由六个数据集共同拟合。

```bash
DATASET_PATH="$PWD/data/pick_tube_01_06_pca30_rdp_zarr" \
TACTILE_PCA_PATH="$PWD/data/PCA_Transform_PickTube/tactile_pca_2x15.npz" \
bash scripts/setup_pick_tube_data.sh convert
```

## 3. 单卡完整训练

冻结 v1 验证结果 JSON 现在是可选的。提供时，两个值必须来自实测，不能混合单位：

```text
{
  "val_active_left_translation_mae_mm": <measured-positive-number>,
  "val_active_left_rotation_mae_deg": <measured-positive-number>
}
```

如果暂时没有 baseline，可以省略 `BASELINE_JSON`。AT/LDP 会正常训练并保存
`latest.ckpt`，但 checkpoint 会保持 `non-deployable`，并且不会进入 top-k；
补齐实测 baseline 后再做正式模型选择。

```bash
GPU_ID=0 \
RUN_ID=pca30_latent32_full6_v1 \
AT_EPOCHS=20 \
LDP_EPOCHS=10 \
AT_BATCH=64 \
LDP_BATCH=64 \
NUM_WORKERS=8 \
AT_CHECKPOINT_EVERY=1 \
LDP_CHECKPOINT_EVERY=1 \
AT_PERIODIC_KEEP=10 \
LDP_PERIODIC_KEEP=10 \
MIXED_PRECISION=bf16 \
bash scripts/train_pick_tube_single_gpu.sh all
```

默认输出为：

```text
data/outputs/pick_tube_01_06/
├── at_pca30_latent32_full6_v1/
│   └── checkpoints/latest.ckpt
└── ldp_pca30_latent32_full6_v1/
    ├── checkpoints/latest.ckpt
    └── normalizer.pkl
```

## 4. 断点续训

使用相同的 `RUN_ID` 再执行同一命令即可。`AT_EPOCHS` 和 `LDP_EPOCHS`
表示目标总 epoch 数，不是额外增加的 epoch 数。例如已有 LDP epoch 7，
`LDP_EPOCHS=10` 会继续训练到总计 10 个 epoch。

只训练 LDP 时显式指定 AT：

```bash
GPU_ID=0 \
RUN_ID=pca30_latent32_full6_v1 \
BASELINE_JSON=/absolute/path/to/frozen_v1_validation_metrics.json \
AT_CKPT=/absolute/path/to/at/checkpoints/latest.ckpt \
bash scripts/train_pick_tube_single_gpu.sh ldp
```

只打印命令、不启动训练：

```bash
DRY_RUN=1 RUN_ID=pca30_latent32_full6_v1 \
BASELINE_JSON=/absolute/path/to/frozen_v1_validation_metrics.json \
bash scripts/train_pick_tube_single_gpu.sh all
```
