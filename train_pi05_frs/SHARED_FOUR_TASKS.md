# 四个冻结 Pi0.5，共享一个触觉 FRS

本流程只更新一个 FRS 的 GRU、Transformer 和触觉位置标识参数。四个任务各自已有的
Pi0.5 checkpoint，以及已有的触觉 ResNet encoder，均保持冻结。

| 任务 | 数据集 | FRS 有效手侧 | 训练 episode | 验证 episode |
| --- | --- | --- | ---: | ---: |
| insert | KaiyueChen/insert_01 | right | 300 | 35 |
| press | KaiyueChen/press_01 | right | 300 | 35 |
| bread | KaiyueChen/bread_03 | left、right | 300 | 35 |
| pick | KaiyueChen/pick_01 | left、right | 300 | 35 |

insert、press 的原始 state 是 7D、action 是 10D；bread、pick 的 state/action 都是 20D。
这次四任务清单以你提供的 insert、press、bread、pick 为准，替换之前的 two_tubes。

## 新服务器上的完整步骤

先把本次代码修改同步到新服务器，代码目录以下按 `/workspace/FRS_Tact`。

```bash
cd /workspace/FRS_Tact
export FRS_WORKSPACE_ROOT=/workspace
export FRS_STORAGE_ROOT=/workspace
export UV_CACHE_DIR=/workspace/uv-cache
export UV_PYTHON_INSTALL_DIR=/workspace/uv-python

bash scripts/setup_env.sh --pi05_frs_train
source ./env_path
```

这会创建 `/workspace/venvs/pi05_frs_train` 和独立的数据工具环境
`/workspace/venvs/lerobot_data_tools`。安装脚本会检查 GPU；系统 NVIDIA 驱动和可用的 CUDA/cuDNN
仍由服务器镜像提供。这里使用 Pi0.5 FRS 固定的 Python 3.12、JAX 0.5.3、Flax 0.10.2。

四任务复用的冻结 encoder 固定为 `KaiyueChen/encoder_ckpt_0824`，
revision `1a432a8b9c1a38f75cb40c58a4924a401486d832`。已核对其 metadata：
ResNet embedding 为 512D，输入图像为 224，与本配置一致。
在仓库根目录、执行过 `source ./env_path` 后下载：

```bash
"$TRAIN_PI05_FRS_PYTHON" deploy_smolvla/src/download_ckpt.py \
  --repo-id KaiyueChen/encoder_ckpt_0824 \
  --revision 1a432a8b9c1a38f75cb40c58a4924a401486d832 \
  --output-dir /workspace/FRS_Tact/checkpoints/encoder_ckpt_0824 \
  --minimal
```

该下载器也用于 tactile encoder，校验 `checkpoint.json` 指向的参数归档；这里使用专用 FRS
解释器运行，不安装额外的根目录 SmolVLA 环境。`--minimal` 只下载 metadata 和模型参数，
不下载 encoder 的 optimizer 或 memory bank。四个任务用同一份权重，训练 FRS 时保持冻结。

四个纯视觉 Pi0.5 已在 `train_pi05_frs/configs/train_pi05_frs_shared_four_tasks.yaml`
中配置并固定 commit，缓存生成时保持冻结：

| 任务 | Hugging Face 权重 | normalization asset |
| --- | --- | --- |
| insert | KaiyueChen/pi05_task3_0830_1w | insert_0102_train90 |
| press | KaiyueChen/pi05_task4_0830_6k | press_0102 |
| bread | KaiyueChen/pi05_task2_0902_12k | bread_01020304 |
| pick | KaiyueChen/pi05_task1_0902_8k | pick_0102 |

一键下载这四个 VLA，不需要再训练它们：

```bash
bash scripts/download_frs_shared_ckpts.sh --dry-run
bash scripts/download_frs_shared_ckpts.sh
```

默认保存到 `/workspace/FRS_Tact/checkpoints/model/<权重仓库名>`，与训练配置一致。
只下载完整 Orbax `params/`、根 metadata 以及各自的 `norm_stats.json`，不下载 optimizer
`train_state/`，四份合计约 23.6 GiB。下载后检查文件大小和统计量维度，重复运行可续传。
下载目录带有 `.frs_pi05_source.json` 来源记录；若目录已有其他来源的内容，需在配置中指定
新的空目录。这里用专用下载入口；`scripts/download_ckpt.sh` 无参数时仍是 SmolVLA 流程。

先下载数据。`--dry-run` 只读取 metadata、显示选中的文件数量和体积：

```bash
bash scripts/download_frs_shared_data.sh --dry-run
bash scripts/download_frs_shared_data.sh
```

下载器只获取各任务随机选中的 335 条 v2.1 episode Parquet，保留 state、action、RGB、触觉全部列，
再通过仓库现有转换器生成完整 v3 子集。生成的 v3 episode ID 从 0 连续编号；原始 ID 对照表
保存在每个数据集根目录的 `frs_subset_source.json`。后续 FRS 缓存按完整 episode 分成 300/35，
最终训练/验证归属以 action cache 的 `split.npy` 和训练清单为准，训练不会重新跨拆分采样。

默认目录均在 `/workspace/frs_shared_four_tasks/`：

```text
datasets/raw/        # 选中的原始 Parquet、下载计划和固定来源 commit
datasets/v30/       # 四个完整可读取的本地 v3 子集
native_configs/     # 为各个原始 Pi0.5 自动生成的独立缓存配置
logs/               # 每次启动的 pipeline_*.log，包含各缓存阶段和训练输出
action_cache/       # 四个 VLA 各自产生的 x_base / predicted / GT / state
tactile_embeddings/ # 冻结 encoder 的逐帧 embedding
runs/shared_frs/    # 唯一一套共享 FRS checkpoint 和验证结果
```

数据体积以 `--dry-run` 输出为准，转换结果会额外占用空间，转换期间也有临时副本；
加上约 23.6 GiB 的四个 Pi0.5、encoder 和环境缓存，建议预留约 200 GB 或更多。重复运行会复用匹配的已完成子集，
不会覆盖其他已有数据。修改来源、seed、数量时使用新的 workspace/dataset 根目录。

检查四任务输入，然后启动：

```bash
bash train_pi05_frs/scripts/start_frs_pi05_shared_train.sh --check

CUDA_VISIBLE_DEVICES=0 \
  bash train_pi05_frs/scripts/start_frs_pi05_shared_train.sh

tmux attach -t frs_pi05_shared_train
```

`--check` 不导入 JAX、不初始化 GPU、不创建输出目录；只检查配置、checkpoint、数据、统计量和运行环境。
启动器默认用 tmux 持续运行。设 `FRS_FOREGROUND=1` 可在当前终端执行。

正常流程按 source 顺序执行：各自 Pi0.5 checkpoint smoke → 冻结触觉 embedding → 各自 Pi0.5 action cache；
四份缓存完成后，创建一个共享 FRS 并联合优化。每个 Pi0.5 在单独子进程中生成缓存，进程结束即释放显存。

如果先准备 insert 的缓存，可执行：

```bash
bash train_pi05_frs/scripts/start_frs_pi05_shared_train.sh --prepare-task insert
```

这只准备 insert 的冻结缓存。之后正常启动仍会补齐其他三个任务，再训练四任务共享 FRS。
若四份缓存均已完成，可直接训练修复网络：

```bash
bash train_pi05_frs/scripts/start_frs_pi05_shared_train.sh --train-only
```

`train.py` 也提供对应入口（应在专用 FRS 环境、私有 `lerobot` 路径下运行）：

```bash
PYTHONPATH="$PWD/train_pi05_frs/src:$PWD" "$TRAIN_PI05_FRS_PYTHON" \
  -m train_pi05_frs.train \
  --shared-config train_pi05_frs/configs/train_pi05_frs_shared_four_tasks.yaml \
  --train-only
```

## 共享网络与训练参数

缓存生成保持每个 Pi0.5 自己的 native action_dim（10、20 或 32），不修改已有模型结构。
FRS 输入输出统一为 20D：左手 `0:10`、右手 `10:20`。insert、press 的 10D 缓存放到右侧；
单手样本的另一侧动作和触觉被屏蔽。损失按有效动作维度归一化，每个 ODE 积分步骤都对空闲手速度置零，
cross-attention 屏蔽空闲传感器。四路触觉有可训练的位置标识，保留左右手、两面的对应关系。

默认 `loss_mode: bimanual_gated` 复用现有分腕三段式 VLA/mixed/GT 端点 flow matching，以及
decode、low-safety、rank、repair 辅助项。空闲手不参与低/高 Gate 分组统计。
Gate 是训练目标/辅助权重，沿用现有 flow decoder 的含义，没有切换到 learned residual Gate。

默认 batch 128、50 epoch、学习率 `1e-4`、warmup 2 epoch，每 5 epoch 验证。
`balance_tasks: true` 按每任务有效帧数的倒数平衡 FM/decode 的训练贡献；安全约束沿用原来的
source 分组归一化。每任务 300 条轨迹不会保证其帧数也相同。
验证分别报告各任务及有效手侧，best 优先满足所有已有验证 Gate 分组的安全/修复约束，
然后比较最差任务 MSE 和任务平均 MSE，避免只用 pooled frame MSE 选 checkpoint。

共享 FRS 使用触觉条件，不混合不同原始语义的 7D/20D normalized state。
各自 Pi0.5 的 state 输入和预处理完整保留。触觉 ResNet 冻结，FRS 的 GRU/Transformer 可训练。

## 保存、恢复和先验证 insert

`runs/shared_frs/` 下生成 `last/`、`best/`、`history.csv`、`shared_training_manifest.json`，
以及逐次验证的 `validation_epoch_*.json`。清单记录全部来源和精确 episode 划分。

```bash
# 从最近完整 epoch 恢复，同一配置、同一缓存。
bash train_pi05_frs/scripts/start_frs_pi05_shared_train.sh --train-only --resume

# 加载共享 best，在 insert 验证集上输出报告。
bash train_pi05_frs/scripts/start_frs_pi05_shared_train.sh --evaluate-only --task insert
```

insert 报告为 `runs/shared_frs/evaluation_insert.json`。
`shared_model.repair_native_actions` 提供 insert 的 native 10D → 20D → 10D 适配，输入必须是原始
Pi0.5 反向积分得到的 `x_base` 和冻结 encoder 的触觉历史 embedding。归一化与反归一化仍使用
该任务自己的统计量。20/32D 接口中空闲/填充维度必须从冻结 VLA 的 predicted endpoint 恢复。

现有部署端不能仅替换 checkpoint 路径就把这份共享 20D 模型当作旧 10D 模型加载；
本次提供训练、离线验证和模型级适配，实机部署入口接线需要单独完成。
