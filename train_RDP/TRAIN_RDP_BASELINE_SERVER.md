# 服务器四任务原始 RDP

代码目录：`/home/ljl/FRS_Tact`。原始数据根目录：`/DATA/ljl/substage/lerobot_v21/KaiyueChen`。

使用基础 RDP 的 L1+KL、随机后验 latent、扩散 MSE 和固定 chunk 基底；不走旧服务器脚本的动作改写、加权损失和发布门槛。每个任务有自己的 PCA、AT、LDP，数据集内各 episode 普通采样，不人为重复某一个数据集。

| GPU | 顺序 | 模型的数据集 | 默认机械臂 |
|---|---|---|---|
| 0 | 1 | insert_01 + insert_02 | 右臂 |
| 0 | 2 | two_tubes_01 + two_tubes_02 + two_tubes_03 + two_tubes_04 | 双臂 |
| 1 | 1 | press_01 + press_02 | 右臂 |
| 1 | 2 | bread_01 + bread_02 + bread_03 | 双臂 |

每卡同一时刻运行一个任务，任务内部依次执行触觉编码、独立 PCA、原始数据转换、AT、LDP。GPU 0 和 GPU 1 的队列并行。一个任务失败会停止同卡队列，另一卡继续；主进程最终返回非零状态。

机械臂默认值依据已有 insert/press/bread 配置，two_tubes 暂按双臂。如需改变，启动时设置 `TWO_TUBES_ARMS=right` 或其他任务的 `INSERT_ARMS` / `PRESS_ARMS` / `BREAD_ARMS`，值为 `right` 或 `both`。它会同步切换数据与模型输入维度。双臂源数据遵循既有 state20/action20 合同，单右臂源数据遵循 state7/action10 合同；选择须与数据实际格式一致。

## 更新与启动

本机推送的分支为 `eric`。在服务器执行：

```bash
cd /home/ljl/FRS_Tact
git pull --ff-only origin eric
DRY_RUN=1 bash train_RDP/scripts/server_ljl_baseline_four_tasks.sh all
```

正式启动：

```bash
cd /home/ljl/FRS_Tact
mkdir -p /DATA/ljl/substage/rdp_original/logs
nohup env GPU_IDS=0,1 NUM_WORKERS=32 RUN_ID=original_rdp_v1 \
  bash train_RDP/scripts/server_ljl_baseline_four_tasks.sh all \
  > /DATA/ljl/substage/rdp_original/logs/original_rdp_v1.log 2>&1 &
```

默认 AT 601 epoch、LDP 401 epoch、两者每卡 batch=64，采用恢复的原版默认值。若要延用此前服务器的 20/10 epoch 训练时长，可在 `env` 后加 `AT_EPOCHS=20 LDP_EPOCHS=10`。AT 用 FP32；LDP 默认 BF16，可设置 `MIXED_PRECISION=no`。

日志：

```bash
tail -f /DATA/ljl/substage/rdp_original/logs/original_rdp_v1.log
tail -f /DATA/ljl/substage/rdp_original/outputs/insert/original_rdp_v1/pipeline.log
```

总日志实时汇总两张 GPU 的输出，带时间、GPU、任务名和阶段前缀，例如 `[21:10:00][GPU 0][insert][AT 4/5]`。它显示训练参数、数据集列表、各阶段开始/完成/耗时、原始数据准备进度、训练 loss 与进度条、错误信息。每个任务的 `pipeline.log` 同时保留该任务的输出，`pipeline.json` 保存实际命令。

子进程无输出时，每隔 30 秒报告当前阶段的 `RUNNING` 和已用时间；这表示进程尚未退出，不代表已经完成新的训练步骤。可通过 `LOG_HEARTBEAT_SECONDS` 调整间隔。

已经运行的旧进程不会因 `git pull` 自动改变日志方式。当前任务可以直接用 `tail -F` 同时跟踪 insert/press 的 `pipeline.log`；新启动的队列使用上述汇总日志。单独运行或续训：

```bash
# 只准备全部数据，仍然每 GPU 一路。
bash train_RDP/scripts/server_ljl_baseline_four_tasks.sh prepare

# 从已有数据训练某个任务，绑定 GPU 1。
GPU_IDS=1 RUN_ID=bread_original_v1 \
bash train_RDP/scripts/server_ljl_baseline_four_tasks.sh train bread

# 用相同 RUN_ID 恢复四个任务，num_epochs 是总目标轮数。
RESUME=true RUN_ID=original_rdp_v1 \
bash train_RDP/scripts/server_ljl_baseline_four_tasks.sh train
```

阶段支持 `all`、`precompute`、`prepare`、`train`、`at`、`ldp`，后面可指定一个或多个任务。省略任务默认全四个；任务按列出的顺序轮流分到 `GPU_IDS` 指定的卡。

## 环境和产物

默认优先使用当前代码目录下 `train_RDP/.venv`、`.venv-jax`；不存在时使用旧服务器脚本已有的 `/home/ljl/RDP_vitamin/.venv`、`.venv-jax`。encoder 优先复用 `/DATA/ljl/substage/rdp_single_right/encoder_ckpt_0824`，否则使用 `/home/ljl/RDP_vitamin/data/encoder_ckpt_0824`。可用 `PYTHON_BIN`、`JAX_PYTHON`、`ENCODER_DIR` 显式指定已有环境和模型。脚本不安装环境或下载权重。

`WORK_ROOT` 默认 `/DATA/ljl/substage/rdp_original`，与旧实验目录独立：

```text
WORK_ROOT/
  tactile_embeddings_encoder0824/KaiyueChen/<dataset>/
  pca/<task>/tactile_pca.npz
  datasets/<task>/replay_buffer.zarr
  datasets/<task>/raw_tactile_manifest.json
  datasets/<task>/prepare_manifest.json
  outputs/<task>/<RUN_ID>/at/checkpoints/latest.ckpt
  outputs/<task>/<RUN_ID>/ldp/checkpoints/latest.ckpt
```

触觉原始缓存通过 manifest 分片引用，不再为了合并任务复制完整 embedding 数组。已完成且输入匹配的数据转换可复用；不匹配会给出错误，不覆盖旧产物。`TACTILE_CACHE_ROOT` 可指向已经完整生成的同一 encoder 缓存根目录。每个训练/验证 DataLoader 默认 32 workers，预编码和图像转换也默认 32；可分别设置 `PRECOMPUTE_WORKERS`、`CONVERT_WORKERS`。

新权重的动作表示是整段共用固定基底。部署必须使用对应的单臂或双臂观测与动作转换，不能直接套用旧 0902 的逐帧增量执行逻辑。

本机验证：35 项测试通过；真实 insert_01 前两段共 1228 帧的数据转换、加载与归一化通过；双相机 ResNet18 + 30D 触觉 + 20D 动作的 CPU loss/backward/采样通过（该短测试缩小 UNet 并使用 2 步采样）。服务器不可直连，尚未在服务器执行 pull 或启动 GPU 训练。
