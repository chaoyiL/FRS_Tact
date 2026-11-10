# 新 4090 实例：四任务共享 Pi0.5 FRS

先将本次最新代码同步到 `/workspace/FRS_Tact`。以下按一张 24GB RTX 4090、Linux GPU
实例编写。每一步成功后再执行下一步。所有模型、环境、数据和训练结果保存在 workspace；
可重建的 Arrow 临时副本放到容器盘 `/tmp`。

新配置为 `configs/train_pi05_frs_shared_four_tasks_4090.yaml`：Pi0.5 action cache batch 4、
触觉预计算 batch 32、FRS 训练 batch 32。其余网络结构、flow matching 目标、四个冻结
Pi0.5、冻结 encoder、每任务 300 训练 + 35 验证均保持一致。它是小 batch 起点，
仍需上机确认实际显存；若 action cache OOM，可将 `cache.batch_size` 再降至 2 或 1。

如果新实例挂载的是原来的 workspace，已有权重和缓存可以复用。尤其原始 Parquet 已清理、
v3 数据仍完整时，不必重新运行数据下载。安装环境后可先用第 5 步的 `--check` 判断输入是否齐全。

## 1. 检查 GPU 和容量

```bash
cd /workspace/FRS_Tact
nvidia-smi
df -h / /workspace /tmp
test -f train_pi05_frs/configs/train_pi05_frs_shared_four_tasks_4090.yaml
```

同时核对 Runpod 控制台给这个卷分配的实际容量；共享挂载的 `df` 数值不能代替个人配额。
之前整个 workspace 已超过 120G，新盘建议留约 200GB 或更多用于下载、转换、环境和训练缓存。
容器临时盘建议至少有 30GB 空闲，足以容纳当前单个任务的 Arrow 副本并留出余量。

## 2. 安装专用环境

```bash
export FRS_WORKSPACE_ROOT=/workspace
export FRS_STORAGE_ROOT=/workspace
export UV_CACHE_DIR=/workspace/uv-cache
export UV_PYTHON_INSTALL_DIR=/workspace/uv-python

bash scripts/setup_env.sh --pi05_frs_train
```

安装完成并通过 GPU 检查后执行：

```bash
source ./env_path
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=0
export FRS_ARROW_TMPDIR=/tmp/frs_arrow_tmp
export FRS_4090_CONFIG=/workspace/FRS_Tact/train_pi05_frs/configs/train_pi05_frs_shared_four_tasks_4090.yaml
```

这会使用 `/workspace/venvs/pi05_frs_train` 和 `/workspace/venvs/lerobot_data_tools`。
新终端中也需要重新执行上面的 `source` 和 `export`。

## 3. 登录 HF，下载四个 VLA 和 encoder

使用自己账号的 Read token，按提示输入：

```bash
"$DATA_TOOL_PYTHON" -c 'from huggingface_hub import login; login()'
```

四个 VLA 合计约 23.6 GiB。冻结 encoder 使用你指定的 `encoder_ckpt_0824`：

```bash
bash scripts/download_frs_shared_ckpts.sh --config "$FRS_4090_CONFIG" --max-workers 1 &&
"$TRAIN_PI05_FRS_PYTHON" deploy_smolvla/src/download_ckpt.py \
  --repo-id KaiyueChen/encoder_ckpt_0824 \
  --revision 1a432a8b9c1a38f75cb40c58a4924a401486d832 \
  --output-dir /workspace/FRS_Tact/checkpoints/encoder_ckpt_0824 \
  --minimal
```

## 4. 下载、转换四任务数据

仅在本地 v3 数据未准备好时执行：

```bash
bash scripts/download_frs_shared_data.sh --config "$FRS_4090_CONFIG" --max-workers 1
```

每任务选取 335 条 episode，保留 RGB、触觉、state、action。转换后的训练数据位于
`/workspace/frs_shared_four_tasks/datasets/v30/KaiyueChen/`，四任务为
insert、press、bread、pick。

遇到大小校验失败时，最新下载器会单独强制重下出错 episode。修复后重跑同一命令即可，
保留已有 `selection.json` 和其他完成的文件，无需重新采样或删除整个数据目录。

## 5. 检查输入，释放原始数据占用

```bash
bash train_pi05_frs/scripts/start_frs_pi05_shared_train.sh \
  --config "$FRS_4090_CONFIG" --check
```

确认数据下载和转换进程均已结束后，可在检查全部通过时清理约 26GiB 原始 Parquet。
这会保留 metadata 和采样清单；训练仍使用 v3 数据。

```bash
if bash train_pi05_frs/scripts/start_frs_pi05_shared_train.sh \
  --config "$FRS_4090_CONFIG" --check; then
  for TASK_DATASET in insert_01 press_01 bread_03 pick_01; do
    RAW_DATA="/workspace/frs_shared_four_tasks/datasets/raw/datasets/KaiyueChen/${TASK_DATASET}/data"
    if [[ -d "$RAW_DATA" ]]; then
      find "$RAW_DATA" -type f -name '*.parquet' -delete
    fi
  done
fi
```

清理后不要重复执行第 4 步，否则下载器会将原始 Parquet 补回来。不要清理
`datasets/v30`、`action_cache`、`tactile_embeddings`、`runs`、权重或虚拟环境。

## 6. 启动四任务缓存生成和共享 FRS 训练

```bash
FRS_TMUX_SESSION=frs_pi05_shared_4090 CUDA_VISIBLE_DEVICES=0 \
  bash train_pi05_frs/scripts/start_frs_pi05_shared_train.sh \
  --config "$FRS_4090_CONFIG"
```

看到会话启动成功后：

```bash
tmux attach -t frs_pi05_shared_4090
```

顺序为：各任务 checkpoint smoke → 冻结 encoder 触觉 embedding → 各自 Pi0.5 action cache
→ 一个共享 FRS。已有的兼容缓存会复用或续算；调整 cache batch 不改变 episode 划分。
每个任务的 Arrow 副本在独立子目录中生成，任务结束或正常报错退出后释放。

新 FRS 输出保存在 `/workspace/frs_shared_four_tasks/runs/shared_frs_4090/`，包含
`best/`、`last/`、验证指标和训练清单。首次启动不加 `--resume` 或 `--train-only`。
正式 FRS 训练产生 `last/` 后，中断恢复才使用同一配置追加 `--resume`。

若没有 tmux，启动器会在当前终端前台执行；先安装 tmux 再启动可避免 SSH 断开影响长任务。
GPU 显存不足时先停止当前进程，再降低对应 batch 后重新启动，勿同时启动第二份任务。
