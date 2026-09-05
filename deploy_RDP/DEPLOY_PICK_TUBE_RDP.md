# Pick-tube RDP 真机部署

本部署复用 `vb3_robot_server-main` 已有的相机、机器人控制和 WebSocket
bridge。RDP 只运行在策略机上；机器人机继续运行原服务器进程。

当前桥接协议为 **`rdp_step_v3`**，策略端与机器人端必须一起更新。0902 单右臂配置由
`scripts/start_pick_tube_rdp_right.sh` 选择；通用 client launcher 默认仍是双臂配置。
本文后面的公开旧权重下载与性能数据是历史参考，不代表0902的实际延迟或发布资格。

## 运行 contract

- 观测：两路 224×224 RGB、四路 224×224 触觉 RGB、20 维 state。
- 模型时序：按采集时间在30Hz时间网格上选择4帧历史，按ratio=2选取2帧慢策略观测；
  episode 起始处复制第一帧完成左侧填充，与训练 sampler 一致。
- 触觉顺序：`left_0, right_0, left_1, right_1`，每路经同一个冻结
  ResNet18 得到 512 维；每只手臂的两路 embedding 拼成 1024 维后分别做
  PCA-15，最终拼成 30 维触觉特征。
- 慢策略：每16个30Hz时间步更新latent plan，默认8个diffusion steps；缺帧时按采集时间
  跳过已过时的解码索引。历史取最近实采帧，不能恢复漏采触觉；长间隔会重新规划。
- 快策略：每个观测周期都用累计触觉历史解码当前一步。
- 动作：每次发送`[1,20]`。单右臂策略输出10D，另一臂填保持动作。服务器首次用最新
  实测Quest位姿初始化基底，之后将逐帧局部增量合成到上次成功入队的目标；拒绝动作
  不提交基底。目标与最新实测反馈的位姿偏差仍受已有位置/旋转限值约束。
- RDP不使用SmolVLA的RTC chunk消费；服务器在读取参考反馈、完成动作转换后，以
  `now + 0.05s`调度。反馈新鲜度使用`robot_receive_timestamp`，不能使用已减去校准偏移的
  `robot_timestamp`。默认反馈接收年龄上限0.1s、预测来源观测年龄上限0.5s。
- 观测带公共采集时间与两路实际帧时间。重复、倒退或任一路未更新的观测不推进策略。
- ACK返回`scheduled/rejected`、`scheduled_count`、目标时间、基底来源和反馈时间。
  `scheduled`仅确认目标入队，**不代表机器人已经到达**。拒绝后服务器立即停止控制器，
  客户端停止并清空当前计划；不会把未执行动作当作成功继续。

> PCA-30 会改变策略输入维度。原先使用 2048 维触觉 embedding 训练的
> AT/LDP checkpoint 不能直接部署，需要重新生成数据并训练新 checkpoint。


## 1. 策略机环境

在 RDP 仓库中创建环境。部署 requirements 已包含训练 requirements：

```bash
cd /path/to/FRS_Tact/deploy_RDP
bash scripts/setup_rdp_env.sh
source .venv/bin/activate
```

已有训练环境只需补装部署依赖：

```bash
uv pip install --python .venv/bin/python \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  -r requirements-rdp-deploy.txt
```

部署不需要 JAX。`tactile_encoder_torch.py` 会直接读取原 Flax
`checkpoint.json` 和 `.npz`，使用经过数值对齐的 PyTorch ResNet18 推理。

从公开仓库 `wjstx/rdp` 下载部署权重：

```bash
HF_ENDPOINT=https://huggingface.co \
  bash scripts/setup_pick_tube_data.sh weights
```

该命令只下载各训练目录下的 `latest.ckpt`、Hydra 配置和 normalizer，不下载
重复的 top-k checkpoint。默认保存到 `data/weights/wjstx_rdp`。

仓库包含配套的 `at_20260813_115524` 和 `ldp_20260813_214114`；下载脚本会
同时取得两者。已验证的 SHA-256 前缀分别为 `caa220f9` 和 `c19d6786`。

正式权重已在 RTX 5080 上用真实 LeRobot 帧完成离线验证。按客户端默认的两次
warm-up 后，8-step LDP 慢路径平均 `12.25 ms`、AT 快路径平均 `2.76 ms`，
10 帧最大 `12.45 ms`，CUDA peak allocated memory 约 `249.8 MiB`；全部输出
均为有限 `[1,20]`，并通过机器人服务器现有动作安全校验。该基准不包含网络和
机器人控制器延迟。

## 2. 配置策略机

编辑 `configs/deploy_pick_tube_rdp.yaml`：

```yaml
model:
  ldp_checkpoint: data/weights/wjstx_rdp/ldp_20260813_214114/checkpoints/latest.ckpt
  at_checkpoint: data/weights/wjstx_rdp/at_20260813_115524/checkpoints/latest.ckpt
  tactile_encoder_dir: /absolute/path/to/encoder_ckpt_0809
  device: cuda:0
  tactile_pca_path: data/PCA_Transform_PickTube/tactile_pca_2x15.npz
  num_inference_steps: 8

connection:
  address: ROBOT_SERVER_IP
  port: 26421

control:
  slow_update_interval: 16
```

LDP checkpoint 内保存的 AT 路径不会被直接使用；启动时会被这里的
`at_checkpoint` 覆盖。策略进程会选择 checkpoint 配置指定的 EMA/model，
并把 LDP normalizer 显式交给 AT。

策略机与机器人机必须使用同一个 token：

```bash
export VB_ROBOT_TOKEN='replace-with-the-shared-token'
```

## 3. 启动机器人服务器

在机器人电脑上：

```bash
cd /path/to/vb3_robot_server-main
export VB_ROBOT_TOKEN='replace-with-the-shared-token'
bash scripts/bimanual_rdp.sh
```

也可以继续使用服务器现有的 `token_list.txt` 或 `VB3_TOKEN_FILE`。
RDP 客户端连接后会协商 `policy_type=rdp`、`data_type=vitac`、224 分辨率和
`rdp_step_v3`单步动作；旧v2 RDP端点必须同步更新。其他策略的旧ACK格式仍保留。

第一次连接建议先进行无硬件协议测试：

```bash
bash scripts/bimanual_rdp.sh --dry-run --dry-run-iterations 6
```

v3 dry-run没有控制器，因此会明确返回`rejected/dry_run_no_hardware`并结束交互；
它验证协议，不伪装成动作已入队。纯离线回归不需要启动这个服务器命令。

## 4. 启动策略客户端

在策略机上另一个终端运行：

```bash
cd /path/to/FRS_Tact/deploy_RDP
source .venv/bin/activate
export VB_ROBOT_TOKEN='replace-with-the-shared-token'
bash scripts/start_pick_tube_rdp_client.sh
```

使用另一份配置时：

```bash
RDP_DEPLOY_CONFIG=/absolute/path/to/deploy.yaml \
  bash scripts/start_pick_tube_rdp_client.sh
```

默认会先接收一帧观测并 warm up 两次。看到 `Ready` 后确认机器人工作区安全，
再按 Enter 发送 `start`。`Ctrl-C` 会尽力发送 `stop` 并关闭连接。

## 5. 上真机前检查

```bash
# 策略端
.venv/bin/python -m pytest -q tests/test_pick_tube_rdp_deploy.py

# 机器人服务器端；PYTHON 可换成其可用的测试环境
PYTHON=/path/to/python
"$PYTHON" -m pytest -q \
  tests/test_rdp_runtime_contract.py \
  tests/test_relative_action_stale_prefix.py \
  deploy_scripts/bimanual_smolvla_online_test.py
```

正式 checkpoint 的输出仍必须通过服务器现有的位置增量、旋转增量和夹爪范围
检查。不要用 smoke checkpoint 驱动真机。
