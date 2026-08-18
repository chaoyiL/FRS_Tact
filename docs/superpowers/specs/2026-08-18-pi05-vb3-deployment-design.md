# pi0.5 / pi0.5+FRS 统一部署设计

## 目标

把 `/home/typhon/ManiSkill-vitac/scripts/infer.sh` 当前工作树中的纯视觉
`pi05_bi` 部署能力迁入本仓库，并让普通 pi0.5 与 pi0.5+FRS 都连接
`vb3_robot_server`。两种部署必须使用同一份 pi0.5 checkpoint、norm stats、模型维度、
相机映射和任务配置，唯一的策略差异是 FRS 模式额外使用触觉 steering。

迁移目标固定为双臂 20D state、20D robot action、32D pi0.5 model action、50-step
action horizon、两路 RGB 输入。旧仓库 Git HEAD 中 15-step、触觉直接输入 pi0.5 的
`pi05_bi_vitac` 路线不在本次范围内。

## 用户入口

保留现有 FRS 命令，并新增对称的普通 pi0.5 命令：

```bash
bash deploy_pi05_frs/scripts/start_pi05.sh
bash deploy_pi05_frs/scripts/start_pi05_frs.sh
```

两个 wrapper 都调用 `deploy_pi05_frs/scripts/start_remote_client.sh`。wrapper 只选择部署
模式和默认配置；通用启动器统一负责参数解析、Python 选择、token、环境变量和 Python
module 启动。

`start_remote_client.sh` 接受 `--mode pi05|frs`、`--config PATH`、`--check` 和
`--max-iterations N`。两个 wrapper 显式传入 mode；为兼容当前直接调用方式，省略
`--mode` 时仍默认 `frs`。`--max-iterations` 原样传给所选 Python module；普通模式按 action
chunk 计数，FRS 模式按 FRS chunk 计数。

配置路径默认使用：

```text
deploy_pi05_frs/configs/deploy_pi05.yaml
```

两种模式都优先读取 `PI05_DEPLOY_CONFIG`。FRS wrapper 继续接受旧的
`PI05_FRS_DEPLOY_CONFIG` 作为次级兼容覆盖。普通 pi0.5 不读取旧的 FRS 专用变量。

## 共享配置

单个 YAML 是两种部署的配置源。公共字段包括：

- `checkpoint`、`seed`、`num_steps`；
- `model`：state/action 维度、horizon、两种 Gemma variant、camera map 和 empty cameras；
- `norm_stats`：目录、asset id 和 quantile normalization；
- `connection`：地址、端口、重试、ping、观测/ACK timeout 和 token；
- `observation`：prompt、双/单臂和 state 模式；
- `control`：control/controller frequency、horizon 和每轮执行步数；
- `runtime`：auto start、warmup 次数、状态输出间隔和最大迭代次数；
- `logging`：是否保存、采样间隔和队列大小；
- `frs`：仅 FRS 模式读取的 checkpoint、触觉键和 decoder 参数。

模式差异只放在 `profiles`：

```yaml
profiles:
  pi05:
    data_type: vision
    observation_output_dir: outputs/pi05_observations
  frs:
    data_type: vitac
    observation_output_dir: outputs/pi05_frs_observations
```

配置加载器接收明确的 mode，并把公共配置和对应 profile 解析为有效配置。普通模式不要求
`frs` 字段可用；FRS 模式继续执行全部 FRS 契约校验。两种模式都从同一公共字段构造
`Pi05DeploymentConfig`，因此 checkpoint 和 norm stats 不会产生两份默认值。

## 组件边界

### Shell 层

- `start_pi05.sh`：普通 pi0.5 便捷入口，传 `--mode pi05`。
- `start_pi05_frs.sh`：FRS 便捷入口，传 `--mode frs`。
- `start_remote_client.sh`：通用启动器，根据 mode 选择 Python module；`--check` 输出 mode、
  config、token source、Python 和 entrypoint，不加载模型或连接机器人；
  `--max-iterations` 传给对应 module。

Python 选择顺序维持现有约定：普通模式使用 `PI05_PYTHON`，FRS 模式使用
`PI05_FRS_PYTHON`；随后依次回退到 `VB3_PYTHON`、仓库 `.venv/bin/python`、`python3`。
token 优先取 `VB_ROBOT_TOKEN`，否则读取 `VB3_TOKEN_FILE`，默认路径仍为
`/home/typhon/vb3_robot_server/token_list.txt`。token 不写入配置示例或日志。

### Python 公共层

从现有 `remote_client.py` 抽出以下共享职责：

- YAML 读取、公共字段校验和 profile 解析；
- `Pi05DeploymentConfig` 构造；
- token 和相对路径解析；
- RGB/state observation 校验；
- 后台 observation saver。

`Pi05RemotePolicy` 继续作为唯一 pi0.5 adapter，负责 checkpoint/norm stats 加载、输入
transform、JAX sampling 和 action unnormalization。`RobotBridgeClient` 继续作为唯一
WebSocket/msgpack/NumPy 协议实现。

### 普通 pi0.5 client

新增普通模式 module，职责只有 legacy chunk 控制循环：

1. 加载共享配置和 `Pi05RemotePolicy`。
2. 连接 bridge，校验 `hello.protocol == robot-bridge-v1`。
3. 发送不包含 `execution_protocol` 的 config，让 `vb3_robot_server` 协商
   `legacy_chunk`。
4. 接收一帧真实 observation，按配置次数进行 JIT warmup。
5. 人工确认或 `auto_start` 后发送 `state=start`。
6. 每轮接收带 `obs_seq` 的 observation，预测 `[1, 50, 32]` normalized action。
7. 裁出 20 个机器人维度并反归一化，得到连续、有限的 float32 `[50, 20]` action chunk。
8. 发送 `action` 和原 observation 的 `obs_seq`。
9. 等待同一 `obs_seq` 的 `action_ack`，成功后才能接收下一轮 observation。
10. 达到 `max_iterations`、收到 Ctrl-C 或发生异常时，在关闭连接前尽力发送
    `state=stop`。

`steps_per_inference` 允许位于 `[1, action_horizon]`，但传输动作始终包含完整
`action_horizon`；默认二者均为 50，以复现选定的旧 `pi05_bi` 部署。

### FRS client

现有 FRS 控制循环保持不变：发送 `execution_protocol=frs_steering_v1`，接收
`frs_chunk_start`，生成源动作和 `x_base`，再按 `frs_steer_request` 返回单个 20D 动作。
本次只把它接到共享配置和公共辅助模块，不改变 FRS 数学、触觉历史或 wire schema。

## `vb3_robot_server` 兼容性

不修改 `/home/typhon/vb3_robot_server`。它已经支持同一 `robot-bridge-v1` 连接上的两种模式：

- config 不带 `execution_protocol`：`legacy_chunk`，接收完整 action chunk，执行成功后发送
  `action_ack`；
- config 带 `execution_protocol=frs_steering_v1`：FRS 逐动作 steering。

两种客户端都连接同一个默认地址 `127.0.0.1:26421`，使用 Bearer token 和二进制 msgpack
帧。普通 client 必须等待 `action_ack`；旧 `infer.py` 的发送后立即收下一帧行为不迁移。

## 错误处理与安全

- 缺少配置字段、非法 profile、维度/horizon 不一致时，在加载 checkpoint 前失败。
- observation 缺少 RGB/state、shape 错误或包含非有限状态时不运行推理。
- action shape 不是 `[action_horizon, robot_action_dim]`，或包含 NaN/Inf 时不发送。
- 非匹配 `obs_seq` 的 ACK、意外 message type、鉴权失败和连接关闭均终止控制循环。
- 所有退出路径尽力发送 STOP；STOP 失败只记录警告，随后仍关闭 socket。
- observation 保存是有界、后台、best-effort 的，写盘失败不阻塞机器人控制。
- 本次不增加自动重连。控制过程中断线后必须重新启动客户端、重新 warmup 并人工确认。

## 测试与验收

无需 GPU 或机器人即可运行的自动测试包括：

- 公共 YAML 在 `pi05`/`frs` profile 下的解析、必填字段和 horizon 约束；
- 两个 shell wrapper 的 `bash -n` 和 `--check` 输出；
- 普通模式 server config 不含 `execution_protocol`，且 `data_type=vision`；
- fake policy/bridge 控制循环验证 warmup、START、`obs_seq`、action shape、ACK 顺序、
  `max_iterations` 和 STOP；
- bridge 对错误 ACK/message 的拒绝；
- 现有 FRS protocol 和相关单元测试回归。

部署机验收分两步：

1. GPU dry run：加载真实 checkpoint/norm stats，用一帧合成或录制 observation 完成 warmup，
   确认 JAX backend、`[50,20]` 输出和有限值。
2. `vb3_robot_server` dry-run/真机：先运行 `start_pi05.sh --max-iterations 2`，确认两轮 ACK、
   STOP 和 server action trace，再在急停可用且有人观察的条件下进入真机。

FRS 使用同一 checkpoint/norm stats 完成对应 smoke test，确认两个启动命令打印相同的模型
资产路径。

## 不在本次范围

- 不迁移旧 Git HEAD 的 15-step `pi05_bi_vitac` 或 AnyTouch 直喂 pi0.5 模型。
- 不修改 `vb3_robot_server`、机械臂控制器、相机设备或标定文件。
- 不改变 FRS 网络、decoder、保护策略或训练 checkpoint 格式。
- 不实现断线自动恢复、跨 episode reset 协议或新的 wire protocol。
- 不复制 `/home/typhon/ManiSkill-vitac` 的 openpi 源码；本仓库 vendored JAX pi0.5 是唯一
  模型实现。

## 完成标准

- `start_pi05.sh` 和 `start_pi05_frs.sh` 从同一 YAML 读取同一 pi0.5 模型资产。
- 普通模式通过 `vb3_robot_server` legacy chunk 协议发送 `[50,20]` 并等待 ACK。
- FRS 现有命令和 `frs_steering_v1` 行为保持可用。
- 自动测试、shell 静态检查和配置 `--check` 全部通过。
- README 清楚说明 server 启动方式、两个 client 命令、配置字段和 GPU/真机验收顺序。
