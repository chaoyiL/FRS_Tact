# 双手纯视觉 SmolVLA 部署交接

更新日期：2026-09-03

## 1. 交接目标

本文档用于继续处理双手纯视觉 PyTorch SmolVLA 的真机部署问题，当前最重要的问题是：

1. 离线验证中模型会控制双手，也会预测左右夹爪闭合；
2. 某次真机运行中，右夹爪预测始终没有越过闭合阈值，因此右夹爪没有闭合；
3. 当前 action pose 是相对动作，小幅同方向位移会逐步累积，曾出现左手持续沿 Z 方向下压；
4. 需要定位离线数据与在线 observation、policy 输出、服务端转换和控制器执行之间从哪一层开始产生差异。

这是纯视觉 PyTorch SmolVLA。不要把它与 SmolVLA + FRS、单右手 SmolVLA 或 PI0.5 的部署逻辑混在一起。

### 真机 OBS 快速入口

真机部署 observation 的服务端保存根目录：

```text
/home/typhon/vb3_robot_server/eval_obs_data/
```

本文已分析的那次双手纯视觉 SmolVLA 部署 OBS：

```text
/home/typhon/vb3_robot_server/eval_obs_data/eval_obs_20260901_011323/
```

对应的动作 trace：

```text
/home/typhon/vb3_robot_server/action_debug_logs/20260901_011323_267747/
```

后续每次部署会生成新的 `eval_obs_<timestamp>` 目录。必须以服务端本次启动时打印的
`[ObsSaver] Observation saving enabled. Directory: ...` 为准，不要直接选择整个仓库中
时间最新的目录，因为 PI0.5、DECO 和 RDP 也可能写入相同根目录。

## 2. 当前代码状态警告

当前状态不是仅由 Git commit 决定：

- `FRS_Tact` 当前基线 commit：`8f310b9`；工作区还有其他 RDP/PI0.5 未提交修改。
- `vb3_robot_server` 当前基线 commit：`d585d7a`；大量服务端文件存在未提交或已暂存修改，其中包括 SmolVLA 的关键运行逻辑。

继续排查前不要执行 `git reset --hard`、`git checkout -- <file>` 或切换到会覆盖工作区的分支。当前 SmolVLA 行为依赖外部服务端工作区中的实际文件，而不只是 `d585d7a`。

重点保护这些服务端文件：

```text
/home/typhon/vb3_robot_server/client/robot_client.py
/home/typhon/vb3_robot_server/configs/smolvla_server_config.py
/home/typhon/vb3_robot_server/deploy_scripts/bimanual_smolvla_online.py
/home/typhon/vb3_robot_server/deploy_scripts/bimanual_smolvla_online_test.py
/home/typhon/vb3_robot_server/real_world/bimanual_umi_env.py
```

## 3. 系统边界与数据流

真机部署由两个独立进程组成：

```text
FRS_Tact PyTorch policy client
  读取 deploy_smolvla_pytorch.yaml
  加载 checkpoint 和官方 LeRobot pre/postprocessor
                  │ WebSocket robot-bridge-v1
                  ▼
vb3_robot_server hardware server
  采集双目 RGB + 20D state
  转换并执行 20D relative action chunk
                  │
                  ▼
双机械臂、夹爪和腕部相机
```

纯视觉模型使用 `FRS_Tact/.venv-smolvla-torch`，不再依赖旧 VB3 模型环境。但机器人、相机、控制器、动作转换、日志和夹爪 latch 仍运行在 `/home/typhon/vb3_robot_server`。这是 client/server 架构关系，不是模型环境依赖。

## 4. 正确启动方式

必须启动服务端和客户端两个终端。

终端 1，机器人服务端：

```bash
cd /home/typhon/vb3_robot_server
bash scripts/bimanual_smolvla.sh --mode vision
```

终端 2，纯视觉模型客户端：

```bash
cd /home/typhon/FRS_Tact
bash deploy_smolvla/scripts/start_smolvla.sh
```

客户端环境和配置快速检查：

```bash
cd /home/typhon/FRS_Tact
bash deploy_smolvla/scripts/start_smolvla.sh --check
```

重要区别：

- 服务端 `scripts/bimanual_smolvla.sh` 使用 `--mode vision`。
- 客户端 `deploy_smolvla/scripts/start_smolvla.sh` 已固定纯视觉 YAML，不接受 `--mode`。
- `deploy_smolvla/scripts/start_smolvla_frs.sh` 是 JAX + FRS，当前任务不要运行它。
- `deploy_smolvla/scripts/start_smolvla_right.sh` 是单右手模型，当前任务不要运行它。

正常启动顺序：

1. 服务端显示等待 policy client；
2. 客户端加载 checkpoint 并连接 `ws://127.0.0.1:26421`；
3. 服务端接收客户端发送的配置契约并初始化硬件；
4. 服务端发送初始 observation；
5. 客户端执行两次 warmup；
6. 客户端显示 `Press Enter to send START`；
7. 确认工作区安全后按 Enter，机器人此时才开始执行。

纯视觉 legacy chunk 协议不要求客户端等待通用 `action_ack`。每轮动作执行结束后，下一条 `obs` 就是继续推理的信号。不要重新加入 `receive_action_ack()`，否则会再次出现 `Expected action acknowledgement, received: obs`。

## 5. 当前模型与部署契约

唯一的双手纯视觉客户端配置：

```text
/home/typhon/FRS_Tact/deploy_smolvla/configs/deploy_smolvla_pytorch.yaml
```

当前关键值：

```yaml
backend: pytorch_smolvla
checkpoint: /home/typhon/FRS_Tact/checkpoints/model/smolvla_task1_0830_1.8w
allow_download: false
device: cuda

observation:
  state_action_profile: dual-arm-20x20
  data_type: vision
  task: 0
  single_arm_mode: false
  no_state_obs_mode: false

control:
  control_frequency: 20.0
  controller_frequency: 80.0
  action_horizon: 20
  steps_per_inference: 10
```

每轮 policy 输出 `(20, 20)`，服务端最多执行前 10 个有效动作，然后重新采集 observation 并推理。

### 5.1 20D state

```text
0:7    左手：相对本次 episode reference 的 xyz + axis-angle + gripper width
7:14   右手：相对本次 episode reference 的 xyz + axis-angle + gripper width
14:20  左手相对右手的 xyz + axis-angle
```

这里的 reference pose 由服务端在硬件初始化后、发布 warmup observation
之前的第一次 `env.get_obs()` 记录。它早于客户端按 Enter/START，START 后不会
重新采集零点；机器人在等待期间完全静止时，它才近似等于按 Enter 时的位姿。

### 5.2 20D action

```text
0:10   左手：relative xyz(3) + rotation-6D(6) + gripper width(1)
10:20  右手：relative xyz(3) + rotation-6D(6) + gripper width(1)
```

服务端把 relative pose 从当前实测姿态开始逐步复合。若模型连续预测同方向的小 Z 位移，它们会累积为持续下压或抬升，而不是始终相对同一个固定起点。

### 5.3 相机和键名

当前服务端 `SmolVLAServerConfig.camera.devices` 的顺序是：

```text
camera0 / left_hand  -> /dev/video6
camera1 / right_hand -> /dev/video8
```

Linux 的 `/dev/video*` 编号可能因重新插拔改变，每次硬件变化后都应重新确认左右相机，不能只相信编号。

服务端发送：

```text
observation.images.camera0
observation.images.camera1
```

checkpoint 声明的输入键是：

```text
observation.images.camera1
observation.images.camera2
```

YAML 当前映射：

```yaml
rename_map:
  observation.images.camera0: observation.images.camera1
  observation.images.camera1: observation.images.camera2
```

客户端保留机器人原始键，官方 checkpoint preprocessor 再应用 rename。不要在 `_prepare_frame()` 中提前把两个键原地覆盖，否则容易造成相机键碰撞或左右相机错位。

图像契约为两张 HWC RGB、`uint8`、服务端 Scheme A 处理后的 256×256 图像。颜色顺序、左右相机顺序、裁剪区域和曝光差异都是当前在线问题的高优先级嫌疑项。

## 6. YAML 与服务端配置的所有权

“client/server 共用一个 YAML”只对启动时协商的 policy contract 成立，不代表所有服务端配置都来自 YAML。

| 配置类别 | 当前来源 |
| --- | --- |
| checkpoint、device、是否允许下载 | `deploy_smolvla_pytorch.yaml` |
| 客户端要连接的 bridge 地址、端口、token | `deploy_smolvla_pytorch.yaml` 和客户端环境变量 |
| 服务端监听 host/port | `SmolVLAServerConfig` 或服务端 CLI `--ip/--port` |
| 服务端允许的 token | 服务端 `--token-file` / `token_list.txt` |
| prompt、task、视觉模式 | `deploy_smolvla_pytorch.yaml` |
| control/controller frequency、horizon、steps | `deploy_smolvla_pytorch.yaml`，启动时发给服务端 |
| 左右夹爪 close/reopen/closed command | `deploy_smolvla_pytorch.yaml`，启动时发给服务端 |
| 相机设备、曝光、白平衡、裁剪方式 | `vb3_robot_server/configs/smolvla_server_config.py` |
| 标定矩阵、动作转换、速度和单步安全限制 | `vb3_robot_server` |
| permanent latch 开关 | `vb3_robot_server/configs/smolvla_server_config.py` |
| observation/action trace 实际保存目录 | `vb3_robot_server/configs/smolvla_server_config.py` |

YAML 中的 `logging:` section 当前没有被 `pytorch_remote_client.py` 读取。因此：

```yaml
logging:
  output_dir: outputs/smolvla_pytorch_observations
```

不是当前纯视觉在线 OBS 的实际保存位置。实际保存发生在服务端，见第 9 节。
修改客户端 YAML 的 address/port/token 也不会同步修改服务端监听配置；连接两端
必须显式保持一致。

## 7. 当前夹爪 trick 的真实行为

客户端 YAML 当前发送：

```yaml
gripper:
  hysteresis_enabled: true
  left_close_threshold: 0.09
  left_reopen_threshold: 0.10
  left_closed_command: 0.01
  right_close_threshold: 0.09
  right_reopen_threshold: 0.10
  right_closed_command: 0.01
```

真正执行位置：

```text
/home/typhon/vb3_robot_server/deploy_scripts/bimanual_smolvla_online.py
/home/typhon/vb3_robot_server/real_world/bimanual_umi_env.py
```

未闭合时，模型输出的 width 先转换为物理夹爪命令：

```text
physical_command = clip((model_width - 0.050) / 1.77, 0.01, 0.04)
```

任意一侧模型 width 第一次 `<= 0.09` 时，该侧 latch 闭合，物理命令强制为 `0.01`。

当前服务端同时满足：

```text
task = 0
gripper_permanent_latch_enabled = True
task1_sequence_enabled = False
```

所以当前双手纯视觉运行的实际行为是：

- 左右夹爪分别在第一次越过 `0.09` 后永久保持 `0.01`；
- 后续模型即使预测 `>= 0.10` 也不会重新打开；
- latch 状态直到本次环境退出并重建才复位；
- YAML 的 reopen threshold 仍参与配置校验，但被 permanent latch 优先逻辑压住；
- 所有 `task1_*` 抬升、归位、右手预闭合和轨迹 choreography 当前均不生效。

因此当前运动完全由模型直接控制，服务端只额外执行夹爪永久锁存及通用安全限制。

## 8. 已经解决的问题

### 8.1 `lerobot.configs.policies` 导入失败

根因是 `FRS_Tact` 内部精简 JAX `lerobot` 包遮蔽了官方 PyTorch LeRobot。现在纯视觉客户端使用独立环境：

```text
/home/typhon/FRS_Tact/.venv-smolvla-torch/bin/python
```

环境缺失时执行：

```bash
cd /home/typhon/FRS_Tact
bash scripts/setup_env.sh --smolvla
```

也可以通过 `SMOLVLA_TORCH_PYTHON` 显式指定 Python。

### 8.2 启动模式混淆

客户端已拆成固定脚本，不再接受 `--mode`。纯视觉只运行 `start_smolvla.sh`，FRS 只运行 `start_smolvla_frs.sh`。

### 8.3 `Expected action acknowledgement, received: obs`

纯视觉客户端已经删除 action ACK 等待。不要恢复该逻辑。

### 8.4 client/server 配置不一致

prompt、task、控制频率、horizon、steps 和夹爪阈值现在由客户端以原子 startup payload 发给纯视觉服务端。硬件和安全配置仍属于服务端。

### 8.5 Task 1 choreography 干扰

当前 YAML 使用 `task: 0`，服务端 `task1_sequence_enabled: false`。旧的左手抬升、回位、右手预闭合等 task1 trick 已休眠，不能再用它们解释当前运动。

## 9. 当前未解决问题与现有证据

### 9.1 右夹爪在线不闭合

目前已分析的对应纯视觉双手运行：

```text
动作日志：/home/typhon/vb3_robot_server/action_debug_logs/20260901_011323_267747/
在线 OBS：/home/typhon/vb3_robot_server/eval_obs_data/eval_obs_20260901_011323/
```

该次运行共 13 个 `(20,20)` action chunk：

- 左夹爪 raw policy 输出最小值 `0.07683`；113/260 个预测点低于 `0.09`，成功触发闭合；
- 左夹爪实际执行的 130 点中，61 点被强制为 `0.01`；
- 右夹爪 raw policy 输出范围 `0.10388–0.10805`；0/260 个点低于 `0.09`；
- 右夹爪实际命令范围 `0.03044–0.03280`，从未进入 `0.01`。

直接结论：该次右夹爪不闭合不是服务端看到了闭合预测却漏执行，而是在线 policy 的右夹爪输出根本没有跨过 `0.09`。优先检查在线输入和闭环轨迹为什么没有进入训练数据中的右手闭合阶段。

### 9.2 左手持续 Z 方向累积

当前 `task: 0` 没有 task1 起始 Z 下限和 choreography 保护。action 是相对位姿并逐步复合，所以持续同号 Z 增量会积累。服务端只有速度、单步位置和旋转 delta 等通用限制，它们限制每一步的幅度，但不会阻止许多安全小步累计成大位移。

在没有确认 policy raw Z、转换后绝对目标和控制器反馈三者之前，不要直接用硬编码反向偏置抵消。

### 9.3 离线好、在线差

离线测试使用数据集里的真实 observation；在线闭环中，早期的小误差会改变后续相机画面和机器人状态，模型可能进入训练分布之外。当前最可能的分支包括：

1. 左右相机或 rename 对应错误；
2. RGB/BGR、裁剪、曝光或白平衡与数据集不同；
3. START 时机器人初始位姿与数据集差异过大；
4. 左手累计误差先改变场景，导致右手阶段永远不出现；
5. state 语义或 gripper width 标定与训练数据不同；
6. policy 确实输出闭合，但在动作转换/控制器层丢失。现有 20260901 trace 对第 6 点不支持，但新运行必须重新逐层确认。

## 10. 日志位置和含义

每次运行时以服务端打印的目录为准，不要简单把“时间最新的目录”当作 SmolVLA；同一服务端仓库也在运行 PI0.5、DECO 和 RDP。

动作诊断目录：

```text
/home/typhon/vb3_robot_server/action_debug_logs/<timestamp>/
├── action_debug.jsonl
├── chunk_trace.jsonl
├── controller_trace.jsonl
├── full_prediction_chunk.png
├── full_prediction_feedback.png
└── executed_vs_actual.png
```

- `action_debug.jsonl`：raw/new action 的派生幅度指标、`is_new`、时间戳和 controller records；它不保存完整 raw action 数组；
- `chunk_trace.jsonl`：完整 `vla_action`、`selected_raw_actions`、`absolute_waypoints` 以及每个推理 chunk 的执行记录；
- `controller_trace.jsonl`：实际控制器反馈；
- PNG：policy、执行目标和反馈轨迹的快速对照。

在线 observation 默认由服务端保存到：

```text
/home/typhon/vb3_robot_server/eval_obs_data/eval_obs_<timestamp>/
```

每个 step 目录包含两张腕部相机图、机器人 pose、gripper width 和 timestamp。服务端 `--save_obs` 当前默认是 `True`。

## 11. 离线验证结果

评估脚本：

```text
/home/typhon/FRS_Tact/tools/eval_smolvla_pytorch_offline.py
```

结果目录：

```text
/home/typhon/FRS_Tact/outputs/smolvla_pytorch_offline/two_tubes_03_ep202_211/
```

数据为 `KaiyueChen/two_tubes_03` episodes 202–211，共 7,804 帧、153,980 个有效预测步。它们按当时仓库的 `eval_split: 0.10` 推定为 held-out；checkpoint 自身没有完整保存多数据集训练清单，所以不能只靠 checkpoint 元数据证明这一点。

主要指标：

| 指标 | 左手 | 右手 |
| --- | ---: | ---: |
| full-horizon translation MAE | 0.360 mm | 0.320 mm |
| rotation geodesic error | 0.177° | 0.166° |
| gripper MAE | 1.184 mm | 0.906 mm |
| lead-0 close events | 10/10 | 10/10 |
| lead-0 close timing MAE | 1.2 帧 / 0.040 s | 1.3 帧 / 0.043 s |
| first-10 close-event F1 | 0.775 | 0.756 |
| first-10 close-event recall | 0.956 | 0.811 |

离线结论：policy 在数据集 observation 上学会了双手动作，并能预测左右夹爪未来闭合。它不等于真机闭环成功，也不能否定当前在线 observation/轨迹存在分布偏移。

主要产物：

```text
metrics.json
episode_metrics.csv
per_horizon.csv
predictions.npz
action_error_heatmap.png
group_error_by_horizon.png
action_timeline_episode_*.png
gripper_timeline_episode_*.png
```

如需基于现有缓存重新运行：

```bash
cd /home/typhon/FRS_Tact
.venv-smolvla-torch/bin/python tools/eval_smolvla_pytorch_offline.py \
  --config deploy_smolvla/configs/deploy_smolvla_pytorch.yaml \
  --dataset-root .cache/eval_datasets/KaiyueChen/two_tubes_03 \
  --episodes 202-211 \
  --output-dir outputs/smolvla_pytorch_offline/two_tubes_03_ep202_211 \
  --device cuda \
  --seed 0
```

## 12. 下一步排查顺序

不要先改 threshold 或加新的轨迹 trick。按以下边界逐层定位：

### 第一步：固定并记录一次安全真机运行

1. 记录服务端和客户端完整 stdout；
2. 记录两边当前 Git commit 和 `git status --short`；
3. 记录服务端打印的 OBS 与 action trace 精确目录；
4. 记录按 Enter 时两臂初始 pose、场景摆放和任务完成到哪一步；
5. 出现异常时立即停止，不要为了采更多数据让机械臂持续下压。

### 第二步：检查输入是否与训练数据一致

对同一阶段并排查看：

- 在线 `camera0_rgb.jpg` 与数据集左手图；
- 在线 `camera1_rgb.jpg` 与数据集右手图；
- 左右视角是否交换；
- RGB、裁剪区域、分辨率、曝光、白平衡；
- 20D state 的范围和初始零点。

如果输入不一致，先修输入，不要用动作后处理掩盖。

### 第三步：检查 raw policy 是否进入右手闭合阶段

右夹爪是 action dim `19`，左夹爪是 dim `9`。完整 raw action 要从
`chunk_trace.jsonl` 的 `vla_action` 读取，而不是从 `action_debug.jsonl`
读取。对每个 chunk 检查：

```text
min(raw_action[:, 9])
min(raw_action[:, 19])
raw_action[:10, 19]
```

判定：

- `raw_action[:,19]` 始终 `> 0.09`：问题位于 policy 输入、闭环状态或模型输出；
- raw 已 `<= 0.09`，但转换后命令不是 `0.01`：检查 hysteresis/latch；
- 转换后已经是 `0.01`，但实际未闭合：检查 controller、夹爪标定和硬件。

### 第四步：定位左手 Z 累积发生在哪一层

依次对比：

1. raw relative action 中左手 Z（dim `2`）；
2. 转换后的左手绝对目标 Z；
3. controller target Z；
4. controller actual/feedback Z。

只有 raw action 已持续同号时，才说明模型直接要求累积运动；如果 raw 没有但绝对目标累积，检查 relative pose 转换；如果目标正常而 feedback 异常，检查控制器或标定。

### 第五步：做在线 OBS 离线回放

最有价值的下一项工具是：读取某次 `eval_obs_<timestamp>`，使用当前 checkpoint 对保存的每一帧重新推理，并把 raw action 与同次 `chunk_trace.jsonl` 对齐。这样可以确认在线实时运行与离线回放是否得到同一输出，并能直接比较在线 OBS 与 `two_tubes_03` 数据集的视觉/状态分布。

## 13. 暂时不要做的事情

- 不要给客户端命令添加 `--mode vision`；
- 不要把 FRS 配置或单右手配置合入当前 YAML；
- 不要恢复纯视觉客户端的 action ACK 等待；
- 不要假设所有 server 配置都来自 YAML；
- 不要启用旧 `task1_sequence_enabled` 来试图修复 task0 的模型问题；
- 不要仅通过降低右夹爪 close threshold 解决“不闭合”，降低数值会让闭合更难触发；
- 不要先添加硬编码左手 Z 反向偏置；
- 不要覆盖或清理 `vb3_robot_server` 的未提交修改和历史日志。

## 14. 关键文件索引

FRS_Tact 客户端：

```text
deploy_smolvla/configs/deploy_smolvla_pytorch.yaml
deploy_smolvla/scripts/start_smolvla.sh
deploy_smolvla/scripts/start_remote_client.sh
deploy_smolvla/remote_client.py
deploy_smolvla/pytorch_remote_client.py
deploy_smolvla/bridge_client.py
deploy_smolvla/README.md
tools/eval_smolvla_pytorch_offline.py
tests/test_eval_smolvla_pytorch_offline.py
```

机器人服务端：

```text
/home/typhon/vb3_robot_server/scripts/bimanual_smolvla.sh
/home/typhon/vb3_robot_server/configs/smolvla_server_config.py
/home/typhon/vb3_robot_server/deploy_scripts/bimanual_smolvla_online.py
/home/typhon/vb3_robot_server/deploy_scripts/observation_profiles.py
/home/typhon/vb3_robot_server/real_world/real_inference_util.py
/home/typhon/vb3_robot_server/real_world/bimanual_umi_env.py
/home/typhon/vb3_robot_server/client/robot_client.py
```

## 15. 给新会话的起始提示

新会话可以直接使用下面这段话：

```text
请先完整阅读 /home/typhon/FRS_Tact/deploy_smolvla/HANDOFF.md，再开始处理。

当前任务只处理双手纯视觉 PyTorch SmolVLA，不处理 FRS、PI0.5、DECO、RDP 或单右手模型。请先以现有日志为证据定位问题，不要猜测，不要覆盖两个仓库里的未提交修改。

优先目标：分析最新一次明确属于 SmolVLA 的在线 OBS 和 action trace，确认右夹爪 action dim 19 为什么没有跨过 0.09，并逐层对比左手 Z 的 raw relative action、绝对目标和 controller feedback。离线 two_tubes_03 验证已经证明 policy 在数据集 observation 上能够预测双手动作和左右夹爪闭合，因此重点排查在线 observation/domain gap 和闭环轨迹偏移。
```
