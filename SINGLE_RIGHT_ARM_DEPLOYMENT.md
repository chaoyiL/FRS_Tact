# Pi0.5 / SmolVLA 右手单臂部署

部署端支持两个显式状态/动作合同：

| 模式 | profile | state | action | `single_arm_mode` | `controlled_arm` |
| --- | --- | ---: | ---: | --- | --- |
| 双臂 | `dual-arm-20x20` | 20D | 20D | `false` | `null` |
| 右手单臂 | `single-right-arm-7x10` | 7D | 10D | `true` | `right` |

不能给双臂 checkpoint 只打开 `single_arm_mode`。模型 checkpoint、norm stats、FRS
checkpoint、机器人服务端的 state/action 合同必须属于同一个 profile；客户端会在启动时
拒绝混用。

## Pi0.5 纯视觉

复制 `deploy_pi05/configs/deploy_pi05.yaml` 为自己的右手配置，然后至少修改：

```yaml
checkpoint: /path/to/single_right_pi05/checkpoint
task: 0

model:
  state_action_profile: single-right-arm-7x10
  state_dim: 7
  robot_action_dim: 10
  # 必须等于训练该 checkpoint 时的模型 action_dim（10、20 或 32）。
  action_dim: 10

norm_stats:
  dir: /path/to/single_right_pi05/checkpoint/assets
  asset_id: your_single_right_asset_id

observation:
  language_prompt: Use the right hand to complete the task.
  single_arm_mode: true
  controlled_arm: right
  no_state_obs_mode: false
```

Pi0.5 客户端沿用既有协议，不会替机器人服务端切换硬件模式。启动客户端前，必须把
`vb3_robot_server` 启动为右手单臂模式，使其发送 7D state 并接收 10D action。

## Pi0.5 + FRS

在纯视觉右手合同基础上，使用配套的单手 FRS checkpoint，并仅请求右手两路触觉：

```yaml
task: 0

frs:
  checkpoint: /path/to/single_right_pi05_frs
  tactile_keys:
    - observation.images.tactile_right_0
    - observation.images.tactile_right_1
```

单手模式不启用现有双臂 Task 1 抓取保护，因此必须使用 `task: 0`。FRS checkpoint
中的 `action_dim`、state conditioning 维度、触觉 token 数量和 key 顺序会在加载时校验。

## SmolVLA 纯视觉

复制 `deploy_smolvla/configs/deploy_smolvla_jax.yaml`，修改 checkpoint 和观测合同：

```yaml
checkpoint: /path/to/single_right_smolvla

observation:
  state_action_profile: single-right-arm-7x10
  data_type: vision
  language_prompt: Use the right hand to complete the task.
  single_arm_mode: true
  controlled_arm: right
  no_state_obs_mode: false
```

SmolVLA checkpoint 必须声明 `state_dim=7`、`action_dim=10`；客户端加载后会核对。

## SmolVLA + FRS

在右手 SmolVLA 配置上使用单手 FRS checkpoint：

```yaml
frs:
  checkpoint: /path/to/single_right_smolvla_frs
  inactive_arm_xyz_threshold_m: null
  gripper_gain: null
  tactile_keys:
    - observation.images.tactile_right_0
    - observation.images.tactile_right_1
```

`inactive_arm_xyz_threshold_m` 是双臂 20D 保护，单手 10D 模式必须关闭。FRS 会保留
右手夹爪第 9 维的基础 VLA 输出，只重定向其余动作维度。

## 切换方式

为双臂和右手各保存一份完整 YAML，不在同一文件里临时改 checkpoint。现有启动器已经
支持通过环境变量切换：

```bash
# Pi0.5 纯视觉
PI05_DEPLOY_CONFIG=/path/to/pi05_right.yaml \
  bash deploy_pi05/scripts/start_pi05.sh --check

# Pi0.5 + FRS
PI05_DEPLOY_CONFIG=/path/to/pi05_frs_right.yaml \
  bash deploy_pi05/scripts/start_pi05_frs.sh --check

# SmolVLA 纯视觉
FRS_DEPLOY_CONFIG=/path/to/smolvla_right.yaml \
  bash deploy_smolvla/scripts/start_smolvla.sh --check

# SmolVLA + FRS
FRS_DEPLOY_CONFIG=/path/to/smolvla_frs_right.yaml \
  bash deploy_smolvla/scripts/start_frs.sh --check
```

`--check` 只检查启动器选择，不加载模型或连接机器人。第一次真机测试应保持
`runtime.auto_start: false`，限制迭代次数，并在确认服务器、state/action 维度、相机和
右手触觉流都正确后再发送 START。
