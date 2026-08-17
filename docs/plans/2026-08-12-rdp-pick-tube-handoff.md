# Pick-tube RDP 快速适配交接

日期：2026-08-12

## 目标与范围

在 `reactive_diffusion_policy-main/` 中快速实现一个实验室可训练、可部署的视觉—触觉 RDP：

- 首版使用 `pick_tube_01`～`pick_tube_04`；后续能简单追加 05、06。
- 输入与现有 VT-SmolVLA 对齐：两路 RGB、四路触觉图像、20 维 state。
- 使用现有 VT-SmolVLA tactile encoder，把每路触觉图像编码成 512 维特征。
- 动作模型预测窗口为 20，RDP 有效动作窗口为 10。
- 采用 RDP slow/fast 执行：慢速视觉 diffusion 生成 latent plan，快速触觉 decoder 每个控制周期更新当前动作。
- 这是实验室快速实现，不建设通用数据框架、复杂恢复协议或大量安全检查。

当前数据转换、AT/LDP 训练配置和 WebSocket 真机部署均已实现；部署入口见
`reactive_diffusion_policy-main/DEPLOY_PICK_TUBE_RDP.md`。

## 已检查的代码和数据

- RDP 根目录：`/home/hillbot/FRS_Tact/reactive_diffusion_policy-main`
- 实际数据：`/home/hillbot/datasets/pick_tube_01`
- 当前四数据集服务器路径参考：
  `/DATA/ljl/substage/pick_tube_01`～`pick_tube_04`
- 触觉 encoder checkpoint：`data/encoder_ckpt_0809`
- 默认触觉缓存根目录：`data/tactile_embeddings_encoder0809`
- 预计算入口：`precompute_pick_tube_v21_tactile_embeddings.py`
- 数据准备入口：`scripts/setup_pick_tube_data.sh`

`pick_tube_01` 的 metadata：

- LeRobot codebase version：v2.1
- 400 episodes
- 174571 frames
- 30 FPS
- 所有图像为 224×224 RGB
- 图像直接存于每个 episode 的 Parquet，不是 MP4
- 两路 RGB：`observation.images.camera0`、`observation.images.camera1`
- 四路触觉：
  - `observation.images.tactile_left_0`
  - `observation.images.tactile_right_0`
  - `observation.images.tactile_left_1`
  - `observation.images.tactile_right_1`
- `observation.state`：[20] float32
- `actions`：[20] float32

## 从实际数据确认的 state/action contract

不要使用 RDP 原始 `RealRunner` 对 20 维动作的默认切片。实际数据的动作是每只手连续 10 维。

### observation.state

实际 state 是 `7 + 7 + 6`：左右臂各 7 维，再加“左末端相对右末端”的 6D 位姿。左右臂前两段位姿以各自 episode 起始位姿为零点，并不是机器人世界坐标中的绝对 TCP 位姿。

| 索引 | 含义 |
|---|---|
| 0:3 | 左臂当前相对位置 xyz（相对 episode 起始位姿） |
| 3:6 | 左臂当前相对旋转 rotvec（相对 episode 起始姿态） |
| 6 | 左夹爪绝对开度 |
| 7:10 | 右臂当前相对位置 xyz（相对 episode 起始位姿） |
| 10:13 | 右臂当前相对旋转 rotvec（相对 episode 起始姿态） |
| 13 | 右夹爪绝对开度 |
| 14:17 | 左末端在右末端坐标系中的位置 xyz |
| 17:20 | 左末端在右末端坐标系中的旋转 rotvec |

尾部 6 维不是相机位姿。其精确定义是：

- `T_left_relative_to_right = inverse(T_right) @ T_left`，再编码成 `xyz + rotvec`。
- 采集/部署端 `get_real_umi_obs_dict` 和数据转换端 `encode_state` 使用同一公式。
- 本地 representation 单元测试也固定检查了该方向，不能交换为右相对左。

快速版仍将完整 20 维 state 原样复制到 RDP Zarr。部署端复用服务器现有的 state 构造，因此会从左右末端实时位姿计算最后 6 维，不需要额外的头部位姿源。

### actions

| 索引 | 含义 |
|---|---|
| 0:3 | 左臂相对平移 |
| 3:9 | 左臂相对旋转的 ortho-6D 表示 |
| 9 | 左夹爪绝对开度 |
| 10:13 | 右臂相对平移 |
| 13:19 | 右臂相对旋转的 ortho-6D 表示 |
| 19 | 右夹爪绝对开度 |

数值验证结果：

- 两段 action rotation 的两个 3D 向量范数约为 1、互相正交。
- action rotation 接近单位旋转 `[1,0,0,0,1,0]`，不是可以逐元素相加的旋转差值。
- 把 state 的 3 维旋转解释为 rotvec 时，
  `R_action = inverse(R_state_t) @ R_state_t+1` 的平均旋转误差约为 `4e-8 rad`。
- `action[9] == state_t[6]`，`action[19] == state_t[13]`，样本分析中误差为 0。
- action xyz 与相邻 state xyz 差值高度相关，但存在控制跟踪误差；不要根据 state 重新生成 action，直接复制原始 `actions`。

因此动作是混合 contract：平移相对量、旋转相对变换、夹爪绝对量。配置中的 RDP `delta_action` 和 `relative_action` 都应设为 `false`，避免二次转换。

## 固定的数据适配方案

采用一次性 LeRobot → RDP Zarr 转换，不重写 RDP 的 sampler。

首版 Zarr：

```text
replay_buffer.zarr/
  data/
    camera1                 [N, 224, 224, 3] uint8
    camera2                 [N, 224, 224, 3] uint8
    observation_state       [N, 20] float32
    tactile_embedding       [N, 2048] float16
    action                  [N, 20] float32
  meta/
    episode_ends            [num_episodes] int64
```

映射：

- source `camera0` → RDP `camera1`
- source `camera1` → RDP `camera2`
- 四路 `[4,512]` tactile cache 按固定 key 顺序展平为 `[2048]`
- `observation.state` 原样复制
- `actions` 原样复制并在 Zarr 中命名为 `action`
- 合并 01～04 时只累计 `episode_ends`，禁止窗口跨 episode

为了快速实现，不额外复制四路触觉原图到 Zarr；训练直接读取预计算 embedding。RGB 仍写入 Zarr，复用 RDP 原生视觉 dataset。

最低限度检查只有：字段存在、state/action 维度为 20、触觉为 `[4,512]`、每个 episode 长度一致。不要增加复杂 manifest、哈希、断点事务和自动修复。

## 触觉 encoder 接入

固定使用现有冻结的 encoder 0809 JAX/Flax ResNet18 权重：

- checkpoint：`/DATA/ljl/substage/checkpoints/encoder_ckpt_05`
- 图像尺寸：224
- 四路共享 backbone
- 每路输出 512 维
- 每帧输出 `[4,512]`，RDP 中展平为 2048 维

训练阶段直接复用 `data/tactile_embeddings_encoder0809` 下的 `.npy` cache，不在 PyTorch DataLoader 中运行 JAX。

部署端使用 `reactive_diffusion_policy/deploy/tactile_encoder_torch.py` 直接读取原 Flax
checkpoint 的 `.npz` 权重，不导入 JAX/Flax。与已有缓存首帧对齐时，2048 维拼接
embedding 的最大绝对误差约 `1.99e-4`，最小 cosine similarity 约 `0.9999976`。

## 固定的 RDP 模型方案

### Stage 1：Asymmetric Tokenizer

- action dim：20
- horizon：20
- `use_conv_encoder: true`
- `use_rnn_decoder: true`
- fast temporal condition：`tactile_embedding [T,2048]`
- `n_latent_dims` 首版从 32 开始
- `n_embed` 首版保持 16
- RNN hidden dim 首版使用 256
- `delta_action: false`
- `relative_action: false`

AT 学习完整 20 步动作，RNN decoder 使用随时间到达的触觉 embedding 对动作进行快速修正。

### Stage 2：Latent Diffusion Policy

slow observation：

```text
camera1              [3,224,224]
camera2              [3,224,224]
observation_state    [20]
tactile_embedding    [2048]  # 当前帧
```

建议保持 RDP 的两个独立 ResNet18 RGB encoder，第一版不共享视觉 backbone。RGB 使用 RDP 原有 resize/crop；触觉 embedding 不做图像增强。

核心时间参数：

```yaml
horizon: 20
n_action_steps: 10
n_obs_steps: 1
control_fps: 30
inference_fps: 6
tcp_action_update_interval: 5
gripper_action_update_interval: 5
latency_step: 0
gripper_latency_step: 0
```

原 RDP 用公式 `horizon - dataset_obs_steps + 1` 计算 `n_action_steps`，这里不要使用该公式，直接固定为 10。

## RDP 风格执行时序

当前数据是 30 FPS，因此首版采用：

- 机器人控制和 fast tactile decoder：30 Hz
- slow visual diffusion：6 Hz
- 每 5 个控制周期更新一次 latent plan
- 每个控制周期只向机器人发送当前一个 20 维动作
- 新 latent plan 到达后覆盖 ensemble buffer 中旧计划的未来部分
- ensemble mode 首版使用原 RDP 的 `new`

不要照搬原示例的 `tcp_action_update_interval: 16`；本项目有效动作窗口只有 10。更新间隔固定为 5，保证旧窗口耗尽前产生新计划。

部署时模型输出 contract：

```text
action_pred: [B,20,20]  # 完整计划
action:      [B,10,20]  # 有效窗口
robot step:  [20]       # 每个 30 Hz 控制周期发送一步
```

原 RDP `RealRunner.post_process_action` 假设动作排列为 `[left_pose9,right_pose9,left_grip,right_grip]`，与数据不符。快速版只改这一处切片/转换，保持模型输出原始布局：

```text
[left_xyz, left_relative_rot6d, left_gripper,
 right_xyz, right_relative_rot6d, right_gripper]
```

相对旋转必须通过矩阵组合，不可逐元素累加。部署应复用 RDP `relative_actions_to_absolute_actions` 的矩阵思想，但为每臂连续 10 维布局写一个很小的专用转换函数。

## 数据划分与归一化

- 首版使用 01～04，合并后按 episode 随机划分 90% train / 10% val。
- 使用 RDP 自己的 normalizer，不要求复用 SmolVLA split 或 normalization。
- split seed 使用 42，跟原 RDP 默认一致。
- state 20 维按 low-dim 标准化；其中 `14:20` 是左末端相对右末端的 6D 位姿。
- action 20 维使用 RDP action normalizer，但不要对动作做额外差分或 relative conversion。
- tactile embedding 2048 维按训练集统计标准化。
- 四数据集直接按帧合并采样；首版不实现 dataset-balanced sampler。

## GPU/环境决定

目标 GPU：RTX 5080、RTX PRO 6000、RTX 4090、H100。

- 不使用原仓库建议的 PyTorch 1.13.1 + CUDA 11.7。
- 新建独立 RDP 环境，使用能支持 Blackwell 的 PyTorch CUDA 12.8 或更新构建。
- 训练使用 BF16 mixed precision。
- AT 首版单 GPU。
- Latent Diffusion 使用 `accelerate launch`，需要时再启用多 GPU。
- 不与现有 SmolVLA JAX `.venv` 混装。

## 实现状态

1. LeRobot 01～04 → RDP Zarr 转换和 AT/LDP batch 验证已完成。
2. pick-tube AT/LDP 配置、环境/数据/训练脚本已完成。
3. PyTorch tactile encoder、RDP WebSocket 客户端和单步 slow/fast runtime 已完成。
4. `vb3_robot_server-main` 已增加 RDP 协商、224 图像和 receive-time 单步调度。
5. fake policy、真实 checkpoint/真实 LeRobot 帧以及服务器回归测试均已通过。

第一版不要做：

- 通用 LeRobot 多版本抽象
- 复杂 cache 校验/恢复
- dataset-balanced sampler
- 六数据集支持（只把 source list 写成容易追加）
- ROS/机器人服务重构
- 大量真机安全策略或自动回退

## 新会话开场提示词

```text
请阅读 docs/plans/2026-08-12-rdp-pick-tube-handoff.md，并基于其中已经确认的
实际数据 contract，在 reactive_diffusion_policy-main 中实现实验室快速版适配。
先完成 LeRobot 01～04 到 RDP Zarr 的转换和最小 batch 验证，再实现 AT/LDP 配置与
启动脚本。保持实现精简，不增加通用框架、复杂安全检查或提交 reactive_diffusion_policy-main。
```

## 尚未阻塞实施的未知项

- 正式 LDP checkpoint 在目标策略机上的 slow diffusion 实测延迟仍需记录；默认先用 8 diffusion steps、每 5 帧更新。
- 策略机到机器人机的网络抖动仍需真机观察；单步动作已改为 receive-time 调度，不依赖旧观测时间戳。

