# State-only structured-noise Flow 部署

本目录独立部署冻结 Pi0.5 和 `train_baseline_state` 生成的 state-only Flow 权重。无需触觉 encoder、触觉图像、触觉历史或接触 Gate。视觉／任务指令仅用于上游 VLA；修正网络的唯一条件是当前 normalized state。

网络：6 层、256D、4 heads、FFN 1024、GELU、Pre-LayerNorm、一个 state 条件 token；右臂 7D state／10D action，双臂 20D state／20D action，动作长度均为 50。SlerpFlow 反向 50 步构造 structured noise；FireFlow 10 步完成修正。

## 入口与配置

```bash
uv sync --project deploy_baseline_state
bash deploy_baseline_state/scripts/start_baseline_state_single_arm.sh --check
bash deploy_baseline_state/scripts/start_baseline_state_bimanual.sh --check
```

两者进入 `remote_client.py`，支持 `--config`、`--check`、`--max-iterations`。右臂默认 task3，task4 配置为 `configs/deploy_baseline_state_task4.yaml`；双臂使用 bimanual 配置。先替换 source、norm_stats 和 decoder 权重路径。`--check` 只检查本地配置／资产，不连接机器人、不读取 token 或加载模型。

```bash
VB3_TOKEN_FILE=/path/to/token_list.txt \
  bash deploy_baseline_state/scripts/start_baseline_state_single_arm.sh --max-iterations 1
```

机器人服务端须支持 `vision` 观察与 `frs_steering_v1` 逐动作协议，接收 `make_server_config` 生成的空 `frs_tactile_keys`。沿用现有 `pi05_vitac_224` wire profile 名称，但客户端没有触觉输入要求；服务器观察至少包含 state 和配置的 224×224 RGB 相机字段。外部机器人服务端不包含在本目录中，需使用匹配该配置的服务入口。

## 执行时序

1. chunk start：冻结 Pi0.5 用初始视觉／state／指令进行一次动作采样，共享 prefix cache 后反向积分得到 structured noise。
2. coarse action 与 structured noise 在该 chunk 内固定；反向积分在 VLA 完整原生宽度执行。
3. 每个唯一 steer request：读取最新 state，归一化，使用同一个 structured noise 完成 FireFlow 解码，选择 action_index。
4. 保留 VLA 夹爪，反归一化并发送动作，等待对应 ACK。
5. 相同请求／state 返回缓存；同一请求 ID 的 state 或 action index 冲突则停止。触觉和新视觉字段不会推进／改变 state-only 条件。

右臂从 wire `state[7:14]` 取模型 state，模型动作映射到 `[10:20]`，左臂保持零位移、单位旋转及当前左夹爪；wire 动作仍为 20D。

采用现有 fail-stop 与有界 trace 保存机制。trace 包含源模型身份、VLA 动作、structured noise、修正动作和时序。权重 schema／mode、网络维度、源模型、归一化、reverse solver／steps 必须与训练一致；其他 baseline 的 checkpoint 会被拒绝。默认 warmup 后等待启动确认。
