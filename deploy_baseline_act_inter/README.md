# Pi0.5 action-origin Flow 独立部署

部署冻结 Pi0.5、0824 ResNet18 与 `train_baseline_act_inter` 的 version-2 `best.pt`。触觉修正网络复刻 TacFRS Flow：6层 Transformer、256D、4 heads、FFN 1024、共享 GRU 512→256、当前 state 条件、10 点触觉历史、history stride 3。前向为 FireFlow 10 步（11 次速度评估）。起点直接采用 VLA 动作，不做反向积分。

| 权重 | state | 动作 | 触觉历史 |
|---|---|---|---|
| 右臂 | `[B,7]` | `[B,50,10]` | `[B,10,2,512]` |
| 双臂 | `[B,20]` | `[B,50,20]` | `[B,10,4,512]` |

当前帧编码器输出原始 `[B,2,512]`／`[B,4,512]` ResNet 特征，历史随后由 runtime 组装。传感器顺序为双臂 `left0,right0,left1,right1`，右臂 `tactile_left_1,tactile_right_1`；后缀表示相机／手编号。

## 入口

客户端使用本目录 `.venv`（Python 3.12）：

```bash
uv sync --project deploy_baseline_act_inter
bash deploy_baseline_act_inter/scripts/start_baseline_act_inter_single_arm.sh --check
bash deploy_baseline_act_inter/scripts/start_baseline_act_inter_bimanual.sh --check
```

两个入口支持 `--config`、`--check`、`--max-iterations`，最终调用 `remote_client.py`。右臂默认 task3，task4 使用 `configs/deploy_baseline_act_inter_task4.yaml`。先修改配置里的本地模型、归一化、encoder 和新版 `decoder_flow_v2/best.pt` 路径。`--check` 不连接 robot、不加载 GPU 模型、不读取 token。

机器人服务仍由外部 `/home/typhon/vb3_robot_server/scripts/bimanual_baseline_pi05.sh` 提供，读取同一份 YAML。通信观察为 `vitac`，采用 `frs_steering_v1` 调度协议。

```bash
bash /home/typhon/vb3_robot_server/scripts/bimanual_baseline_pi05.sh \
  --config /path/to/deploy_baseline_act_inter_task3.yaml

VB3_TOKEN_FILE=/path/to/token_list.txt \
  bash deploy_baseline_act_inter/scripts/start_baseline_act_inter_single_arm.sh --max-iterations 1
```

## 在线执行时序

1. 用 episode 初始观察初始化触觉历史；早期窗口重复最早帧。
2. 每个 chunk 调用一次 Pi0.5，缓存归一化 VLA 动作 `[1,50,A]`。
3. 每个唯一 steer request 编码最新触觉，追加历史，读取并归一化当前 state。
4. 组成 `[1,10,S,512]` 历史，从同一 VLA 动作重新执行 FireFlow，选择 action_index。
5. 保留 VLA 夹爪，反归一化后发送动作，等待匹配 ACK。
6. chunk 结束只清理粗动作／请求缓存；触觉历史跨 chunk 保留。

重复请求校验触觉与 state：完全相同则返回缓存，不推进历史；冲突则停止。warmup 完成后重置初始历史，避免编译样本污染真实时序。重开 episode 调用 `reset_episode`。

右臂模型读取 wire `state[7:14]`，输出放在 `[10:20]`；左臂保持零位移、单位旋转及当前左夹爪。wire 始终为 20D。

## 权重与运行记录

旧 2 层 Direct Decoder 的 version-1 权重无法加载；不支持仅修改 metadata 冒充新版。训练／部署网络的结构、输入输出、权重键、state 维度、触觉顺序和积分参数必须一致。

部署采用 fail-stop：异常、非有限动作、幅值／修正幅度超限、调度冲突或 trace 写盘错误时停止。`logging.save_observations` 保存观察、粗动作、修正结果、wire 动作和耗时。`runtime.auto_start: false` 在实际观察 warmup 后等待启动确认。
