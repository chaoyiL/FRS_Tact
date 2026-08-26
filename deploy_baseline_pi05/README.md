# Pi0.5 触觉 Direct Decoder 独立部署

本目录部署纯视觉 Pi0.5 与触觉直接动作 decoder。Pi0.5 每个 chunk 只生成一次归一化粗动作 `[B, 50, 20]`；每个唯一动作请求都读取最新四路触觉图像，编码为 `[B, 4, 512]`，重新运行 2 层 Transformer decoder，并发送当前索引对应的完整 20 维物理动作（包含第 9、19 维夹爪）。decoder 预测完整动作，不做粗动作回退。

四个触觉 token 的固定顺序是 `left0, right0, left1, right1`，对应 YAML 中：

1. `observation.images.tactile_left_0`
2. `observation.images.tactile_right_0`
3. `observation.images.tactile_left_1`
4. `observation.images.tactile_right_1`

## 服务器路径与环境

先编辑 `configs/deploy_baseline_pi05.yaml`。至少把以下 asset 路径改为服务器真实路径：

- `source.checkpoint`：纯视觉 Pi0.5 checkpoint；
- `norm_stats.dir` 和 `norm_stats.asset_id`：同一 Pi0.5 数据规范化资产；
- `tactile_encoder.checkpoint`：冻结的触觉 encoder checkpoint；
- `direct_decoder.checkpoint`：`train_baseline_pi05` 生成的正式 `best.pt`；
- `connection.address/port` 和 `logging.output_dir`。

上述四类模型/资产路径若写成相对路径，统一相对于 YAML 所在目录解析，与启动命令的当前目录无关。仓库提交的 YAML 含服务器占位路径，在按服务器实际位置修改前，`--check` 预期失败。

项目固定使用 Python `>=3.12,<3.13`，环境只位于本目录的 `.venv`：

```bash
cd /home/typhon/FRS_Tact
uv sync --project deploy_baseline_pi05
```

启动器不会回退到仓库根环境、原部署环境或系统 Python。可先做依赖轻量的配置与资产检查；它会真正调用 remote client 的 `--check`，确认 Pi0.5 checkpoint/`params/`、`norm_stats.json`、encoder `checkpoint.json` 及其参数文件、decoder `best.pt` 均可读，但不连接机器人、不加载 JAX/Torch，也不需要 token：

```bash
bash deploy_baseline_pi05/scripts/start_baseline_pi05.sh --check
```

机器人认证可以直接设置 `VB_ROBOT_TOKEN`，或把 token 文件路径放进 `VB3_TOKEN_FILE`。建议首次服务器联调限制为一个 chunk：

```bash
VB3_TOKEN_FILE=/path/to/token.txt \
bash deploy_baseline_pi05/scripts/start_baseline_pi05.sh \
  --config deploy_baseline_pi05/configs/deploy_baseline_pi05.yaml \
  --max-iterations 1
```

## 协议与 trace

机器人服务器必须提供 `frs_steering_v1` 调度消息，并以 `vitac` observation 同时发送两路视觉、20 维 state 和固定顺序的四路触觉图像。客户端完成 hello 后先发送 CONFIG，接收一帧服务器 observation，用同一帧完整 warmup Pi0.5、触觉 encoder 与 decoder；warmup 不发送动作，随后 START 会绑定该帧的 `obs_seq`。这里复用的只是 wire names 和 START/STEER/ACK/END 调度顺序：decoder 路径无 FRS、无两次 flow matching 积分、无 residual、无 x_base。纯视觉 Pi0.5 自身的标准采样不属于第二次触觉 FRS 修正。

当 `logging.save_observations: true` 时，每次运行会在 `logging.output_dir/session_<uuid>/` 创建独立 trace session。每条 chunk/steer 记录位于编号子目录，`metadata.json` 保存配置/checkpoint identity、请求 ID、action index、各阶段耗时和 delta RMS，`arrays.npz` 保存 observation、粗动作、decoder 完整输出与最终发送动作。

运行采用 fail-stop：触觉编码、decoder、有限值/幅值检查、协议顺序或 trace 写入任一失败时，不会发送替代粗动作；客户端尽力发送 stop、关闭连接，然后以非零状态退出。

本地 CPU 测试覆盖结构一致性、配置/checkpoint、协议顺序和合成输入，但不能证明服务器 GPU、真实 checkpoint、相机/触觉数据或 robot 闭环已经通过。正式运行前仍需在服务器完成 GPU warmup、一次受限 chunk 和急停验证。
