# deploy_deco

独立加载 DECO Stage 1 TorchScript，通过现有 VB `robot-bridge-v1` 普通 action-chunk
协议部署。它不导入 `train_deco`、`deploy_smolvla`、LeRobot 或 JAX。

## 权重合同

每个部署目录外部的 artifact 必须同时提供 TorchScript 和同名 sidecar：

```text
<checkpoint>.ts
<checkpoint>.ts.json
```

支持两个精确的 artifact profile：

- `dual-arm-20x20`：20D `7+7+6` state 和 `32x20` action；
- `single-right-arm-7x10`：7D 右臂 state 和 `32x10` action，并声明
  `controlled_arms: [right]`。客户端会将它扩展成服务端需要的 20D 双臂 action。

启动时会校验 profile、sidecar、TorchScript SHA256、双相机顺序、state/action shape、
Rotation6D columns、绝对夹爪和 30Hz 采样率。TorchScript 内部已经完成图像预处理、
state normalization 和 action denormalization。两个 checked-in 配置都要求 GPU 0；
实际权重路径由各自 YAML 的 `checkpoint` 决定。

## 启动

使用独立环境安装依赖后：

```bash
uv sync --project deploy_deco
uv run --project deploy_deco python -m deploy_deco.remote_client \
  --config deploy_deco/configs/deploy_deco.yaml --check

VB_ROBOT_TOKEN='<token-list 中的一项>' \
uv run --project deploy_deco python -m deploy_deco.remote_client \
  --config deploy_deco/configs/deploy_deco.yaml
```

也可以设置 `PYTHON_BIN` 后运行 `bash deploy_deco/scripts/start_deco.sh`。

客户端的 `VB_ROBOT_TOKEN` 必须是服务端允许列表中的一个非空 token。服务端 launcher
默认读取 `/home/typhon/vb3_robot_server/token_list.txt`；也可给服务端传入
`--token-file /path/to/token_list.txt`，或设置 `VB3_TOKEN_FILE=/path/to/token_list.txt`。
无论选择哪种文件，客户端都要把其中同一个 token 设置为 `VB_ROBOT_TOKEN`。

### Right-arm insert deployment

The right-arm launcher runs the 7D-to-10D insert policy while the robot server
remains in bimanual mode. The client sends an identity hold action for the left
arm and places the model output in the right-arm action block.

先离线检查右臂配置和 artifact（该命令不连接机器人）：

```bash
bash deploy_deco/scripts/start_deco_right.sh --check
```

然后从服务端 token 文件选择一个非空 token，供客户端使用。服务端 launcher 默认使用
`/home/typhon/vb3_robot_server/token_list.txt`；若使用其他文件，可传
`--token-file /path/to/token_list.txt` 或设置 `VB3_TOKEN_FILE`。启动服务端后，再运行一次
需要人工确认的客户端迭代：

```bash
# Terminal 1 (server; defaults to /home/typhon/vb3_robot_server/token_list.txt)
bash /home/typhon/vb3_robot_server/scripts/bimanual_deco.sh
```

```bash
# Terminal 2 (client; use one token present in the server's selected token file)
export VB_ROBOT_TOKEN='<token-list 中的一项>'
bash deploy_deco/scripts/start_deco_right.sh --max-iterations 1
```

### Stage 2 tactile right-arm deployment

Stage 2 uses the four tactile fields in metadata order:
`tactile_left_0`, `tactile_right_0`, `tactile_left_1`, `tactile_right_1`.
Only the client blacks out `camera0`; the server remains bimanual.

先检查配置和 artifact（不连接机器人）：

```bash
bash deploy_deco/scripts/start_deco_stage2_right.sh --check
```

查看服务端启动参数但不连接服务端：

```bash
bash deploy_deco/scripts/start_deco_stage2_right.sh --server-dry-run --max-iterations 1
```

仅观察客户端流程、不发送动作：

```bash
bash deploy_deco/scripts/start_deco_stage2_right.sh --observe-only
```

启动一次客户端迭代（服务端需已启动）：

```bash
bash deploy_deco/scripts/start_deco_stage2_right.sh --max-iterations 1
```

正式启动 Stage 2 客户端（服务端需已启动）：

```bash
bash deploy_deco/scripts/start_deco_stage2_right.sh
```

## 服务端兼容

当前 `vb3_robot_server` 已经提供与训练一致的：

- `camera0` / `camera1` 两路腕部 RGB；
- relative-start `7+7` 加 left-relative-right `6` 的 20D state；
- TCP delta + Rotation6D matrix columns + absolute gripper 的动作转换和安全检查。

因此使用 legacy chunk，不设置 `frs_steering_v1`。配置暂时复用服务端已有的
`deco_vision_224` observation profile；该名称只代表两路 224x224 RGB wire contract。
