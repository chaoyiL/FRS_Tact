# deploy_deco

独立加载 DECO Stage 1 TorchScript，通过现有 VB `robot-bridge-v1` 普通 action-chunk
协议部署。它不导入 `train_deco`、`deploy_smolvla`、LeRobot 或 JAX。

## 权重合同

部署目录外部的 artifact 必须同时提供：

```text
deco_stage1_best.ts
deco_stage1_best.ts.json
```

启动时会校验 sidecar、TorchScript SHA256、双相机顺序、20D `7+7+6` state、
32x20 action、Rotation6D columns、绝对夹爪和 30Hz 采样率。TorchScript 内部已经完成
图像预处理、state normalization 和 action denormalization。

当前示例配置临时指向 `/home/hillbot/deco/deco/deco_stage1_latest.ts`。这份权重完整，
当前 traced artifact 的图内设备为 `cuda:0`，所以本版本配置明确要求 GPU 0。
但 checkpoint 记录的历史 best loss 更低；找回 best artifact 后应更新 `checkpoint`。

## 启动

使用独立环境安装依赖后：

```bash
uv sync --project deploy_deco
uv run --project deploy_deco python -m deploy_deco.remote_client \
  --config deploy_deco/configs/deploy_deco.yaml --check

VB_ROBOT_TOKEN=... \
uv run --project deploy_deco python -m deploy_deco.remote_client \
  --config deploy_deco/configs/deploy_deco.yaml
```

也可以设置 `PYTHON_BIN` 后运行 `bash deploy_deco/scripts/start_deco.sh`。

### Right-arm insert deployment

The right-arm launcher runs the 7D-to-10D insert policy while the robot server
remains in bimanual mode. The client sends an identity hold action for the left
arm and places the model output in the right-arm action block.

Start the server and then run one manually confirmed client iteration:

```bash
bash /home/typhon/vb3_robot_server/scripts/bimanual_deco.sh
bash deploy_deco/scripts/start_deco_right.sh --max-iterations 1
```

## 服务端兼容

当前 `vb3_robot_server` 已经提供与训练一致的：

- `camera0` / `camera1` 两路腕部 RGB；
- relative-start `7+7` 加 left-relative-right `6` 的 20D state；
- TCP delta + Rotation6D matrix columns + absolute gripper 的动作转换和安全检查。

因此使用 legacy chunk，不设置 `frs_steering_v1`。配置暂时复用服务端已有的
`deco_vision_224` observation profile；该名称只代表两路 224x224 RGB wire contract。
