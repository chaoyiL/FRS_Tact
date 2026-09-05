# Pi0.5 触觉 Direct Decoder 独立部署

`deploy_baseline_pi05` 独立加载冻结 Pi0.5、0824 触觉编码器和 `train_baseline_pi05` 生成的 `best.pt`。2 层 Transformer decoder 预测完整归一化动作；无 FRS、无两次 flow matching 积分、无 residual、无 x_base。模型代码不依赖原 `deploy_pi05` 或训练运行时。

| 权重类型 | 模型状态 | 模型动作 | 触觉 token |
|---|---|---|---|
| 右臂 | 7D | `[B,50,10]` | `[B,2,512]` |
| 双臂 | 20D | `[B,50,20]` | `[B,4,512]` |

四路默认模型顺序为 `left0, right0, left1, right1`，右臂权重顺序为 `tactile_left_1, tactile_right_1`；实际 token 数量和顺序必须匹配 checkpoint。Pi0.5 每个 chunk 只推理一次；每个新动作请求读取最新触觉并运行 decoder，选择该请求索引的动作。

## 独立入口与配置

模型客户端分为两个入口：

- `scripts/start_baseline_pi05_single_arm.sh`：默认使用 task3 右臂配置。
- `scripts/start_baseline_pi05_bimanual.sh`：默认使用双臂配置。

两者均支持 `--config` 覆盖默认配置，以及 `--check`、`--max-iterations`。环境与认证逻辑共用 `start_baseline_pi05.sh`。

机器人服务：`/home/typhon/vb3_robot_server/scripts/bimanual_baseline_pi05.sh`，进入独立的 `deploy_scripts/bimanual_baseline_pi05_online.py`。它使用独立 `pi05_baseline_deployment_config.py`、`pi05_baseline_server_config.py`、`baseline_execution.py`、`baseline_protocol.py` 和 `baseline_trace.py`，不进入旧 Pi0.5/FRS 服务入口。通用机器人驱动、相机和控制器仍由 robot server 提供。`frs_steering_v1` 是为兼容通用 RobotClient 保留的 wire 调度名称，不表示运行 FRS 算法。

两端读取同一份 YAML：

- `configs/deploy_baseline_pi05_task3.yaml`：本机 task3 右臂，使用修正触觉来源后重训的 `outputs/baseline_pi05/task3_right_two_face/decoder/best.pt`，decoder 使用 CUDA。该文件必须在重训完成后存在。
- `configs/deploy_baseline_pi05_bimanual.yaml`：双臂模板，填写与双臂 decoder 配套的模型、归一化和权重路径。
- `configs/deploy_baseline_pi05.yaml`：保留原双臂示例，含占位 decoder 路径。

`observation.single_arm_mode: true` 对应模型 action/state 为 10/7，false 对应 20/20。服务器观察和动作通信始终为 20D：右臂客户端提取 `state[7:14]`，将模型动作放到 wire `[10:20]`；左臂填零位移、单位旋转，第9索引夹爪使用当前请求的 `state[6]`。服务端还会验证左臂保持动作。

## 相机与触觉映射

`source.camera_map` 是模型视觉槽到服务器相机字段的映射，用户按设备布局修改。`tactile_encoder.tactile_keys` 是训练权重中的传感器顺序，不应随硬件命名更改；通过 `tactile_encoder.key_map` 将这些模型名称映射到服务器字段。

服务器的命名规则是“相机内部左右触觉区域＋相机编号”：

| 服务器字段 | 物理来源（默认相机顺序） |
|---|---|
| `tactile_left_0` / `tactile_right_0` | 左手相机的两块触觉区域 |
| `tactile_left_1` / `tactile_right_1` | 右手相机的两块触觉区域 |

完整字段前缀均为 `observation.images.`。task3 训练和部署均使用右手相机的两块触觉区域，模型顺序与 `key_map` 如下：

```yaml
tactile_encoder:
  tactile_keys:
    - observation.images.tactile_left_1
    - observation.images.tactile_right_1
  key_map:
    observation.images.tactile_left_1: observation.images.tactile_left_1
    observation.images.tactile_right_1: observation.images.tactile_right_1
```

旧 `outputs/baseline_pi05/task3/decoder` 权重使用了错误的跨臂 `tactile_right_0/right_1` 输入，新部署代码会拒绝该旧合同。旧部署配置保存在该目录的 `before_touch_fix_deploy_baseline_pi05_task3.yaml`，仅供历史实验溯源；不能通过修改映射、重命名 checkpoint 元数据或直接加载旧权重来替代重训。新训练输出与部署日志独立保存在 `task3_right_two_face`。

`server.cam_path` 可设置两个物理相机路径，或给服务脚本重复传两次 `--cam-path`。本次触觉纠正保持原来的 10 Hz 控制频率与 50 步重规划设置，便于对照重训效果。

## 环境、检查与启动

客户端固定使用本目录 `.venv`，Python 3.12：

```bash
cd /home/typhon/FRS_Tact
uv sync --project deploy_baseline_pi05
```

两端 `--check` 均不连接机器人、不读取 token 或启动硬件。客户端检查本地资产可读性，不加载模型权重；服务端检查共享配置与硬件参数。首次真实模型运行包含 GPU JIT warmup。

```bash
bash /home/typhon/vb3_robot_server/scripts/bimanual_baseline_pi05.sh \
  --config /home/typhon/FRS_Tact/deploy_baseline_pi05/configs/deploy_baseline_pi05_task3.yaml --check

bash /home/typhon/FRS_Tact/deploy_baseline_pi05/scripts/start_baseline_pi05_single_arm.sh --check
```

确认映射后，终端一启动服务：

```bash
bash /home/typhon/vb3_robot_server/scripts/bimanual_baseline_pi05.sh \
  --config /home/typhon/FRS_Tact/deploy_baseline_pi05/configs/deploy_baseline_pi05_task3.yaml
```

终端二启动模型客户端：

```bash
VB3_TOKEN_FILE=/home/typhon/vb3_robot_server/token_list.txt \
CUDA_VISIBLE_DEVICES=0 \
bash /home/typhon/FRS_Tact/deploy_baseline_pi05/scripts/start_baseline_pi05_single_arm.sh --max-iterations 1
```

默认 `runtime.auto_start: false`，模型完成真实观察 warmup 后等待回车发送 START。`--max-iterations 1` 执行一个 chunk；需要持续运行时去掉该参数。双臂部署时，机器人服务的 `--config` 换成配置好的 bimanual YAML，模型客户端使用 `start_baseline_pi05_bimanual.sh`。模型相关相对路径以 YAML 所在目录解析。

## 调度与日志

服务器发送 `vitac` 观察；START 绑定 warmup 的 `obs_seq`。baseline 默认 `server.task: 0`，禁用旧任务专用动作后处理；这个字段不是训练 task3/task4 编号。夹爪迟滞默认关闭，直接执行 decoder 的夹爪输出。

运行采用 fail-stop：推理异常、非有限动作、幅值/增量限制、请求顺序或客户端 trace 写盘失败时停止，不发送替代粗动作。`logging.save_observations` 保存独立客户端 trace session，包含源模型身份、粗动作、decoder 输出、实际 wire 动作和耗时；服务端使用独立 baseline trace；服务端诊断队列满时会提示并丢弃该条日志，不阻塞控制。

离线测试覆盖10/20D权重加载、两/四触觉、最新触觉重算、左臂保持、重复请求、调度与清理。新 task3 权重应在重训完成后独立验证；旧权重的离线结果不能代表新权重，离线测试不代表真实 robot 闭环任务成功率。
