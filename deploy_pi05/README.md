# Pi0.5 部署：纯视觉与 FRS

`/home/typhon/FRS_Tact/deploy_pi05` 是一个自包含的 Pi0.5 部署目录：它同时
提供纯视觉和 Pi0.5 + FRS 两种客户端，但它们使用不同的配置和一键启动脚本。
其中私有的 Pi0.5 JAX、FRS 与触觉依赖都位于本目录；不要使用
`/home/typhon/FRS_Tact` 根目录为 SmolVLA 准备的 Python 环境。

本迁移**不修改** `vb3_robot_server`，也不会复制 checkpoint、token、tokenizer
缓存或机器人端文件。机器人端必须由操作者使用其已经验证过的启动流程启动。

## 目录和模式

| 模式 | 配置 | 启动脚本 | 服务器协议 |
| --- | --- | --- | --- |
| 纯视觉 Pi0.5 | `configs/deploy_pi05.yaml` | `scripts/start_pi05.sh` | 旧版 `obs` / `action` 交换，无通用 ACK |
| Pi0.5 + FRS | `configs/deploy_pi05_frs.yaml` | `scripts/start_pi05_frs.sh` | `frs_steering_v1` |

两个 YAML 都是完整、独立的配置。纯视觉配置使用 `observation.data_type: vision`，
不含 `frs` 段；FRS 配置使用 `observation.data_type: vitac`，并保留 FRS steering
和触觉 encoder 参数。二者目前使用同一 Pi0.5 checkpoint、归一化统计、相机映射、
状态/动作维度与任务提示，只有观测和后处理路径不同。

## 首次安装

在部署机上执行：

```bash
cd /home/typhon/FRS_Tact/deploy_pi05 && uv sync --frozen
```

这会创建/更新 `deploy_pi05/.venv`，并依据本目录的 `pyproject.toml` 和
`uv.lock` 安装 Pi0.5 所需的固定 JAX 依赖，同时以 editable 方式安装本目录私有的
`src/lerobot`、`utils`、`train_pi05_frs` 和 `tactile_encoder`。不要在
`FRS_Tact` 根目录运行该命令或将这些依赖合并到根目录的 `.venv`。用于完整训练机
初始化时，也可以在本目录运行
`bash scripts/setup_env.sh`；该脚本可能安装系统依赖并检查 GPU，不是部署 dry-run。

## 启动顺序

1. 在独立终端先启动并确认 `vb3_robot_server` 已就绪、地址和 token 与客户端配置
   一致。对于 FRS，服务器必须已验证支持 `frs_steering_v1`。
2. 在另一个终端进入 `/home/typhon/FRS_Tact`，设置认证 token。
3. 先运行客户端 `--check`；确认 mode、config、Python 解释器和 token 来源正确。
4. 仅在机器人、服务器协议和安全条件全部确认后，启动一个有界客户端运行。

```bash
cd /home/typhon/FRS_Tact
export VB_ROBOT_TOKEN='...'

# 纯视觉 Pi0.5
bash deploy_pi05/scripts/start_pi05.sh --check
bash deploy_pi05/scripts/start_pi05.sh --max-iterations 2

# Pi0.5 + FRS：只在支持 frs_steering_v1 的真实 server 上运行
bash deploy_pi05/scripts/start_pi05_frs.sh --check
bash deploy_pi05/scripts/start_pi05_frs.sh --max-iterations 2
```

`--check` 不连接机器人，也不会打印 token 内容；它只显示模式、配置路径、解释器、
入口模块和 token 来源。一次只启动一种客户端模式。

## 环境变量覆盖

- `VB_ROBOT_TOKEN`：优先使用的机器人认证 token；绝不提交或打印它。
- `VB3_TOKEN_FILE`：未设置 `VB_ROBOT_TOKEN` 时读取的 token 文件，默认
  `/home/typhon/vb3_robot_server/token_list.txt`。
- `PI05_DEPLOY_CONFIG`：两个模式都使用的最高优先级 YAML 覆盖。
- `PI05_FRS_DEPLOY_CONFIG`：仅 FRS 启动脚本的兼容性覆盖；优先级低于
  `PI05_DEPLOY_CONFIG`。
- `PI05_PYTHON` / `PI05_FRS_PYTHON`：分别覆盖纯视觉 / FRS 的 Python 解释器。
- `VB3_PYTHON`：以上二者未设置时的共享解释器覆盖。
- `OPENPI_DATA_HOME`：可选的 openpi checkpoint/tokenizer 缓存目录。

若没有解释器覆盖，公共启动脚本依次选择本目录的 `.venv/bin/python` 和系统
`python3`。脚本将私有 `src`、本部署目录放在 `PYTHONPATH` 前面，避免导入
`FRS_Tact` 根目录的 SmolVLA 版本依赖。

公共启动脚本会先切换到 `/home/typhon/FRS_Tact/deploy_pi05` 再运行客户端。因此
配置中的相对日志目录 `outputs/...` 实际写入
`/home/typhon/FRS_Tact/deploy_pi05/outputs`，不会写到 `FRS_Tact` 仓库根目录。

## 外部资产

部署 YAML 只保存资产的路径；运行前确认这些路径存在且来自同一个训练版本：

- Pi0.5 checkpoint 与其 `assets/<asset_id>/norm_stats.json`；
- FRS checkpoint（仅 FRS 模式）；
- tactile encoder checkpoint（仅 FRS 模式）；
- 模型所需的 tokenizer / openpi 缓存（如配置允许下载）；
- `vb3_robot_server` 的 token 与已启动服务。

当前示例路径可在两个 `configs/deploy_pi05*.yaml` 中查看并按实际部署环境改写。
训练与 cache 准备工具所引用的默认配置也随目录迁移：
`configs/train_pi05_frs.yaml` 与 `configs/train_tactile_encoder.yaml`。
其中 FRS 的三阶段训练链路（tactile embedding、action cache、FRS training）可用
`bash scripts/start_frs_pi05_train.sh configs/train_pi05_frs.yaml` 从本目录启动；它会
检查实际数据集、encoder、checkpoint 与 GPU，不能当作部署客户端或 dry-run 使用。

## 协议与 dry-run 边界

纯视觉路径与旧版 ManiSkill-vitac 部署保持一致：发送完整 action chunk 后直接等待
下一帧观测，不消费通用 `action_ack`。如果你的旧版
`vb3_robot_server` 提供 `bimanual_smolvla.sh --dry-run`，它只能覆盖这条 legacy
纯视觉 `obs` / `action` 流程：

```bash
# Terminal A：legacy 纯视觉 dry-run（仅当该 server revision 提供此脚本）
cd /home/typhon/vb3_robot_server
bash scripts/bimanual_smolvla.sh --dry-run

# Terminal B：本目录的纯视觉客户端
cd /home/typhon/FRS_Tact
export VB_ROBOT_TOKEN='...'
bash deploy_pi05/scripts/start_pi05.sh --max-iterations 2
```

该 dry-run **不实现** `frs_steering_v1`，因此不存在可信的 FRS 硬件外 dry-run。
它不能用来验证 FRS 客户端，也不能替代真实服务器协议协商。

## 硬件安全

`--check` 和纯视觉 dry-run 都不是 FRS 机器人验证。开始真实 FRS 运行前，必须确认：

1. 已连接的真实 `vb3_robot_server` 明确协商为 `frs_steering_v1`；
2. checkpoint、norm stats、FRS checkpoint 和 tactile encoder 路径与任务匹配；
3. 只启动一个客户端，并使用小的 `--max-iterations`；
4. 受训操作员全程在场，急停可立即使用。

若发生断连、意外观测或动作错误，立即停止客户端并重新检查服务器和机器人状态；
客户端不会自动重连。
