# 原始 RDP 训练入口

本入口依据本机 `train_RDP/reactive_diffusion_policy-main` 恢复原始 RDP 的训练目标和动作表示。核心参数取原版 `at/at_peel.yaml` 和两个训练 workspace 配置。它是一组重新训练的基线，不能据此预先断定 0902 的异常已经解决。

## 启动

在 `/home/typhon/FRS_Tact` 执行：

```bash
RUN_ID=insert01_original_rdp \
bash train_RDP/scripts/train_rdp_baseline.sh all
```

依次训练 AT 和 LDP；LDP 加载这次 AT 的 `checkpoints/latest.ckpt`。默认解释器为 `train_RDP/.venv/bin/python`，日志使用 W&B offline 模式，无需登录。默认 FP32；若希望 LDP 使用 BF16，加 `MIXED_PRECISION=bf16`。AT 保持原版 FP32 训练。

按当前训练安排，默认 AT 60 epoch、LDP 40 epoch，batch size 均为 64；训练目标与模型结构沿用原版。可以通过 `AT_EPOCHS`、`LDP_EPOCHS`、`AT_BATCH`、`LDP_BATCH` 调整，例如：

```bash
RUN_ID=insert01_original_rdp_20e AT_EPOCHS=20 LDP_EPOCHS=20 \
bash train_RDP/scripts/train_rdp_baseline.sh all
```

查看实际命令但不启动：

```bash
DRY_RUN=1 bash train_RDP/scripts/train_rdp_baseline.sh all
```

单独训练 LDP：

```bash
AT_CKPT=/完整路径/at/checkpoints/latest.ckpt \
RUN_ID=insert01_original_rdp_ldp \
bash train_RDP/scripts/train_rdp_baseline.sh ldp
```

输出为 `train_RDP/data/outputs/rdp_baseline/<RUN_ID>/{at,ldp}/`，各阶段保存 `.hydra/config.yaml`、`logs.json.txt` 和 `checkpoints/latest.ckpt`，LDP 另存 `normalizer.pkl`。最新文件在最后一个训练 epoch 也会保存；另保留普通训练损失最小的 top-1。续训需指定原 `RUN_ID`，并在命令末尾加 `training.resume=true`；`num_epochs` 表示总目标 epoch 数。

## 默认数据与输入

默认读取 `insert_01` 的 500 段已有转换数据：

```text
data/rdp_insert01_encoder0824/insert01_pca30_single_right_rdp_zarr/replay_buffer.zarr
data/rdp_insert01_encoder0824/tactile_cache/KaiyueChen/insert_01/embeddings.npy
checkpoints/model/rdp_0902/insert/pca/tactile_pca_insert_01_02_encoder0824_2x15.npz
```

分别为示教轨迹/RGB、encoder0824 的原始触觉 embedding 缓存、固定 PCA 文件。数据适配读取 `action_raw`，再恢复绝对目标，不读取旧版本改写后的 `action`，也不使用 replay 中可能来自其他版本的 PCA 结果。这里沿用 0902 的 PCA 文件只是固定输入特征变换，AT 与 LDP 网络均重新训练。

默认按 episode 划分 90% 训练、10% 验证，seed=42。前五段仅用于本次短运行验证；启动上述完整训练命令时没有五段限制。可用 `task.dataset.episode_limit=5` 显式限制调试数据。

LDP 输入为右侧 RGB `camera2`、相对于当前 chunk 基底的右臂 TCP 9D、绝对夹爪宽度 1D、右侧触觉 PCA 15D。RGB 沿用本机缓存的 224×224。AT 编码动作轨迹，解码时逐帧接收触觉条件；输入不再保留无效左侧通道。

## 恢复的训练方法

| 项目 | 基线行为 |
|---|---|
| AT 目标 | 归一化动作普通 L1 重建 + `1e-6 × KL` |
| AT 容量 | 原版 peel：latent channels=4、Conv/GRU hidden=32、posterior channels=16 |
| 后验与 LDP 目标 | 原版 posterior sample；LDP 预测 diffusion epsilon，普通 MSE |
| 动作表示 | 32 帧轨迹，共用最后一帧观测姿态作为相对基底 |
| 归一化 | 位置和夹爪 min/max；rotation 6D 保持 identity scale/offset |
| 采样 | 普通 episode 窗口；边界重复首尾帧 |
| 视觉网络 | 原版 ResNet18 + GroupNorm + 0.9 RandomCrop + ImageNet normalization |
| LDP | 原版 UNet `[512,1024,2048]`、EMA、DDIM 100 步 |
| 权重选择 | 普通 latest / train-loss top-1，无自定义发布门槛 |

新入口不使用 physical_v2 位姿损失、静止/微动加权、静止动作改写、canonical no-op padding、zero_centered_v2、posterior-mode 替代采样或额外光照增强。

保留的本机适配是：单右臂输入、column-6D 旋转格式、encoder0824/PCA 特征、缓存按需读取和 PyTorch 新版本加载兼容。数据末帧缺少下一状态时保持末帧姿态及宽度。grip/touch 归一化统计只使用训练 episode；原版这两类统计覆盖全部 replay。梯度累计、续训计数和末轮保存采用正确的训练状态处理，不改变损失目标。

核心实现位于 `rdp_baseline/`，原始参考目录保持原样。共用的基础 UNet、normalizer、裁剪及 VAE 基础组件沿用现有包，已核对相关计算路径；旧训练入口仍供旧实验复现，新训练请使用本文入口。

## 新权重的动作语义

新权重标记为 `single_right_chunk_relative10d_v1`。同一条输出轨迹内的所有目标都满足：

```text
T_target[t] = T_base @ T_predicted_relative[t]
T_base = 本次 chunk 最后一个观测时刻的 TCP 姿态
```

这是整段共用一个基底的目标轨迹；0902 当前部署采用逐帧相对增量处理。**新权重不能直接交给现有 `deploy_RDP` 的 0902 启动脚本执行**。后续部署适配必须同时处理新的观测字段、32 帧时序和固定基底目标还原。允许 unqualified checkpoint 的参数只会跳过旧资格检查，不能转换动作语义。

## 验证范围

测试覆盖原版 AT loss/gradient 和 LDP sampled-target 数值一致性、固定基底几何、边界采样、归一化和新入口配置。真实数据短测试使用 `insert_01` 前五段，在 CPU 上训练少量 batch，验证保存、加载与 AT→LDP 衔接；LDP 短测试缩小 UNet 和采样步数，仅用于验证代码链路。

全量 500 段数据共 252295 个有效相邻转换，目标重建位置最大误差 `3.28e-8 m`、旋转最大误差 `9.29e-6°`；全部窗口索引与原版一致。结果见 `outputs/rdp_baseline_verification_20260905/data_audit.json`（仓库根目录下）。这只验证数据几何与采样一致性，不代表所有示教都成功或视觉分布一致。

本次没有启动完整训练，也没有用新权重运行机器人。短测试通过不代表抓取成功率或开爪保持误差已经改善。

来源及许可证见 `reactive_diffusion_policy-main/` 与 `rdp_baseline/LICENSE`。
