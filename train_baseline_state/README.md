# State-only structured-noise Flow baseline

本目录与 `deploy_baseline_state/` 构成独立 baseline。训练、部署、离线评估分别提供明确入口。Pi0.5 使用视觉、state 和任务指令产生动作，再通过与 TacFRS 相同的冻结 VLA 反向积分得到 structured noise。修正网络以此为起点，只接收当前 normalized state 条件，监督目标为专家动作（GT）。

网络沿用 TacFRS 的 6 层、256D、4 heads、FFN 1024、GELU、Pre-LayerNorm、正弦时间／动作位置编码和 state MLP。移除触觉 encoder、GRU、触觉 token、触觉历史和接触 Gate。训练和部署均不需要触觉数据或权重。

- VLA 原生动作宽度可为 10、20 或 padded 32；反向积分在完整模型宽度上执行，再截取修正器对应的 10／20D。
- 默认 VLA Euler 采样 10 步，SlerpFlow 反向 50 步；这些 prefix-cache／denoise-step／solver 代码复制自 TacFRS。
- `StateConditionedFlowDecoder(x_t, t, state=state)` 预测速度；`decode(structured_noise, state=state, vla_action=coarse)` 以 FireFlow 10 步生成 `[B,50,A]` 动作，夹爪使用 VLA 输出。
- State 为右臂 `[B,7]` 或双臂 `[B,20]`，编码为单个 256D 条件 token；训练整 token dropout 为 0.1。
- 损失为 structured noise→GT 的线性 Flow Matching，加权重 4 的 GT 端点 MSE；terminal／padding 通过 valid mask 排除。此消融没有触觉 Gate 或以其构建的目标。

## 训练

```bash
bash train_baseline_state/scripts/setup_env.sh
bash train_baseline_state/scripts/start_train.sh \
  --config train_baseline_state/configs/train_baseline_state_task3.yaml --check
bash train_baseline_state/scripts/start_train.sh \
  --config train_baseline_state/configs/train_baseline_state_task3.yaml
```

按本机修改 YAML 中 dataset、Pi0.5 checkpoint、norm stats、cache 和输出路径。提供双臂、task3 insert、task4 press 和 5080 路径配置。总入口先独立运行 JAX 动作／反向起点缓存进程，再运行 PyTorch 训练，避免两个训练运行时竞争显存。

缓存包括 `coarse_actions.npy`、`structured_noise.npy`、`states.npy`、`expert_actions.npy`、`valid_masks.npy` 以及 episode／split 元数据。缓存支持不可变来源校验、断点续写和复用；缺少 structured noise 的其他 baseline 缓存会被拒绝。

`--max-samples N --max-steps N` 用于独立目录的小规模验证。设置 `decoder.resume: true` 可从 `last.pt` 恢复优化器、随机数和批次位置；`best.pt` 根据验证集解码误差选择。权重 schema 为 `state_conditioned_flow_decoder`，mode 为 `state_only_structured_noise`，不兼容 tactile／action-origin baseline 权重。

## 部署

客户端入口位于同级部署目录：

```bash
bash deploy_baseline_state/scripts/start_baseline_state_single_arm.sh --check
bash deploy_baseline_state/scripts/start_baseline_state_bimanual.sh --check
```

每个 chunk 进行一次 VLA 采样＋反向积分；每次唯一 steer request 以固定的 chunk structured noise 和最新 state 重新解码，再取 action_index。详见部署 README。

## 离线评估

```bash
train_baseline_state/.venv/bin/python -m train_baseline_state.evaluate \
  --config train_baseline_state/configs/train_baseline_state_task3.yaml \
  --checkpoint /path/to/decoder_state_v1/best.pt \
  --split test --output /tmp/state_baseline_metrics.json
```

评估读取已有缓存和权重，使用同一 structured-noise 起点及 FireFlow 解码。报告原始 VLA／修正网络误差、相对 MSE 改善、修正幅度、物理空间 MAE/RMSE、夹爪误差，支持 JSON／CSV。

## 验证

```bash
python -m pytest train_baseline_state/tests deploy_baseline_state/tests
```

覆盖 state-only 网络及输入接口、TacFRS 数值对照、structured-noise 缓存、训练续训、权重互通、逐请求时序、无触觉观察及 fail-stop。CPU 合成测试不代表真实 GPU 训练与机器人闭环效果。
