# Pi0.5 action-origin Flow baseline

本项目复刻 Pi0.5 TacFRS **Flow** 修正网络，以冻结 VLA 输出的归一化动作作为 Flow 的 `t=0` 起点。唯一的起点差异是省略 TacFRS 对 VLA 动作的反向积分；不使用 structured noise，不使用 v3 learned-residual Gate 网络。

网络为独立的 PyTorch 实现，保持本项目与训练／部署主实验运行时隔离。数值对照测试使用相同权重检查其与 `train_pi05_frs/utils/model.py` 的速度预测和 FireFlow 解码结果。

| 参数 | 配置 |
|---|---|
| 动作长度／维度 | 50；右臂 10D，双臂 20D |
| Transformer | 6 层、256D、4 头、FFN 1024D、GELU、Pre-LayerNorm、attention dropout 0 |
| 触觉 | 冻结 0824 ResNet18，224×224，每路原始 512D 特征；不做旧版 RMS 归一化 |
| 触觉时序 | 10 个时间点、history stride 3、共享单层 GRU 512→256 |
| State | 右臂 7D／双臂 20D；当前 normalized state 经 LayerNorm→MLP→一个条件 token |
| State dropout | 训练时整 token 屏蔽率 0.1；验证和部署不屏蔽 |
| 时间／位置 | 正弦编码；time MLP 256→1024→256、SiLU |
| 前向解码 | FireFlow modified midpoint 10 步，11 次速度评估 |

速度网络接口为 `model(x_t, t, tactile_seq, state=state)`；`tactile_seq` 为 `[B,10,S,512]`。`model.decode(vla_action, tactile_seq, state=state)` 从 VLA 动作解码完整 `[B,50,A]` 动作，并保留 VLA 夹爪。触觉路数 `S=2/4`，state 形状为 `[B,7/20]`。

双臂触觉顺序为 `left0,right0,left1,right1`；右臂为 `tactile_left_1,tactile_right_1`。后缀 1 表示右手相机，left/right 表示同一夹爪的两块触觉面。

## 1. 训练

环境入口：

```bash
bash train_baseline_act_inter/scripts/setup_env.sh
```

训练总入口使用项目 `.venv`，依次生成原始触觉特征缓存、VLA 动作／专家动作／normalized state 缓存，然后训练 Flow 网络：

```bash
bash train_baseline_act_inter/scripts/start_train.sh \
  --config train_baseline_act_inter/configs/train_baseline_act_inter_task3.yaml --check

bash train_baseline_act_inter/scripts/start_train.sh \
  --config train_baseline_act_inter/configs/train_baseline_act_inter_task3.yaml
```

先按本机路径修改 YAML 的数据集、Pi0.5 checkpoint、norm assets、0824 encoder 和输出目录。配置提供双臂、task3 insert、task4 press，以及 5080 机器路径版本。`--check` 只检查配置与本地资产，不加载 GPU 模型、不创建输出。

训练数据按 episode 划分；动作与触觉按绝对帧索引对齐。历史按旧到新排列，在 episode 开头重复第一帧，禁止跨 episode。每个样本还提供当前 normalized state 和该 episode 第一帧触觉，后者用于 Gate 标签。终止／补齐动作通过 valid mask 排除。

右臂采用 TacFRS composite endpoint 目标，双臂采用独立腕部 Gate 目标：`tau=0.4/0.65`、temperature `0.1/0.15`，低高区域阈值 `0.3/0.7`。总损失包括线性 Flow Matching、端点 decode（权重 4）、低 Gate safety（0.5，margin 0.03）、高 Gate rank（4）和 repair（2），起点统一替换成 VLA 动作。只更新 GRU、state MLP 和 Flow Transformer，Pi0.5 与 ResNet 均冻结。

`best.pt` 按验证集 decoder Smooth L1 选择；`last.pt` 保存模型、优化器、随机数和批次位置。设置 `decoder.resume: true` 后用相同入口续训，来源和结构不匹配时拒绝恢复。

小规模检查可追加 `--max-samples N --max-steps N`。小规模缓存与正式缓存应使用不同目录；完整触觉缓存按编码器、数据集、预处理和样本选择合同复用，动作缓存支持断点续写。

## 2. 部署

部署位于同级 `deploy_baseline_act_inter/`，提供独立的同构 PyTorch 网络和权重校验：

```bash
bash deploy_baseline_act_inter/scripts/start_baseline_act_inter_single_arm.sh --check
bash deploy_baseline_act_inter/scripts/start_baseline_act_inter_bimanual.sh --check
```

每个 chunk 只调用一次 VLA。每个唯一动作请求读取最新触觉和 state，更新 episode 历史，从同一个 VLA chunk 重新做 FireFlow 解码，选择请求的动作索引。历史跨 chunk 保留；重复请求不重复解码或推进历史。夹爪由 VLA 控制。

交接 `best.pt`、训练 YAML、源模型／归一化／触觉来源信息；两侧网络结构、触觉顺序、state 维度、历史窗口和 FireFlow 参数必须一致。

## 3. 离线评估

离线评估读取已有的新版缓存，不自动生成缓存：

```bash
train_baseline_act_inter/.venv/bin/python -m train_baseline_act_inter.evaluate \
  --config train_baseline_act_inter/configs/train_baseline_act_inter_task3.yaml \
  --checkpoint /path/to/decoder_flow_v2/best.pt \
  --split test --output /tmp/baseline_metrics.json
```

评估使用同一 FireFlow 解码；报告原 VLA／修正动作误差、相对 MSE 改善、修正幅度、物理空间 MAE/RMSE、夹爪误差。可跨 episode 打乱整段触觉历史，保持当前 state 和 VLA 动作不变，检查触觉贡献。支持 JSON／CSV。

## 格式迁移与验证

旧版 2 层 direct decoder 权重、缺少 state 的 action cache、RMS-normalized tactile cache **均不兼容**。新版 checkpoint 和 action cache 使用 version 2，mode 为 `action_origin_flow`；触觉缓存使用 raw-ResNet v2 预处理标识。示例配置已切换到 `_flow_v2` 缓存／输出目录，保留旧产物，需要重新生成缓存并训练。

```bash
python -m pytest train_baseline_act_inter/tests deploy_baseline_act_inter/tests
```

测试覆盖同权重 TacFRS 数值一致性、训练续训、episode 历史和 state 对齐、权重互通、逐请求执行时序。数值对照测试需要 Flax/JAX；没有真实数据、权重和 robot 的 CPU 测试不代表 GPU 训练或闭环成功率。
