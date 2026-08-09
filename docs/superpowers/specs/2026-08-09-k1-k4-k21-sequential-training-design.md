# K1/K4/K21 顺序训练设计

## 目标

在同一对 GPU 上依次训练 VT-SmolVLA 的 K1、K4、K21 三组论文对照；三组使用相同训练预算，任一前置实验失败时停止后续实验。

## 实现

- 扩展 `scripts/start_vtsmolvla_low_token_train.sh`：顺序固定为 K1 → K4 → K21。
- 新增薄入口 `scripts/start_vtsmolvla_k1_k4_k21_train.sh`，只转发参数到上述实现，避免复制 launcher 逻辑。
- K1 使用 repeat factor 1，K4 使用 factor 4，K21 使用 factor 21。
- 将 K21 配置对齐为 `steps: 80000`、`save_freq: 10000`、`scheduler_decay_steps: 80000`；K1/K4 已使用相同设置。
- 三组复用六数据集、离线缓存、触觉缓存、train-only normalization、BF16、batch size、seed 和模型训练模式。

## 运行与失败语义

运行 `bash scripts/start_vtsmolvla_k1_k4_k21_train.sh --gpus 0,1`。外层 launcher 只创建一个 tmux session；内部三次调用现有双卡训练入口并使用 foreground 模式。脚本保持 `set -Eeuo pipefail`，因此 K1 失败不会启动 K4，K4 失败不会启动 K21。

## 验证边界

按用户要求不运行测试套件，只检查 YAML 可解析、三组预算及 repeat factor 正确、Bash 语法有效和 `git diff --check` 通过。
