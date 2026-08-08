# VT-SmolVLA 训练包构建设计

## 目标

参照 `train_frs/` 的自包含训练包形式，把 VT-SmolVLA 的训练入口、默认
YAML、一键启动器和运行说明集中到 `train_vtsmolvla/`。迁移完成后，删除
旧 VT 训练入口、根级配置和根级启动脚本，不保留兼容转发层。

本次只重构运行与打包边界，不改变 VT 模型结构、训练超参、checkpoint
参数 key、tactile embedding cache 格式或 resume 语义。已有未提交的
`train_vtsmolvla/data.py`、`train_vtsmolvla/lora.py`、`train_smolvla/lora.py`
及对应测试是受保护的并行改动，不得覆盖或回退。

## 选定方案

采用“视觉包拥有中性训练编排，VT 包注入扩展组件”的方案。

- `train_smolvla.train` 拥有数据切分、模型初始化、训练循环、验证调度、
  checkpoint 保存和 W&B 日志等中性编排。
- `train_vtsmolvla.train` 负责校验 VT YAML，并向共享编排显式提供
  `VTSmolVLAConfig`、`VTJaxSmolVLA`、`VTLeRobotJaxDataLoader`、
  `VTJaxSmolVLATrainer` 以及 VT checkpoint/validation/LoRA 函数。
- `train_vtsmolvla` 继续单向依赖 `train_smolvla`，不复制视觉网络、action
  expert、优化器、RTC、sharding 或完整训练循环。

不采用以下方案：

- 不保留“VT wrapper 修改 `sys.argv` 后导入旧 `tools/train_smolvla_jax.py`”的
  隐式耦合。该脚本仍硬编码旧混合类型，不能证明新 VT 包真正接管运行时。
- 不把旧通用训练脚本复制进 VT 包。两套训练循环会使数据切分、
  validation、checkpoint 和 resume 逻辑逐渐漂移。

## 目标目录与资源

```text
train_vtsmolvla/
├── __init__.py
├── checkpoint.py
├── configuration.py
├── data.py
├── lora.py
├── modeling.py
├── policy.py
├── preprocessing.py
├── tactile_cache.py
├── training.py
├── validation.py
├── train.py
├── launcher.py
├── configs/
│   └── train.yaml
├── scripts/
│   └── train.sh
└── README.md
```

迁移映射：

| 旧路径 | 新路径 |
| --- | --- |
| `tools/train_vtsmolvla_jax.py` | `train_vtsmolvla/train.py` |
| `configs/train_vtsmolvla_jax.yaml` | `train_vtsmolvla/configs/train.yaml` |
| `scripts/start_vtsmolvla_train.sh` | `train_vtsmolvla/launcher.py` + `train_vtsmolvla/scripts/train.sh` |

`tools/precompute_tactile_embeddings.py` 是 VT 和 FRS 共享的工具，保持在 `tools/`。
`pyproject.toml` 必须把 VT 的 YAML、Shell 和 README 声明为 package data，确保
wheel 安装后仍可使用默认配置和模块 CLI。

## 共享训练编排接口

`train_smolvla.train` 使用一个聚合的运行时组件对象，而不是在通用逻辑中分散
`if tactile` 分支。组件对象要显式提供：

- config、model、data loader 和 trainer 类型；
- checkpoint 解析、参数加载与必要的参数初始化函数；
- module mode / LoRA 解析函数；
- checkpoint contract 构建和 validation 函数。

共享入口对上述组件只依赖明确的可调用接口。纯视觉 CLI 使用默认视觉
组件；VT CLI 创建 VT 组件对象并调用同一编排函数。这个接口只支持当前
的视觉和 VT 两个后端，不扩展为通用插件系统。

`train_smolvla.train` 的中性编排是 VT 入口的前置依赖。若它尚未按已有
SmolVLA 一键训练计划实现，本次实施先完成该共享编排，再接入 VT；
不会为了绕过该前置依赖而复制训练循环。

## 配置与入口

VT 直接训练入口为：

```bash
uv run --no-sync python -m train_vtsmolvla.train \
  --config train_vtsmolvla/configs/train.yaml
```

一键入口为：

```bash
bash train_vtsmolvla/scripts/train.sh
```

Python CLI 只暴露 `--config`。batch size、steps、output、resume、dataset、
checkpoint、cache、validation、W&B 和 launcher 参数都来自同一份 YAML。

`train_vtsmolvla/configs/train.yaml` 保留现有 VT 数据集、模型、触觉
encoder、embedding cache 和训练超参，只做路径迁移和 launcher 配置归位。
新增顶层 `launcher` mapping：

```yaml
launcher:
  tmux_session: vtsmolvla_train
  foreground: false
  logs_dir: train_vtsmolvla/outputs/logs
```

`train_vtsmolvla/scripts/train.sh` 只定位项目根和 `uv`，然后执行
`python -m train_vtsmolvla.launcher --config ...`，不解析或硬编码任何训练超参。

## Launcher 数据流

```text
train_vtsmolvla/configs/train.yaml
  → 加载 .env.frs（若存在）并定位 uv
  → 检查 dataset/checkpoint/tactile encoder/cache/output/resume
  → 检查 JAX GPU
  → 根据 launcher.foreground 和当前 tmux 状态选择前台或后台
  → 若 tactile_embedding_cache.enabled，运行共享预计算工具补齐 cache
  → python -m train_vtsmolvla.train --config ...
```

每次运行生成独立日志：

- `precompute_YYYYMMDD_HHMMSS.log`：本次 cache 检查/补齐输出；
- `train_YYYYMMDD_HHMMSS.log`：正式训练输出。

日志同时接收 stdout 和 stderr。仓库忽略整个 `train_vtsmolvla/outputs/`。已存在
同名 tmux session 时拒绝替换，并输出 `tmux attach -t vtsmolvla_train`。

## 校验与错误处理

所有可预见的输入错误在创建 tmux 或训练子进程前失败：

- YAML 根节点必须是 mapping，未知顶层字段立即报错。
- `model.use_tactile_encoder` 必须为 true。encoder path、tactile keys、embedding
  dimension 和 token count 必须完整。
- `tactile_keys` 数量必须等于 `tactile_num_tokens`，且不得与
  `image_keys` 重叠。
- cache 启用时必须配置 root；预计算失败时不进入训练。
- 本地 dataset、checkpoint 或 encoder 路径已显式配置但不存在时报错；
  允许下载的 Hub ID 不按本地路径误判。
- JAX 未发现 GPU 时拒绝正式启动。
- output 已有 checkpoint 但 resume 为空时拒绝覆盖。
- tmux session 冲突时拒绝重复启动。

错误信息保留中文操作建议。共享训练编排的异常和子进程非零退出码原样
向上传播，不伪装成成功。

## 兼容性与边界

- 保留现有 VT checkpoint 文件布局、config 字段和参数 key。
- 保留 tactile cache metadata 中的 `num_tactile_tokens`，不与模型配置字段
  `tactile_num_tokens` 误合并。
- `tools/precompute_tactile_embeddings.py` 保持共享，FRS 不反向依赖 VT launcher。
- `train_smolvla` 不导入 `train_vtsmolvla` 或 `tactile_encoder`；VT 组件只由
  VT 入口注入。
- 对 `freeze_tactile_encoder=False` 不新增训练语义。当前 tactile encoder 保持
  冻结；若 YAML 请求不支持的解冻行为，入口应明确拒绝，而不静默忽略。

## 测试设计

实施遵循 TDD，先用失败测试锁定新契约：

1. `train_vtsmolvla.train` 和 `train_vtsmolvla.launcher` 可导入，CLI 只暴露
   `--config`。
2. 默认 YAML、Shell 和 README 均位于 VT 包，旧 VT 入口/配置/脚本不存在。
3. VT 入口向共享编排传入 VT Config/Model/DataLoader/Trainer/checkpoint/validation
   组件，不回退到纯视觉类型。
4. launcher 的 YAML 解析、时间戳日志、前台/后台、tmux 冲突、resume
   防覆盖和 cache 先行顺序都由单元测试覆盖。
5. wheel 包含 `train_vtsmolvla/README.md`、`configs/train.yaml` 和
   `scripts/train.sh`，解包后在仓库外执行模块 `--help` 成功。
6. 现有 VT tactile cache、integration、LoRA、checkpoint、validation、FRS 共享工具和
   部署路径回归不变。
7. Shell 通过 `bash -n`，静态扫描不存在旧 VT 训练路径或旧
   `lerobot.policies.smolvla_jax` 运行时导入。

验证按“新单元测试 → VT focused tests → 相关 JAX/FRS/部署/工具回归 →
静态路径扫描”的顺序进行。当前工作区中已知的部署默认 checkpoint 断言差异
单独记录，不得误认为本次迁移造成，也不得为此回退用户配置。

## README 内容

`train_vtsmolvla/README.md` 至少说明：

- 先运行仓库级环境、数据和 checkpoint 准备步骤；
- 一键命令和直接训练命令；
- 所有参数在 `configs/train.yaml` 修改；
- tactile cache 会在启用时自动检查和补齐；
- tmux attach 命令和时间戳日志位置；
- resume 字段与防覆盖行为。

## 完成标准

- `bash train_vtsmolvla/scripts/train.sh` 进入 VT 而不是纯视觉训练管线。
- `python -m train_vtsmolvla.train --help` 在仓库外当前目录和 wheel 安装环境中
  都可执行。
- VT 训练入口使用共享中性编排和 VT 组件，不复制训练循环。
- 旧 VT 训练入口、根级 YAML 和根级 Shell 已删除，当前调用方全部使用新路径。
- checkpoint、cache、resume 和默认 VT 训练参数保持兼容。
- VT focused tests、相关回归、Shell 检查、wheel 资源检查和静态路径扫描通过；
  任何预存失败均有独立证据。
