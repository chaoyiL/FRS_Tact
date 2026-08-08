# SmolVLA 训练包拆分设计

## 目标

把现有 `src/lerobot/policies/smolvla_jax/` 中混合的纯视觉与触觉实现彻底拆分为两个仓库内的顶层 Python 包：

- `train_smolvla`：拥有 SmolVLA JAX 核心，只包含纯视觉训练、验证、checkpoint 和推理能力。
- `train_vtsmolvla`：依赖 `train_smolvla`，只包含触觉配置、编码、融合、缓存和 VT 训练扩展。

旧的 `lerobot.policies.smolvla_jax` 包、旧训练入口和旧训练配置将被删除，不提供兼容转发层。VT、FRS、部署、评估、辅助工具和测试全部切换到新导入路径。

## 选定架构

采用“纯视觉拥有核心，VT 作为扩展层”的方案。不会复制两套模型实现，也不会增加第三个 `smolvla_core` 包。

```text
train_smolvla
    ├── pure-vision model/config/data/checkpoint/trainer
    └── neutral extension interfaces
                    ▲
                    │ depends on
train_vtsmolvla
    └── tactile config/model/data/checkpoint/validation extensions
```

`train_smolvla` 可以在没有 `tactile_encoder` 包和触觉 checkpoint 的情况下导入、显示帮助、加载配置及运行纯视觉测试。其源文件、配置、启动脚本和说明文档不得出现触觉专用配置字段或触觉专用导入。

## 目录结构

```text
train_smolvla/
├── __init__.py
├── architecture.py
├── atomic_checkpoint.py
├── checkpoint.py
├── configuration.py
├── data.py
├── functional.py
├── lora.py
├── modality_dropout.py
├── modeling.py
├── policy.py
├── preprocessing.py
├── rtc.py
├── sharding.py
├── training.py
├── validation.py
├── train.py
├── launcher.py
├── configs/
│   └── train.yaml
├── scripts/
│   └── train.sh
└── README.md

train_vtsmolvla/
├── __init__.py
├── checkpoint.py
├── configuration.py
├── data.py
├── lora.py
├── modeling.py
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

只有触觉确实需要覆盖的模块才放入 `train_vtsmolvla`。VT 模块通过继承、组合或显式组件注入复用纯视觉实现，不复制视觉网络、action expert、优化器、RTC、sharding 或通用 checkpoint I/O。

## 包边界与公共接口

`train_smolvla` 继续公开纯视觉调用方需要的稳定名字：

- `JaxSmolVLAConfig`
- `JaxSmolVLA`
- `JaxSmolVLAPolicy`
- `JaxSmolVLATrainer`
- `LeRobotJaxDataLoader`
- checkpoint、validation 和数据辅助函数

`JaxSmolVLAConfig` 删除 `use_tactile_encoder`、`tactile_encoder_path`、`freeze_tactile_encoder`、`tactile_keys`、`tactile_embedding_dim`、`tactile_num_tokens` 和 `tactile_image_size`。纯视觉模型、预处理器、数据加载器、训练器、LoRA 分区、checkpoint 和 validation 同时删除所有触觉分支。

`train_vtsmolvla` 公开对应的 VT 类型：

- `VTSmolVLAConfig`
- `VTJaxSmolVLA`
- `VTJaxSmolVLAPolicy`
- `VTJaxSmolVLATrainer`
- `VTLeRobotJaxDataLoader`

`VTSmolVLAConfig` 扩展纯视觉配置并拥有全部 tactile 字段。VT 数据加载器包装或扩展纯视觉加载器，加入 tactile key 映射和 embedding cache；VT 模型加入 tactile encoder、projection 和 prefix token；VT checkpoint 与 validation 负责触觉参数初始化及契约检查。

共享 trainer 不再显式读取 `tactile_images`、`tactile_embeddings` 或 `tactile_masks`。训练循环把完整 batch 交给模型定义的 loss/evaluation 接口，由纯视觉模型和 VT 模型分别解释各自输入，确保纯视觉核心不了解触觉字段。

## 配置与一键启动

两个包各自只有一份默认训练配置：

- `train_smolvla/configs/train.yaml`
- `train_vtsmolvla/configs/train.yaml`

训练、模型、数据、验证、日志、checkpoint、缓存和 launcher 参数全部从对应 YAML 读取。Python CLI 仅保留 `--config` 用于显式选择配置文件，不再提供 batch size、steps、output 等参数覆盖，避免参数来源分散。

一键命令分别为：

```bash
bash train_smolvla/scripts/train.sh
bash train_vtsmolvla/scripts/train.sh
```

两个 Shell 脚本只负责定位仓库和调用各自的 launcher，不内嵌训练超参数。launcher 从 YAML 读取 tmux 会话名、输出和运行配置，执行以下预检：

1. 加载项目 `.env.frs`（若存在）并找到 `uv`。
2. 检查 YAML、数据集目录、checkpoint 和 VT encoder/cache 等输入。
3. 检查 JAX GPU。
4. 检查已有 checkpoint 与 resume 配置，防止覆盖。
5. 在可用且未处于 tmux 时创建后台会话，否则前台运行。
6. 为每次训练创建独立的时间戳日志，并同时记录标准输出和标准错误。

纯视觉训练默认使用 tmux 后台会话 `smolvla_train`。日志目录由
`launcher.logs_dir` 配置，默认固定为仓库内的
`train_smolvla/outputs/logs/`，日志文件名为
`train_YYYYMMDD_HHMMSS.log`。仓库通过 `.gitignore` 忽略整个
`train_smolvla/outputs/`，防止日志及同目录下的运行产物被提交。若同名 tmux
会话已经存在，launcher 必须拒绝重复启动并输出 attach 命令。

一键训练不安装系统/Python 环境，不登录外部服务，也不自动下载大文件。现有根级 `scripts/setup_env.sh`、`scripts/download_data.sh` 和 `scripts/download_ckpt.sh` 继续作为训练前的独立准备步骤。

VT launcher 在训练前按 YAML 设置决定是否补齐 tactile embedding cache；纯视觉 launcher 没有该阶段。

## 配置迁移

现有 `configs/train_smolvla_jax.yaml` 搬到 `train_smolvla/configs/train.yaml`，并清除带有 `tactile` 命名的示例 dataset、output、W&B name/tag，保证默认文件表达纯视觉基线。

现有 `configs/train_vtsmolvla_jax.yaml` 搬到 `train_vtsmolvla/configs/train.yaml`，保留 VT 数据、tactile encoder、embedding cache 和模型融合参数。Shell/tmux/runtime 相关可配置值归入 YAML 的 `launcher` 配置块。

现有 `train_for_agent.md` 的有效内容整理进两个包各自的 README，修正 `srcipts` 和不可执行的旧命令。README 明确准备步骤、一键命令、日志位置、tmux 查看方式和 resume 修改位置。

## 下游迁移

- 删除 `src/lerobot/policies/smolvla_jax/`，并从 `src/lerobot/policies/__init__.py` 移除旧的 lazy exports。
- 删除 `tools/train_smolvla_jax.py`、`tools/train_vtsmolvla_jax.py`、根级旧训练 YAML 和旧 VT launcher。
- 纯视觉调用方改为导入 `train_smolvla`。
- VT 调用方改为导入 `train_vtsmolvla`；需要通用视觉数据函数时可以显式导入 `train_smolvla.data`。
- FRS 根据实际职责导入纯视觉核心或 VT 扩展。FRS 不通过旧包名间接获得任何类型。
- FRS 的训练配置和一键启动脚本位于 `train_frs/configs/train_frs.yaml` 与
  `train_frs/scripts/start_frs_train.sh`；FRS 自有阶段通过 `python -m train_frs.<module>`
  运行，SmolVLA merge 和 tactile embedding 预计算仍使用 `tools/` 中的共享工具。
- 部署入口依据 YAML 的模型类型显式选择 `JaxSmolVLAPolicy` 或 `VTJaxSmolVLAPolicy`。
- conversion、merge、publish、inference、evaluation 和 cache 工具切换到新包路径；工具本身不因本次训练目录重构而无关搬家。
- `pyproject.toml` 显式包含 `train_smolvla*` 和 `train_vtsmolvla*`，并确保 YAML、Shell 和 README 作为需要的包数据或仓库资产保留。

## Checkpoint 兼容性

不保留 Python import 路径兼容，但保留现有 checkpoint 文件格式和模型参数 key。纯视觉 checkpoint 不再写 tactile 配置字段；VT checkpoint 仍写现有 tactile 参数 key，以避免已有 VT 权重失效。

读取旧纯视觉 checkpoint 时忽略其 `config.json` 中值为 false/空的旧 tactile 字段。读取 VT checkpoint 必须通过 `train_vtsmolvla`；纯视觉 loader 若看到启用的 tactile 字段，应给出明确错误并提示使用 VT 包。

## 错误处理

- 未知 YAML 顶层字段立即报错。
- 纯视觉 YAML 出现 tactile 字段立即报错，而不是静默忽略。
- VT YAML 缺少 encoder、tactile keys、token 数或 cache root 时在启动训练前报错。
- 旧导入 `lerobot.policies.smolvla_jax` 应失败，以证明没有遗留兼容层。
- launcher 预检失败时不创建训练进程；tmux session 冲突、GPU 不可见、输出目录可能被覆盖时给出可操作的中文错误。

## 测试设计

测试继续位于仓库的 `tests/` 下，并按包边界组织。迁移先用失败测试锁定新接口和旧路径删除，再移动实现。

必须覆盖：

1. `train_smolvla` 可独立导入，且其 import graph 不加载 `tactile_encoder` 或 `train_vtsmolvla`。
2. 纯视觉配置类型和 YAML 拒绝所有 tactile 字段。
3. `train_vtsmolvla` 依赖 `train_smolvla`，VT 配置和 batch 能产生与迁移前相同的触觉融合行为。
4. 新的两个训练入口从任意工作目录执行 `--help`，并能定位自己的默认 YAML。
5. 两个 `scripts/train.sh` 通过 Shell 语法检查，且不包含训练参数常量。
6. 旧 `lerobot.policies.smolvla_jax` 导入失败，旧目录和旧训练入口不存在。
7. 现有训练恢复、LoRA、modality dropout、数据切分、validation、atomic checkpoint 和 data-parallel 测试改用新路径后继续通过。
8. VT tactile cache、tactile integration、FRS safety、发布、部署 launcher 和 inference 相关测试继续通过。
9. checkpoint 参数名及旧 checkpoint 加载行为保持兼容。
10. 全仓库搜索不存在旧包导入或旧训练配置/入口路径。

验证时先运行纯视觉与 VT focused tests，再运行 `tests/jax`、`tests/flow_decoder`、部署相关测试和静态路径扫描。现有用户未提交的 `configs/deploy_smolvla_jax.yaml` 修改及已删除文档不属于本次重构，不覆盖、不恢复，也不把其已知测试差异误判为本次回归。

## 完成标准

- 两个一键命令分别能进入正确训练管线，并且所有可调参数来自各自 YAML。
- `train_smolvla` 中没有触觉专用实现或 import。
- `train_vtsmolvla` 复用而不复制纯视觉核心。
- 旧 `src/lerobot/policies/smolvla_jax/`、旧训练入口和旧训练 YAML 全部删除。
- VT、FRS、部署、评估、工具和测试均使用新包路径。
- focused 回归与适用的全量测试通过；任何预存失败必须与重构无关并单独记录。
