# FRS 训练包彻底迁移设计

## 目标

把当前散落在仓库根目录、`tools/`、`scripts/`、`configs/` 和
`tactile_flow_steering/` 中的 FRS 专属训练代码集中到顶层
`train_frs/` 包。迁移只改变文件归属路径和 Python 导入路径，不改变原文件名、
训练算法、配置字段、checkpoint 格式、cache 格式或默认训练行为。

旧 FRS 路径在迁移完成后直接删除，不保留兼容转发层。仓库内所有 Python、Shell、
YAML、测试和有效文档调用方必须同步切换到新路径。

## 选定方案

采用“按领域彻底迁移”方案：FRS 专属实现全部归 `train_frs` 所有；同时被 VT、
SmolVLA、tactile encoder 或其他工具使用的模块保持共享位置，不复制到
`train_frs`。这避免让非 FRS 代码反向依赖 FRS，也避免同一实现出现两份源代码。

不采用以下方案：

- 不把所有传递依赖复制到 `train_frs`，因为会造成 VT/FRS 双源维护。
- 不新建额外公共基础包，因为这会把本次路径迁移扩大为无关的基础设施重构。
- 不保留旧路径转发层，因为用户已明确选择彻底迁移。

## 目标目录结构

所有现有文件保留 basename，现有 `utils/` 子目录结构也保持不变：

```text
train_frs/
├── __init__.py
├── train_frs.py
├── train.py
├── prepare.py
├── prepare_frs_caches.py
├── compare_frs_reverse_solvers.py
├── evaluate.py
├── plot_history.py
├── configs/
│   └── train_frs.yaml
├── scripts/
│   └── start_frs_train.sh
├── utils/
│   ├── __init__.py
│   ├── checkpoint.py
│   ├── data.py
│   ├── history_plot.py
│   ├── integration.py
│   ├── metrics.py
│   ├── model.py
│   ├── mp_batches.py
│   ├── visualize.py
│   └── window_io.py
└── README.md
```

## 文件迁移映射

| 旧路径 | 新路径 |
| --- | --- |
| `tools/train_frs.py` | `train_frs/train_frs.py` |
| `tactile_flow_steering/train.py` | `train_frs/train.py` |
| `prepare.py` | `train_frs/prepare.py` |
| `tools/prepare_frs_caches.py` | `train_frs/prepare_frs_caches.py` |
| `tools/compare_frs_reverse_solvers.py` | `train_frs/compare_frs_reverse_solvers.py` |
| `tactile_flow_steering/evaluate.py` | `train_frs/evaluate.py` |
| `tactile_flow_steering/plot_history.py` | `train_frs/plot_history.py` |
| `tactile_flow_steering/utils/*` | `train_frs/utils/*` |
| `configs/train_frs.yaml` | `train_frs/configs/train_frs.yaml` |
| `scripts/start_frs_train.sh` | `train_frs/scripts/start_frs_train.sh` |

`tactile_flow_steering/__init__.py` 的包说明整理进新的
`train_frs/__init__.py`。原包测试迁入仓库统一测试目录，而不是放进生产包。

## 保持共享位置的代码

以下代码是 FRS 管线依赖，但存在非 FRS 消费者，因此不搬入 `train_frs`：

- `tools/precompute_tactile_embeddings.py`：同时被 VT 训练调用。
- `tools/merge_smolvla_peft_to_jax.py`：通用 SmolVLA checkpoint 工具。
- `tactile_encoder/`：独立 encoder 包及训练代码。
- `src/lerobot/`：数据集和 SmolVLA 底层实现。
- `modalities_eval/`：SmolVLA 加载及评估公共支持。
- `utils/cache.py`：仍被 tactile encoder、cache 工具和旧通用 flow 代码使用。
- `utils/source_model.py` 与 `utils/integration.py`：action cache 生成和其他 flow
  工具共享。
- `scripts/setup_env.sh`、`scripts/download_data.sh` 和
  `scripts/download_ckpt.sh`：仓库级环境与资源准备脚本。

FRS 新包通过清晰的绝对导入使用这些共享模块。此次不复制、重命名或删除它们。

## 入口和配置

正式训练入口变为：

```bash
uv run --no-sync python -m train_frs.train_frs \
  --config train_frs/configs/train_frs.yaml
```

一键完整管线变为：

```bash
bash train_frs/scripts/start_frs_train.sh
```

Shell 脚本必须能从任意当前工作目录定位仓库根目录和默认 YAML。完整管线顺序保持：

1. 合并或检查 SmolVLA PEFT checkpoint。
2. 按配置执行反向求解器 A/B 检查。
3. 预计算或补齐 tactile embeddings。
4. 生成或补齐 action caches。
5. 启动 multi-dataset FRS 训练。

共享的 merge 和 embedding 工具仍从 `tools/` 调用；已迁移的 FRS 工具通过
`python -m train_frs.<module>` 调用。配置字段、默认数值和外部资源路径保持不变。

## 导入路径和调用方迁移

生产代码的旧导入按以下规则替换：

- `tactile_flow_steering.train` → `train_frs.train`
- `tactile_flow_steering.utils.<module>` → `train_frs.utils.<module>`
- `tools.train_frs` → `train_frs.train_frs`
- `tools.prepare_frs_caches` → `train_frs.prepare_frs_caches`
- `tools.compare_frs_reverse_solvers` → `train_frs.compare_frs_reverse_solvers`
- 根级 `prepare` → `train_frs.prepare`

必须扫描并更新：

- 所有静态 `import` 和动态 `importlib.import_module`。
- `scripts/start_vtsmolvla_train.sh` 对共享 tactile embedding 工具的调用；该调用继续
  指向 `tools/precompute_tactile_embeddings.py`，不得错误改成依赖 FRS。
- `tests/flow_decoder/`、原 `tactile_flow_steering/tests/` 和其他引用旧路径的测试。
- YAML 注释、Shell 命令、README 以及仍代表当前行为的设计/计划文档。
- `pyproject.toml` 的 setuptools package include，显式包含 `train_frs*`。

历史文档中用于记录旧架构的文字可以保留，但当前可执行命令和当前路径描述不得继续
指向已删除入口。

## 测试布局

原 `tactile_flow_steering/tests/` 移到 `tests/train_frs/`，保持测试文件名：

- `test_data.py`
- `test_model.py`

`tests/flow_decoder/` 中仍验证当前 FRS/cache 行为的测试更新为新 import；与旧通用
非 tactile flow decoder 实现绑定的测试不因本次迁移改写业务含义。

新增路径契约测试应先失败再实施迁移，并至少覆盖：

1. `train_frs`、`train_frs.train_frs`、`train_frs.train` 和核心 utils 可导入。
2. `python -m train_frs.train_frs --help` 从仓库外当前目录运行时成功。
3. 新默认配置定位为 `train_frs/configs/train_frs.yaml`。
4. `train_frs/scripts/start_frs_train.sh` 通过 `bash -n`。
5. 旧 `tactile_flow_steering`、`tools.train_frs` 和根级 `prepare` 导入失败。
6. 旧文件路径不存在。
7. 全仓库当前代码不存在旧 import、旧命令或旧默认配置路径。

## 错误处理与兼容性

- 迁移不得改变 cache manifest、tactile embedding metadata 或 checkpoint 的文件格式。
- resume 的 `last/checkpoint.json` 探测及 optimizer 恢复行为保持不变。
- 默认配置、dataset source、rename map、loss、优化器和输出目录字段保持不变。
- 新入口遇到缺失 cache、encoder、dataset 或 checkpoint 时保持现有错误语义。
- 不提供旧 Python import 或旧文件路径的兼容层；旧路径残留视为迁移失败。

## 非目标和生成资源

本次不移动或提交数据集、merged SmolVLA checkpoint、tactile encoder checkpoint、
action cache、embedding cache、训练输出、JAX compilation cache、pipeline 日志、
`.env.frs`、`.venv` 或 Python cache。

本次也不重构训练算法、不调整超参数、不重新设计 FRS 模型，并且不执行已另行规划的
SmolVLA/VT 包拆分。后续 SmolVLA/VT 拆分实施时，应直接更新 `train_frs` 的共享依赖
导入，而不是恢复旧 FRS 路径。

## 完成标准

- 上述 FRS 专属文件全部位于 `train_frs/`，basename 保持不变。
- 旧 FRS 包、入口、配置和脚本全部删除且无转发层。
- 全仓 Python、Shell、YAML、测试和当前文档调用方使用新路径。
- 共享 VT/SmolVLA/encoder 代码没有被复制进 `train_frs`，非 FRS 调用方不反向依赖
  `train_frs`。
- focused FRS/cache 测试、CLI help、Shell 语法检查及适用的全仓回归通过。
- 最终 `rg` 路径扫描没有发现当前代码中的旧 FRS import 或命令；预存且与迁移无关的
  工作区修改保持不变。
