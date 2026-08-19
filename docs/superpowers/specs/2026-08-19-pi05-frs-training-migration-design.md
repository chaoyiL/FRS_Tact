# Pi0.5 FRS 完整训练链迁移设计

## 目标

把 `/home/typhon/FRS_Tact-pi05-frs-jax` 中当前可用的 Pi0.5 FRS 完整训练链迁移到
`/home/typhon/FRS_Tact/train_pi05_frs`。迁移后的目录必须可以独立完成：

1. 四路触觉 embedding 预计算；
2. Pi0.5 action cache 生成；
3. FRS decoder 训练、checkpoint 保存、评估与历史曲线输出；
4. 验证生成的 checkpoint 能被 `deploy_pi05` 的 FRS runtime 加载。

这里的“完整”指完整训练闭环，不表示复制整个源仓库。

## 范围

### 必须迁移

- `train_pi05_frs` 的模型、loss、数据加载、checkpoint、训练、评估、可视化和原有测试；
- `configs/train_pi05_frs.yaml` 的 Pi0.5 FRS 配置；
- `scripts/start_frs_pi05_train.sh` 的一键三阶段入口；
- 触觉 embedding、Pi0.5 action cache 和 FRS 训练三个工具入口；
- action cache 生成所需的最小 Pi0.5 模型、变换、归一化、dataset 和 cache 支持；
- 独立的 Python 项目元数据、锁文件和环境安装脚本；
- 训练产物到部署 runtime 的 checkpoint 兼容性测试。

### 明确不迁移

- `deploy_pi05_frs` 或 `deploy_pi05` 客户端副本；
- encoder 训练代码；目标仓库已有 `train_encoder`，训练链只调用其运行时 API；
- `modalities_eval` 整目录；仅把 action cache 实际需要的 Pi0.5 数据转换逻辑放进训练包；
- SmolVLA、SmolVLA FRS 或 VT-SmolVLA 代码；
- 源仓库中的通用开发工具、无关测试、文档计划、缓存、checkpoint、输出目录、`.venv`、
  `__pycache__` 和 `.pyc`；
- 对目标根 `lerobot`、`train_encoder`、`utils` 或 `deploy_pi05` 的覆盖。

## 选择的架构

采用自包含训练项目：

```text
train_pi05_frs/
├── README.md
├── pyproject.toml
├── uv.lock
├── configs/train_pi05_frs.yaml
├── scripts/setup_env.sh
├── scripts/start_frs_pi05_train.sh
├── tools/
│   ├── precompute_tactile_embeddings.py
│   ├── prepare_frs_pi05_cache.py
│   └── train_frs.py
├── train.py
├── evaluate.py
├── plot_history.py
├── utils/
├── pi05_cache/               # cache producer 的 Pi0.5 专用适配层
├── src/lerobot/              # 仅 Pi0.5 FRS 所需的私有模型与 dataset 闭包
└── tests/
```

私有 `src/lerobot` 必须在训练进程的 `PYTHONPATH` 中排在仓库根目录之前，从而避免目标根
SmolVLA `lerobot` 包遮蔽 Pi0.5 模型。`train_encoder` 作为顶层包从目标仓库根目录复用。

## 环境隔离

训练项目使用第三套实际环境：

```text
/home/typhon/FRS_Tact/.venv                    根训练/SmolVLA
/home/typhon/FRS_Tact/deploy_pi05/.venv        Pi0.5 部署
/home/typhon/FRS_Tact/train_pi05_frs/.venv     Pi0.5 FRS 完整训练
```

`train_pi05_frs/scripts/setup_env.sh` 只同步训练项目自己的 `uv.lock`，不修改另外两套环境。
训练启动器只选择 `TRAIN_PI05_FRS_PYTHON` 或本目录 `.venv/bin/python`，不回退到根 `.venv`
或部署 `.venv`。这避免任意一份 lock 卸载或降级另一项目的依赖。

训练环境保持 Pi0.5/OpenPI 所需的 JAX 0.5.3、Flax 0.10.2、Orbax 0.11.13 和
Transformers 4.53.2。源 FRS decoder 当前使用 `nnx.List`，但 Flax 0.10.2 不提供该类型；
迁移时采用已在 `deploy_pi05` 验证过的普通 list 参数树表示，并用参数路径与跨环境加载测试
证明 checkpoint 格式未漂移。这是唯一允许的模型结构兼容修复，不改变层数、张量形状、loss、
solver 或训练超参数。

## 三阶段数据流

### 阶段 1：触觉 embedding

读取 YAML 中的多个 LeRobot v3 数据源，按每个数据源的 `rename_map` 解析四路触觉 key，
加载目标仓库 `train_encoder` 的 frozen ResNet checkpoint，并写入可恢复的逐帧 embedding cache。
已有完整且 provenance 一致的 cache 必须跳过；不一致时必须明确报错，不允许静默覆盖。

### 阶段 2：Pi0.5 action cache

一次加载 YAML 指定的 Pi0.5 checkpoint，对每个数据源构造与训练时一致的两路视觉、20/32D
动作映射、prompt、norm stats 和 reverse-flow 设置。输出沿用源 `utils.cache` 的 manifest 和数组
语义，但实现放入训练项目自己的适配层，不能覆盖目标根 `utils/cache.py`。

远程 checkpoint 继续支持 OpenPI cache；本地 checkpoint 必须包含 `params/`。配置中的
`paligemma_variant`、`action_expert_variant`、action dimension/horizon、camera map 和 norm stats
必须在长任务开始前校验。

### 阶段 3：FRS decoder

消费 action cache 与 tactile embedding cache，保持源训练的 gated loss、state conditioning、
history stride、decoder solver、resume、best-checkpoint 约束和可视化行为。输出格式必须兼容
`deploy_pi05/frs_inference`，包括 `decoder_input_version`、参数路径、metadata 和 NPZ 数组。

三个阶段由一个前台或 tmux 一键脚本顺序执行；任一阶段失败立即停止，日志写入配置的输出目录。

## 配置与入口

- 默认配置移动为 `train_pi05_frs/configs/train_pi05_frs.yaml`；
- 默认启动为：

  ```bash
  cd /home/typhon/FRS_Tact
  bash train_pi05_frs/scripts/setup_env.sh
  bash train_pi05_frs/scripts/start_frs_pi05_train.sh
  ```

- 启动器接受一个可选 YAML 路径；所有相对路径都相对配置文件或仓库根目录进行明确解析；
- 提供 `--check`，只验证 Python、依赖、配置、checkpoint、encoder、数据集、输出目录和 GPU
  前置条件，不生成 cache、不加载完整模型、不启动训练；
- 原配置中的 `/workspace` 示例路径保留为模板，但 README 必须提醒部署机按真实路径修改。

## 复用边界

- 复用 `train_encoder` 的 checkpoint、图像预处理、ResNet 编码接口；若目标 loader 不支持源
  content-hashed 文件名，仅在 Pi0.5 FRS 适配层补兼容查找，不回改 encoder 训练包；
- 复用目标根 `utils.cache` 只能通过已经一致的公开读取语义。action cache producer 缺失的
  record-selection/provenance 能力放在 `train_pi05_frs.pi05_cache`；
- 不从 `deploy_pi05` import 私有实现。训练与部署只通过 checkpoint wire format 集成；
- 不改目标根 `pyproject.toml` 或 `uv.lock`。

## 错误处理与安全

- 环境目录与另外两套 `.venv` 相同时，在任何 `uv sync` 前拒绝；
- 缺少 GPU、dataset v3 metadata、encoder、norm stats、camera map 或 checkpoint params 时早失败；
- cache manifest 与当前 checkpoint/dataset/config 不一致时拒绝续写；
- resume checkpoint 的 decoder config 不一致时拒绝恢复；
- cache 和 checkpoint 使用原子的 metadata/array 写入策略；
- 不自动删除旧 cache、checkpoint 或训练输出。

## 验证

迁移必须提供以下证据：

1. 源 `train_pi05_frs` 17 个跟踪 Python 文件都有明确的迁移映射，且没有缓存文件；
2. shell 语法、`setup_env.sh --check`、训练入口 `--check` 通过；
3. 独立 `uv lock --check --offline` 与 `uv sync --frozen --dry-run` 通过；
4. 原 `test_data.py`、`test_model.py` 及新增配置、launcher、package-boundary 测试通过；
5. 小模型完成 forward、loss、一步 optimizer、保存和恢复；
6. 使用训练环境写 checkpoint，再用部署环境加载，参数路径、张量值与 forward 输出一致；
7. mock 小数据端到端验证三阶段顺序、失败即停、resume/skip 行为；
8. 对源/目标进行边界扫描，确认未复制 encoder 训练、modalities analysis、部署客户端和
   SmolVLA 代码；
9. 若现场 GPU、真实 dataset 或 checkpoint 不可用，必须明确记录未执行的真实长任务，不能以
   mock 测试声称完成真实训练。

## 完成标准

- `/home/typhon/FRS_Tact/train_pi05_frs` 是唯一新增的 Pi0.5 FRS 运行代码项目；仓库外层只允许
  增加迁移设计/计划与边界回归测试；
- 用户能通过目录内 setup 与一键脚本运行完整三阶段链路；
- 训练产物可直接供 `/home/typhon/FRS_Tact/deploy_pi05` 使用；
- 目标现有 root、SmolVLA、encoder、modalities 和 deployment 行为不发生非必要改变；
- 所有可运行的自动验证通过，并清楚报告真实硬件/数据长任务的验证限制。
