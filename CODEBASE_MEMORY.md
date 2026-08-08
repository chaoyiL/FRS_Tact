# FRS_Tact 工作记忆

## modalities_eval 触觉模态分析（2026-08-08）

- 当前实现分支：`codex/modalities-eval-tactile`，隔离 worktree 为 `/home/yunjing/FRS/.worktrees/FRS_Tact-modalities-eval`。
- 原始 `eric` 工作区存在用户未提交的环境改动，当前任务不得覆盖或清理。
- 独立六路 v3 数据目标：`/home/yunjing/FRS/eval_data/KaiyueChen/pick_tube_02`；原始 v2.1 HF 缓存只读保留。
- VT checkpoint：`/home/yunjing/FRS/KaiyueChen/vtsmolvla_01_3w`，chunk 20，四路独立 tactile tensor。
- 视觉 checkpoint：`/home/yunjing/FRS/FRS_Tact/checkpoints/pick_tube_02_3w_jax`，chunk 10，无 tactile 分支。
- 两者不是只改变 tactile 的受控 A/B：expert/action/state 权重、normalizer、chunk 和训练协议均不同。因此 VT 内部 full/no-tactile 是主要触觉证据；跨模型只能称系统级比较。
- 两模型输入必须显式使用相同 RGB rename：`camera0 -> camera1`、`camera1 -> camera2`。
- 运行顺序固定为：VT 单帧 tactile action-error -> 两模型多帧 action-error -> likelihood/Hutchinson -> 辅助 t-SNE。
- 分别拟合的 t-SNE 坐标不可直接比较；跨模型可视化若需要比较，必须联合拟合。

后续每次代码、数据或实验产物发生变化，都在本文件追加命令、结果和限制。

### 当前执行记录

- 已创建实现方案：`docs/plans/2026-08-08-modalities-eval-tactile.md`。
- 已将 SmolVLM tokenizer 的 5 个必要文件下载到 `/home/yunjing/FRS/eval_cache/huggingface/hub`；设置 `HF_HUB_CACHE` 后，`AutoTokenizer(..., local_files_only=True)` 验证成功，类型为 `GPT2TokenizerFast`、词表大小 49152。
- 数据源 v2.1 大小约 4.3 GiB；已复制到工作区目标，源缓存未改。副本原始全局 index 不唯一（61,811 帧仅 52,802 个唯一 index），转换前需在副本修复为 `0..61810`。
- 当前受控命令中 `nvidia-smi` 可见空闲 RTX 4090，但 JAX 0.8.3 和 PyTorch 2.11.0 都检测不到 CUDA，正式模型评估暂不能在此进程中启动；不得把 likelihood/Hutchinson 误跑到 CPU。
- 六路 v3 数据已生成在 `/home/yunjing/FRS/eval_data/KaiyueChen/pick_tube_02`：`codebase_version=v3.0`，150 episodes、61,811 frames、50 个 parquet；两路 RGB 与四路 tactile 均可解码为 `[3,224,224]`，state/action 均为 20 维，action 字段仍名为 `actions`。
- 转换后的全局 index 为 `0..61810` 且 61,811 个全部唯一；episode 内 frame index 连续。当前 loader 已抽样读取首/中/尾帧及 episode 0 的 delta action chunk。
- 源 cache 的 inventory SHA256 复核仍为 `1b8419298fb5c7d819f0f19e795ad97302bc873dac1a330bcefd4dead0120719`，源未改。
- 转换过程的两个失败中间目录被移动并保留：`pick_tube_02_v30.failed-20260808T013055`、`pick_tube_02_v30.failed-20260808T013249-cachelock`；它们不是最终数据，不应用于评估。
- TDD 红灯证据：新增 6 个聚焦测试后，旧 evaluator 为 `6 failed in 3.22s`，失败覆盖 tactile/action padding 丢失、state mask 缺失、padding/horizon/physical metrics 接口缺失。

### 最终代码状态与验证

- `EvalObservation` 已独立保存 `tactile_images`、`tactile_embeddings`、`tactile_masks`、`tactile_keys` 与 `state_mask`；sampling 和 velocity/likelihood context 全量透传。
- tactile 消融现在只把 `tactile_masks` 置 False；纯视觉 checkpoint 的 tactile 消融明确报不适用。state 消融使用 `state_mask=False`。
- `EpisodeData` 保留 `action_is_pad`；action-error 按有效 step 聚合，并支持 `--evaluation-horizon` 与 checkpoint 反归一化后的 physical MSE/RMSE/MAE。
- action-error 和 likelihood 支持显式 `--frames ...`；`--frames` 与 `--sample-interval` 互斥。
- likelihood、plot wrapper 和 reverse t-SNE 对任何 padded action chunk 都在计算前 fail-fast，提示选择 H_safe 完整帧。
- `JaxSmolVLA.sample_actions` 只新增向后兼容的可选 `state_mask=None` 参数，默认模型行为不变。
- TDD 红灯还覆盖：`tactile_keys=None`、raw tactile images、`--frames` CLI、likelihood/plot padding 旁路。
- 最终 fresh 回归：`103 passed, 1 skipped in 9.08s`；`py_compile` 与 `git diff --check` exit 0。只读代码审查最终 verdict：无 Critical/Important，Ready。

### 实际运行与产物

- 沙箱外 GPU 复验：JAX backend `gpu`、device `CudaDevice(id=0)`；PyTorch CUDA 可用。
- 固定 episode 48，frames `0,50,100,150,200,249,300,350`，所有 H=20 chunk 无 padding；action-error 使用共同 H=10、k=10、seed=0。
- VT tactile action-error：normalized delta MSE mean `0.057198`，5/8 为正；physical delta MSE mean `1.1261e-6`。
- VT action-error 的 vision/state/language normalized delta mean 分别为 `0.460663 / 0.567575 / 0.429842`。
- Visual action-error 的 vision/state/language normalized delta mean 分别为 `0.510826 / 0.585254 / 0.431766`；tactile 为 N/A。
- likelihood 使用 Euler k=20、Hutchinson samples=1/seed=0。VT tactile contribution mean `45.121`、median `5.712`、5/8 为正，范围 `-38.382..227.433`，阶段依赖明显。
- likelihood contribution mean：VT vision/state/language `421.074 / 136.951 / 146.786`；Visual vision/state/language `360.088 / 383.639 / -2.079`。
- 全部结果位于 `/home/yunjing/FRS/eval_outputs/modalities_eval_20260808`；汇总为该目录 `RESULTS_SUMMARY.md`。
- 输出 fresh 检查：15 个主 metric CSV 共 113 rows 全 finite；另有单帧 action-error smoke CSV；18 个 PNG；VT/Visual t-SNE NPZ shape 分别为 `[8,20,20]`、`[8,10,20]`，坐标 finite。
- 两个 t-SNE 分别拟合，仅作为各模型内部辅助图，禁止跨图比较坐标。
- 结果仅来自一个 episode 的 8 帧且无共同 holdout 证明；两个 checkpoint 也不是 tactile-only A/B，不得作强因果或泛化结论。

## Naive tactile token ratio baseline 设计（2026-08-08）

- 用户批准按 conditioning prefix 计算 tactile token 占比，不包含 action/time suffix。
- 方法固定为投影后无参数 `repeat_interleave`，继续直接 concat：`RGB -> tactile repeats -> language -> state`。
- 新配置字段定为 `tactile_token_repeat_factor`，默认 1；旧 `tactile_num_tokens=4` 继续表示四路 tactile keys/cache streams，不改变 cache `[F,4,512]`。
- K=8：32 tactile tokens，最大 prefix 209，占 15.31%，论文记作约16%。
- K=21：84 tactile tokens，最大 prefix 261，占 32.18%，论文记作约32%。
- 不新增参数、modality/type/copy-slot embedding 或位置编码；沿用连续 RoPE。
- 旧 config/checkpoint 缺字段时按 K=1；不同 K 禁止 strict optimizer resume，只能作权重 warm-start的新 run。
- 正式实验 K=1/8/21 必须从相同 base initialization、数据 split、seed 和训练协议开始。
- 书面 spec：`docs/designs/2026-08-08-naive-tactile-token-ratio-design.md`。
- 实施计划：`docs/plans/2026-08-08-naive-tactile-token-ratio.md`；已覆盖 config/checkpoint 持久化、launcher fail-closed 校验、投影后 token/mask 同序复制、旧 resume metadata K=1 迁移、K=1/8/21 YAML 科学变量一致性及 CPU/GPU 验证。
- spec 和 plan 已按后文“最终实现与整体验证”落地；token baseline 生产代码已实施且通过 CPU/GPU smoke，但尚未启动新训练。

### Naive tactile token ratio 实施进度：Task 3

- `modeling.py` 新增纯函数 `_repeat_tactile_tokens_and_masks`：共享 `model.tactile_proj` 投影后，沿 token 轴做无参数 key-major `jnp.repeat`，mask 同序复制。
- `embed_tactile` 仍输出基础 `[B,S,H]`；`embed_prefix` 先校验基础 `[B,S]` mask，再按 `tactile_token_repeat_factor` 扩展并拼接。cache shape、参数、位置编码与 suffix 均未改变。
- TDD 证据：helper 缺失时 import RED；真实 prefix RED 为实际 `(1,12,3)` 对期望 `(1,40,3)`；实现后聚焦 CPU 测试 `12 passed in 3.54s`。
- K=1/8/21 分别验证 4/32/84 tokens，且 key-major value 顺序、mask parity、K=1 value-preserving 全部覆盖。

### Naive tactile token ratio 最终实现与整体验证（2026-08-08）

- 实现仍位于分支 `codex/modalities-eval-tactile`、worktree `/home/yunjing/FRS/.worktrees/FRS_Tact-modalities-eval`，保持未暂存、未提交；既有 `modalities_eval` 工作没有被 token-baseline tasks 覆盖或清理。
- 最终生产/config 范围：`configuration.py` 新增严格正整数 `tactile_token_repeat_factor`（legacy default K=1）和 `effective_tactile_num_tokens`；`checkpoint.py` 保存该字段；`modeling.py` 在共享 `tactile_proj` 之后按 key-major 顺序无参数复制 token 和 mask；`training.py` 只把旧 resume metadata 缺失字段迁移为 K=1；VT launcher 在 checkpoint/data 访问前 fail-closed 校验；三份 YAML 为 K=1/8/21 独立完整配置。
- 参数/cache 契约不变：`tactile_num_tokens=4` 仍表示四路源 tactile streams，cache 输入仍为 `[F,4,512]`，`model.tactile_proj.weight` 的现有 checkpoint shape 实测仍为 `(960, 512)`；未新增 parameter key、projection、cache schema、modality/type/copy-slot embedding 或位置编码逻辑。
- prefix 口径固定为 `128 RGB + 4K tactile + 48 language + 1 state`（不含 action/time suffix）：K=1 为 4/181=`2.21%`，K=8 为 32/209=`15.31%`（论文约 16%），K=21 为 84/261=`32.18%`（论文约 32%）。prefix 顺序仍为 `RGB -> repeated tactile -> language -> state`。
- 分任务 RED/GREEN：Task 1 config RED `3 failed, 8 passed`，review-fix direct/replace RED `10 failed, 12 passed`，最终 `22 passed`；Task 2 launcher RED `5 failed, 1 passed`，GREEN `6 passed`；Task 3 helper RED 为 collection `ImportError`，prefix RED `1 failed, 11 passed`，GREEN `12 passed`；Task 4 初版 legacy resume RED 为缺少 `model.tactile_token_repeat_factor`、GREEN `10 passed`，final review 暴露 metadata 整体缺失的 cross-K 漏洞后补测并修复，最终 fresh `13 passed in 3.61s`；Task 5 YAML RED 为 tactile16 文件缺失，GREEN `7 passed`。命令均为 `JAX_PLATFORMS=cpu PYTHONPATH=src:. ... python -m pytest -p no:cacheprovider <focused test> -q`，精确输出保存在 `.git/worktrees/FRS_Tact-modalities-eval/sdd/task-{1..5}-report.md`。
- 2026-08-08 post-fix fresh full CPU regression：`env JAX_PLATFORMS=cpu PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp HF_DATASETS_CACHE=/tmp/frs_tactile_ratio_hf_cache .../python -m pytest -p no:cacheprovider modalities_eval/test tests/jax -q` -> exit 0，`134 passed, 1 skipped in 9.56s`；首次 review 前为 `131 passed, 1 skipped in 9.25s`，新增 3 个 metadata-absent resume cases 后为最终计数。skip 仍是需要显式 opt-in 的真实 checkpoint reconstruction test。
- fresh 静态/schema 验证：brief 指定的 9 文件 `py_compile` exit 0、无输出；`git diff --check` exit 0、无输出；现有 VT checkpoint schema smoke exit 0 并输出 `parameter_schema_unchanged (960, 512)`，同时验证 K=1/8/21 effective tokens 为 4/32/84。
- 沙箱外 RTX 4090 checkpoint-backed GPU smoke（每个 K 为全新 Python 进程，episode 48 frame 249，`num_steps=1`）均 exit 0、sample actions 全 finite：K=1 prefix 181，`19.527281855 s`，peak `1,904,225,280` bytes；K=8 prefix 209，`19.807580470 s`，peak `1,903,243,776` bytes；K=21 prefix 261，`20.760682782 s`，peak `1,903,823,872` bytes。它们只证明兼容性/性能 smoke，不是论文训练或结果。
- K=1/8/21 YAML 解析后去除 K、`output`、W&B `name/tags` 即完全相等；输出分别为 `/workspace/vtsmolvla_tactile_01`、`/workspace/vtsmolvla_tactile_repeat16`、`/workspace/vtsmolvla_tactile_repeat32`。正式 paper runs 仍必须从相同 base initialization、split、seed、训练协议开始，且不同 K 禁止 strict optimizer resume，只能 weight warm-start 新 run。
- 本次没有启动训练，也没有生成 paper result。独立 final review 曾发现 1 个 Important：metadata-absent legacy checkpoint 隐含 K=1 却可被 K=8/21 strict restore 接受；现已在 fallback 中把 saved 缺失 K 规范为 1 并与 current K 精确比较，新增 K1->K1 pass 和隐含 K1->K8/21 reject 测试。post-fix 独立复审为 Critical 0、Important 0、`Ready to integrate: Yes`。GPU smoke 未重复运行，因为修复只触及 `training.py` legacy restore 与测试，之前 smoke 使用的 `modeling.py`/configuration/checkpoint/config 均未再变。剩余限制：GPU smoke 仅一个 checkpoint/episode/frame/一步采样；exact output/name/tag 字符串没有独立测试锁定；worktree 的 evaluator 与 token baseline 仍是同一份未提交累计 diff。

### Rebase 后测试 package-boundary 修复（2026-08-08）

- `Lee` rebase 后，repeat-factor 测试被自动合入纯视觉 `tests/jax/test_checkpoint.py` 与 `tests/jax/test_training.py`，导致 VT feature 测试跨越 package boundary。
- 修复范围仅为测试重分包：repeat-factor config/checkpoint 测试迁到 `tests/jax/test_tactile_checkpoint.py`，resume helper/tests 迁到 `tests/jax/test_tactile_training.py`；新文件显式使用 `lerobot.policies.smolvla_jax` legacy VT config/checkpoint/trainer，TinyModel 使用该 trainer 的 `loss(params, batch, rng)` API。
- origin/eric 原有 `test_effective_config_persists_tactile_fusion_settings` 保留在 `tests/jax/test_checkpoint.py`；未改生产代码、contract 文件或纯视觉 baseline bug。
- 修改前 RED：正确测试解释器执行聚焦集合 exit 2，`tests/jax/test_training.py` collection 因 `train_smolvla.training` 导入缺失的 `initialize_tactile_fusion_params` 失败。迁移后两个新 VT 文件 fresh 为 `19 passed in 1.00s`；加上保留的 origin/eric tactile-fusion test 为 `20 passed in 1.01s`。四文件 focused 仍只在纯视觉 `test_training.py` 的同一 upstream import 处 collection exit 2；未越界修复。
- 静态与范围验证：四个测试文件 `py_compile` exit 0，`git diff --check` exit 0；两个旧测试文件相对 `origin/eric` 完全一致（`git diff --exit-code` 为 0），确认只迁移 feature 新增测试且保留原测试；未提交。

### Rebase 后 repeat-factor contract 兼容（2026-08-08）

- 当前待发布分支为 `Lee`。最新 `origin/eric` 新增 checkpoint/publish/deploy contract 后，原实现尚未把 `tactile_token_repeat_factor` 纳入该边界，导致 K=1/8/21 可被误判为同一部署合约。
- `CheckpointContract` 现显式保存 `tactile_token_repeat_factor`；训练 effective config、checkpoint `config.json`、发布 training YAML/manifest 与部署 YAML 都保留显式 K，并在合约比较中 fail closed。`tactile_num_tokens` 仍表示源 tactile 数量，不乘 K，也不改变 cache `[F,4,512]`。
- 兼容规则固定为：旧 checkpoint config、旧 conversion manifest、旧 deploy YAML 缺字段时回落 K=1；显式值必须是严格正整数，拒绝 0、负数、bool、float 与数字字符串。默认部署 YAML 已显式写入 K=1。
- TDD RED：首轮聚焦为 `16 failed, 2 passed, 7 errors`，均来自 contract 字段缺失/未贯穿；补充 manifest 严格类型边界后为 `5 failed`。最小实现后，排除已知默认部署身份漂移测试的三文件回归为 `106 passed, 1 deselected in 5.03s`。
- 完整三文件回归为 `106 passed, 1 failed in 5.01s`；唯一失败仍是 rebase 基线中 `test_default_deployment_config_pins_the_bimanual_vt_contract` 期待 `KaiyueChen/vtsmolvla_01_4w`，而仓库默认 YAML 指向 `/home/typhon/models/pick_tube_02_3w_jax`。这不是 repeat-factor 回归，未越界改写 checkpoint/机器人部署身份。
- contract 相关 6 个 Python 文件 `py_compile` exit 0，`git diff --check` exit 0；未提交、未 push。
- 同轮 package 对齐还把通用 `state_mask` forwarding 同步到 `train_smolvla/modeling.py`；`tests/train_smolvla/test_package_boundary.py` 通过实际 capture `build_prefix_context` kwargs 验证透传。该项 TDD 为 RED `1 failed`、GREEN `1 passed`。
- 父任务在全部 rebase 兼容改动后独立复跑：相关聚焦集合 `160 passed, 1 skipped, 1 deselected in 7.09s`，仅显式 deselect 上述已知默认部署身份漂移；完整 `tests/jax + modalities_eval + package-boundary` 在 collection 阶段仍被 `origin/eric` 现有 `train_smolvla.tactile_cache` 缺失与 `initialize_tactile_fusion_params` 缺失阻断（3 errors）。checkpoint schema 探针 exit 0，`tactile_proj.weight=(960,512)`，K=1/8/21 的源 token 数仍为 4、effective 分别 4/32/84。
- rebase 后又在宿主 RTX 4090 使用真实 VT checkpoint、v3 episode 48 frame 249、`num_steps=1` fresh 跑 K=1：prefix 181、actions finite、`21.491298085 s`、peak `1,902,840,832` bytes，exit 0。该 smoke 只验证 active legacy VT 导入与 K=1 推理兼容，不是论文结果。

### Unified BF16 compute parameter contract（2026-08-08）

- `JaxSmolVLAConfig.trainable_compute_dtype` 现在只接受并显式序列化 `"bfloat16"`；旧 config、resume metadata、publish manifest、training YAML 和 deploy contract 缺字段时统一迁移为 BF16，显式非法值 fail closed。
- 唯一计算转换入口为 `src/lerobot/policies/smolvla_jax/training.py::prepare_params_for_compute(params, config)`：仅按现有 `is_trainable_parameter` 将 trainable floating leaves 转为 BF16，frozen 与 integer leaves 保持原 dtype；trainer train/eval、`JaxSmolVLAPolicy` 和 `modalities_eval.SmolVLAEvalModel` 共用该入口。
- trainer state 和 `model.safetensors` 继续保存 FP32 trainable master parameters；固定 batch/rng/noise 的 save-load-prepare 数值测试逐元素一致。未修改 tactile cache、schema 或 repeat-factor 语义。
- 用户批准的 Task 2 BF16 支持面仅为 active VT tactile runtime `src/lerobot/policies/smolvla_jax` 及其 `modalities_eval`、publish、deploy consumers；预先损坏的纯视觉 `train_smolvla` package 和 `tests/jax/test_training.py` 明确不属于验收面，也不在本任务中修复。
- Task 2 聚焦 RED 为 `43 failed, 105 passed`，direct-contract 补充 RED 为 `1 failed`；最终聚焦 GREEN 为 `151 passed`。较宽但仍可运行的回归实际为 `269 passed, 1 skipped, 2 deselected`，不是先前误记的 267；被排除项来自上述纯视觉 package boundary 与既有默认部署身份漂移，不改变 VT focused 验收结论。

### Immutable train-only normalization protocol（2026-08-08）

- K8/K21 YAML 现在显式共享 `/workspace/normalization_protocols/pick_tube_vt_k8_k21`，split 在 normalization 前解析；协议只用选中 train episode 的 v3 `stats/*` parquet 元数据，并通过官方 `cast_stats_to_numpy` / `aggregate_stats` 做 count-weighted 聚合。validation episode 不参与，协议阶段不读取 frame/data parquet 或解码图像/视频。
- 协议 artifact 包含 immutable `data_split.json`、pre/post safetensors 和 `normalization_manifest.json`；manifest 固定 source 顺序、requested/resolved revision/action key、rename map、sorted train episode IDs、逐 episode/source/final float32 stats digest、split digest、asset hashes 与 20D state/action contract。创建用同父目录 staging + atomic rename；已有 identical artifact 只读复用，缺失、损坏或 drift fail closed。
- 本地 v3 root 直接读取 `meta/info.json` 后做 predicate-pushed episode stats 读取，避免 eager metadata container 加载全局 stats/未过滤 episode metadata；remote/unmaterialized source 才使用 `LeRobotDatasetMetadata` 完成 metadata 解析。显式 preprocessor 下 loader 完全跳过 full-dataset stats，validation 复用 train preprocessor。
- checkpoint 保存精确复制协议 split/manifest，并保存 byte-identical normalization assets；strict resume 先用当前选中 episode metadata 校验 checkpoint-authoritative artifact，再加载 normalization assets，最后才 restore optimizer/trainer state，保证 step 1 前 fail closed。
- Task 3 严格 TDD RED 包括新 module collection `ModuleNotFoundError`、resume-order helper 缺失、K8/K21 protocol config 缺失、aggregate-preserving per-episode drift 未被发现、requested action-key drift未被发现及 eager metadata 读取；最终 focused/broader 计数以 `.git/worktrees/FRS_Tact-modalities-eval/sdd/task-3-report.md` 为准。未启动 H100 训练；真实双 H100 K8/K21 save/resume smoke 仍是 production gate。
