# FRS base model 切换：SmolVLA → pi0.5（JAX，vendored from openpi）

分支：`pi05-frs-jax`（从 `eric` 切出）。目标：FRS（`train_pi05_frs`）不再使用 SmolVLA，
全部按 pi0.5（Physical Intelligence 的 openpi，JAX 原生实现）的要求来——包括主环境的
jax/flax/transformers/orbax 版本，不用兼顾 SmolVLA 还能不能跑。这个文件记录已确认的架构决策、
关键发现和还没做完的事，避免跨 session 丢上下文。

## 2026-08-12 重写：pi05_jax 改成 openpi JAX 源码的忠实副本 + 补全训练栈

这一轮把 `src/lerobot/policies/pi05_jax/` 从"vendor 模型代码 + 手写周边"改成"逐字照搬 openpi 的
JAX 代码"。下面这一节记录改动；**本文件后面几节里凡是和这里冲突的描述，以这里为准**
（尤其是 `sharding.py` 是 no-op shim、`normalize.py` 不引 pydantic、`pi0.py` 里新增三个方法这几条，
都已经不成立了）。目录级的逐文件对应表在
[src/lerobot/policies/pi05_jax/README.md](src/lerobot/policies/pi05_jax/README.md)。

对照物：upstream openpi `main` 分支 commit `15a9616a00943ada6c20a0f158e3adb39df2ccac`
（就是当初 vendor 的那个 commit）。

**改回逐字原版的（之前是改写/裁剪过的）：**

- `pi0.py`：原来在文件末尾多了 `Pi0PrefixCache`/`build_prefix_cache`/`denoise_step`。现在这三个
  东西搬到新的 `frs.py`，写成接收 model 作参数的自由函数，`pi0.py` 恢复成逐字原版。这样
  "FRS 自己加的模型级逻辑" 和 "upstream 代码" 物理隔离，`frs.py` 有 bug 也影响不到
  `sample_actions`。
- `tokenizer.py`：原来只留了 `PaligemmaTokenizer`（58 行），现在是完整的 371 行原版
  （`FASTTokenizer`/`BinningTokenizer`/`FSQTokenizer` 都在），并连带 vendor 了它依赖的
  `models/utils/fsq_tokenizer.py`。代价：import 这个包现在会连带 import `transformers`
  （`AutoProcessor`）和 `chex`——upstream 就是这样，pi0.5 本身只用 `PaligemmaTokenizer`。
- `normalize.py`：原来是绕开 pydantic 的手写版（含自定义 `apply`/`unapply`）。现在是原版
  （pydantic + numpydantic 的 `NormStats`、`RunningStats`、`serialize_json`/`save`/`load`）。
  原来的 `apply`/`unapply` 对应 upstream `transforms.Normalize`/`Unnormalize`，所以连带 vendor
  了完整的 `transforms.py`，调用方改用它。
- `sharding.py`：原来是逐条意译的版本，现在是逐字原版，并按 upstream 的位置移到
  `training/sharding.py`（`gemma.py`/`siglip.py` 的 import 跟着改成 `from .training import sharding`，
  和 upstream 的 `import openpi.training.sharding as sharding` 结构一致）。

**新增的（原来完全没有）：**

- `training/`：`utils.py`（`TrainState`）、`optimizer.py`（`CosineDecaySchedule`/`AdamW`/
  `create_optimizer`）、`weight_loaders.py`（`CheckpointWeightLoader` 会自动补 LoRA 权重）、
  `checkpoints.py`（orbax `CheckpointManager`，checkpoint 里带 `assets/`）——全部逐字原版。
- `training/config.py`：upstream 的 `AssetsConfig`/`DataConfig`/`ModelTransformFactory`/
  `DataConfigFactory`/`TrainConfig`/`cli()`/`get_config()` 骨架照搬，把 upstream 的
  Aloha/Libero/DROID 配置换成 `LeRobotPickTubeDataConfig` + 三个注册配置
  （`pi05_pick_tube` LoRA、`pi05_pick_tube_full` 全量、`debug` 假数据）。
- `training/data_loader.py`：upstream 的协议/`TransformedDataset`/`TorchDataLoader`/
  `DataLoaderImpl` 照搬，**唯一必须改的地方**是把
  `import lerobot.common.datasets.lerobot_dataset`（官方 lerobot，正是包名冲突的来源）换成本仓库的
  `lerobot.datasets.LeRobotDataset`；另外支持多数据集拼接（`DataConfig.sources`，pick_tube 有 4 个）
  并加了两个本地 transform（`RenameKeys`/`PromptFromTask`）来做每数据集的相机重命名，
  这样 `transforms.py` 能保持逐字原版。RLDS/DROID 那条路没搬（要 TensorFlow）。
- `policies/pick_tube_policy.py`：对应 upstream 的 `policies/libero_policy.py`，
  `PickTubeInputs`/`PickTubeOutputs`。pick_tube 没有第三人称相机，`base_0_rgb` 补零 + mask=False，
  和 `LiberoInputs` 处理它用不到的 `right_wrist_0_rgb` 是同一套写法。
- `tools/train_pi05_jax.py` = openpi `scripts/train.py`（只改 import 和 `sys.path`）；
  `tools/compute_pi05_norm_stats.py` = openpi `scripts/compute_norm_stats.py`。
- `policy_config.py`（取代原 `checkpoint.py`）：upstream `policies/policy_config.py` 里
  `create_trained_policy` 的加载权重那一半，加上能吃 URL 的 `load_norm_stats`。

**训练入口从 YAML 改成 openpi 原版的 `TrainConfig` + tyro**：`configs/train_pi05_pick_tube.yaml`
删除，配置写在 `training/config.py` 的 `_CONFIGS` 里；`scripts/start_pi05_train.sh` 改成
`train_pi05_jax.py <config_name> --exp_name=<run>`，并在训练前自动检查/生成 norm stats。
注意这只影响"微调 pi0.5 本身"这条链路——**FRS 那条链路
（`configs/train_frs_pick_tube_pi05.yaml` + `scripts/start_frs_pi05_train.sh`）的 YAML 接口没变**。

**为此新增的依赖**（都是 openpi 自己 pyproject 里就有的）：`pydantic`、`numpydantic`、
`etils[epath]`、`tyro`、`chex`。之前刻意回避 pydantic/numpydantic/etils 的决定作废。

**跟着改的调用方**（FRS 链路保持可用）：

- `modalities_eval/pi05_utils.py`：`Pi05SampleProcessor` 构造签名不变，但 `prepare_sample` 内部
  改成拼 openpi 的 transform 链（repack → `PickTubeInputs` → `Normalize` → `ResizeImages` →
  `TokenizePrompt` → `PadStatesAndActions`），顺序和 `ModelTransformFactory` +
  `transform_dataset` 完全一致——**action cache 和训练现在走同一份预处理代码**。
  删掉了手写的 `_prepare_image`/`_resize`。
- `utils/pi05_source_model.py`：`model.build_prefix_cache(...)` → `frs.build_prefix_cache(model, ...)`。
- `prepare_pi05.py`：`load_norm_stats` 从 `normalize` 移到包顶层（`policy_config`）。
- `tests/pi05/test_normalize.py` 改用 `transforms.Normalize`/`Unnormalize` 并加了 JSON 往返测试；
  `tests/pi05/test_training.py` 整体重写（配置注册表、pick_tube transform 链、`RenameKeys`、
  LoRA 权重合并、schedule）；`test_cache_optimizations.py` 删掉了针对已移除的 `_resize` 的用例。

**本地验证到什么程度**：这台 macOS 上 `.venv` 里 jax/torch/numpy 全都没装（和本文件最后一节说的
一致），所以只做了静态检查：全仓 AST 语法通过；一个专门写的脚本对 10 个逐字 vendor 的文件做了
"除 provenance 头和 import 行外与 upstream 完全一致"的机器校验（全过）；另一个脚本 AST 解析了
39 个 pi0.5 相关文件的所有一方 import，确认每个名字在目标模块里真实存在；两个 shell 脚本
`bash -n` 通过。**真正的运行验证还没做**，清单见 pi05_jax/README.md 的 Status 一节，第一步是
`python tools/train_pi05_jax.py debug --exp_name=smoke`（假数据，不需要数据集和 checkpoint）。

## 2026-08-12：打通"先微调 pi0.5，再用它做 FRS"这条路

确定的路线：**A（微调 pi0.5）的产物 → B（FRS）的 base model**。之前 B 一直用官方 `pi05_base`。
接的时候发现一个会**静默出错**的坑，已修：

`BaseModelConfig.load` 默认 `remove_extra_params=True`，会把 checkpoint 里当前模型结构装不下的
参数直接 `intersect_trees` 掉。用 `pi05_pick_tube`（LoRA）训出来的 checkpoint 如果被默认的
`Pi0Config`（`gemma_2b`/`gemma_300m`，不带 LoRA）加载，结果是：LoRA 权重全部被丢弃 → 剩下的
base 权重因为 LoRA 训练时是**冻结**的，和原始 pi05_base 一模一样 → 拿到的就是没微调过的模型，
**全程不报任何错**。整个 action cache 会白跑。

修法：

1. `policy_config.load_pi0` 加 `_reject_unused_params`：加载前用 `nnx.eval_shape` 拿到该配置
   期望的参数树，和 checkpoint 的参数树求差集，有多余的就报错；如果多余的键里含 "lora"，
   错误信息直接点名"这是 LoRA checkpoint 配了非 LoRA config"。想故意丢弃要显式传
   `allow_extra_params=True`。测试见 `tests/pi05/test_training.py`。
2. variant 打通到 FRS 侧：`configs/train_frs_pick_tube_pi05.yaml` 的 `model` 段新增
   `paligemma_variant`/`action_expert_variant`，经 `tools/prepare_frs_pi05_cache.py` →
   `prepare_pi05.prepare_cache` → `Pi05SampleProcessor` 一路传下去，`start_frs_pi05_train.sh`
   的 checkpoint 冒烟检查也带上。默认值仍是 `gemma_2b`/`gemma_300m`（对应官方 pi05_base）。

**切换 checkpoint 时必须同时改三处**（YAML 顶部注释里写了）：`checkpoint`、
`model.*_variant`、`norm_stats.dir`+`asset_id`。第三处不能忘：训练用哪份归一化统计量，
生成 action cache 就必须用同一份，否则 pi0.5 看到的 state/action 尺度和它训练时对不上。

**顺带解决的**：本文件后面"norm stats 用借来的 trossen（14 维 vs pick_tube 20 维，多出 6 维不
归一化）"那一节，在这条路线下自动作废——`tools/compute_pi05_norm_stats.py` 会用 pick_tube 自己的
数据算真实的 20 维统计量，训练时写进 checkpoint 的 `assets/pick_tube/`，FRS 直接指过去即可。

## 架构决策：把 pi0.5 模型代码搬进本仓库，不装 openpi 这个包

openpi 自己的 `pyproject.toml` 会拉官方 `lerobot` 包（pin 死某个 commit），和本仓库自己的
`lerobot` 包（`src/lerobot/`，本仓库就是这个包）**撞名**——同一个环境里 `lerobot` 这个 import
路径只能指向一份代码，装官方的就会顶掉本仓库自己的代码，反过来也一样。这和版本号无关，纯粹是
两边都想叫 `lerobot`。

解决办法：不 `pip install openpi`，而是把 pi0.5 **模型代码本身**（网络结构 + checkpoint 加载，
不要 openpi 的训练/数据/policy 那套基础设施）直接搬进本仓库，放在
[src/lerobot/policies/pi05_jax/](src/lerobot/policies/pi05_jax/)，和 `smolvla_jax/`
（手写的 JAX 版 SmolVLA）并列。这样主环境里只有本仓库自己的 `lerobot` 包，没有名字冲突。

代价：既然要用 pi0.5 的代码，主 `pyproject.toml` 的 jax/flax/transformers/orbax-checkpoint
就按 openpi 的精确 pin 版本改了（`jax==0.5.3` / `flax==0.10.2` / `transformers==4.53.2` /
`orbax-checkpoint==0.11.13` / `ml-dtypes==0.4.1`），不再是之前 SmolVLA 用的版本范围。
`smolvla_jax/` 的代码大概率会因此跑不起来——按你的要求，这个分支不需要管。

**一处故意没跟 openpi 保持一致**：openpi pin 的是 `numpy>=1.22.4,<2.0.0`，但本仓库的
`opencv-python-headless`/`pyarrow`/`datasets` 已经要求 `numpy>=2.0.0`，两边范围不相交。
如果强行降到 numpy<2，大概率把这些包也一起弄坏。所以 `pyproject.toml` 里 numpy 保持
`>=2.0.0,<2.3.0` 不变。训练服务器已实际解析并安装 NumPy 2.2.6，JAX/Flax 导入、GPU
发现、官方 checkpoint 恢复和目标测试均通过，因此这项兼容风险已经排除。

## 现状：代码已完成，环境和 checkpoint 已在 H100 服务器验证

已经做的：

- [src/lerobot/policies/pi05_jax/](src/lerobot/policies/pi05_jax/)：从 openpi vendor 的
  pi0.5 模型代码（`model.py`/`pi0.py`/`pi0_config.py`/`gemma.py`/`siglip.py`/`lora.py`/
  `tokenizer.py`/`array_typing.py`/`nnx_utils.py`/`image_tools.py`/`download.py`），加上
  自己写的 `sharding.py`（单机 no-op shim）、`checkpoint.py`（`load_pi0()`）、`normalize.py`
  （`NormStats` + z-score/quantile 的 apply/unapply，公式抄自 openpi 但不引入
  pydantic/numpydantic/etils 三个额外依赖），以及 `pi0.py` 里新增的
  `Pi0PrefixCache`/`build_prefix_cache`/`denoise_step`（从 upstream `sample_actions` 的内联
  闭包里拆出来的单步 velocity 计算，`sample_actions` 本身一个字节没动）。细节见该目录
  [README.md](src/lerobot/policies/pi05_jax/README.md)。
- [modalities_eval/pi05_utils.py](modalities_eval/pi05_utils.py)：`Pi05EvalModel`，SmolVLA 那边
  `SmolVLAEvalModel` 的 pi0.5 版——把 LeRobotDataset 的一条 sample 变成 pi0.5 的 `Observation`
  （处理三路相机映射/空相机补黑图、state 归一化后离散化拼进 tokenized prompt、action 归一化）。
- [utils/pi05_source_model.py](utils/pi05_source_model.py)：`sample_and_reverse`/
  `reverse_integrate_actions`，把 `build_prefix_cache`/`denoise_step` 接到
  `utils/integration.py` 的 euler/fireflow 求解器上——SmolVLA 那边对应
  `utils/source_model.py`。故意没有直接改 `utils/source_model.py` 或从里面 import，
  避免 pi0.5 这边被拉去 import `lerobot.policies.smolvla_jax`。
- [prepare_pi05.py](prepare_pi05.py) + [tools/prepare_frs_pi05_cache.py](tools/prepare_frs_pi05_cache.py)：
  和 SmolVLA 那边 `prepare.py`/`tools/prepare_frs_caches.py` 结构完全一样（记录选取、
  manifest/resume、memmap 数组都复用同一套 `utils/cache.py`），只是模型侧换成上面几个新模块。
  产出的 action_cache 落盘格式不变，`tools/train_frs.py` 不用改一行。
- 顺手做的两处小重构（低风险、纯粹是把已有的、和具体 base 模型无关的纯函数挪到共享位置，
  让 SmolVLA 和 pi0.5 两边用同一份实现，不会各写一份然后行为慢慢分叉）：
  - `utils/cache.py` 里的 `build_records`（上一轮已经做的，这里不重复）；
  - 新增 `lerobot/datasets/sample_utils.py`（`resolve_action_key`/`action_delta_timestamps`/
    `lerobot_sample_to_observation`，从 `smolvla_jax/data.py` 挪出来，那边保留原名字重新
    导出，其他地方的 import 不用改）、`lerobot/datasets/dataset_sources.py`
    （`DatasetSource`/`parse_dataset_sources`/`resolve_source_visual_keys`）和
    `lerobot/datasets/tactile_cache.py`（触觉 embedding cache）；后两者让 pi0.5 的
    预计算和 FRS 训练不再 import `smolvla_jax`。以及 `utils/flow_matching.py`
    （`deterministic_noise`/`inversion_mse`，从 `utils/source_model.py` 挪出来，同样保留重新导出）。
- 修了一个真实 bug（不是 TODO，是实际改对了）：`prepare_pi05.py`/
  `tools/prepare_frs_pi05_cache.py` 一开始把 `gs://...` checkpoint URL 直接传给
  `pathlib.Path(...)`，Python 的 `Path` 会把 URL 里的 `//` 归一化成单个 `/`
  （`Path("gs://a/b")` 变成 `"gs:/a/b"`），下载会失败。现在用 `_is_local_path`
  （基于 `urllib.parse.urlparse`）先判断是不是 URL，是的话全程留在字符串形态，
  只有确认是本地路径才包一层 `Path`。

**2026-08-10 在 Linux 双 H100 80GB 服务器完成的验证：**

1. `uv lock` 和 `uv sync --frozen` 成功；实际安装 JAX 0.5.3、Flax 0.10.2、
   Transformers 4.53.2、Orbax Checkpoint 0.11.13 和 NumPy 2.2.6。
2. `scripts/setup_env.sh` 完整执行成功，JAX 和 PyTorch 都识别两张 H100。
3. `load_pi0("gs://openpi-assets/checkpoints/pi05_base")` 使用
   `Pi0Config(pi05=True, action_dim=32, action_horizon=50)` 成功恢复官方 checkpoint；
   因此配置里的 action horizon 50 已确认，不再是占位值。

**仍需用真实数据验证的：**

1. **`build_prefix_cache`/`denoise_step` 拆分对不对**——这是这次唯一真正新写的模型级逻辑
   （其余都是原样 vendor），最值得单独验证：用同一份输入分别跑 upstream 原版
   `sample_actions`（t:1→0）和"手动逐步调 denoise_step"，两边应该给出完全一样的结果。
2. `Pi05EvalModel.prepare_sample` 拼出来的 `Observation`/`Pi0PrefixCache` 在
   `nnx.split`/`nnx.merge` + `jax.jit`（`utils/pi05_source_model.py`）下能不能正常跑通一次
   完整的 `sample_and_reverse`，shape 对不对。特别是 `prepare_pi05.py:_pad_observation_batch`
   用 `jax.tree.map` 给 `Observation`（flax `struct.dataclass` + jaxtyping 装饰）补批次维度
   这一步——只做过代码审查（`array_typing.py` 里那个 patch 会在 tree unflatten 时跳过类型
   检查，所以理论上没问题）。触觉 encoder `KaiyueChen/encoder_ckpt_0809` 已下载并成功
   加载；四个 pick_tube 数据集已下载、从 v2.1 转换为 v3.0，并逐个读取真实样本成功。
   全量三阶段流水线已在服务器启动，触觉 embedding 和 action cache 仍在生成中。

## 全盘代码审查（2026-08-08）

做了一遍完整审查，**查出 4 个真问题，都已修**：

1. **图像通道顺序错了**（commit `3aa4611`，最严重）：`LeRobotDataset` 返回 CHW
   （`lerobot/datasets/io_utils.py` 的 `pil_to_chw_tensor` → `ToTensor()`），pi0.5 的
   `siglip.py` 要 HWC，原来没转置。不会崩溃，但视觉特征全是垃圾——典型的静默错误。
2. **jit 缓存会串用别的模型的权重**（commit `dd46f41`）：缓存 key 用 `id(model)` 且把权重
   焊进闭包，而 `tools/prepare_frs_pi05_cache.py` 每个数据集新建并释放一个 `Pi0`，CPython
   会复用释放掉的地址。同一个闭包还会把 4 个模型全钉在显存里。已改成"缓存只放编译好的函数、
   权重每次从调用方传入"（和 `utils/source_model.py` 久经验证的形式一致）。
3. **触觉历史窗口长了 5 倍**（commit `584770d`）：`tactile_window_divisor: 1` 是从 SmolVLA
   配置抄过来的占位值，但 SmolVLA 的 `chunk_size=10`、pi0.5 的 `action_horizon=50`，
   同一个 divisor 会让窗口从 10 个 token（约 1 秒）变成 50 个（5 秒）。能整除所以不报错。
   已改成 `divisor: 5`，数值上和 SmolVLA 版完全对齐。
4. **ruff F401**（commit `dd46f41`）：为向后兼容做的重新导出会被 lint 判成未使用 import。

**已确认没问题的**：vendor 的每个文件都逐一 diff 过 upstream（`pi0.py` 到 `sample_actions`
结尾与原版完全一致，新增的三个方法单独在文件末尾）；`siglip.py` 唯一的差异经查证是**原版的
bug**（`MAPHead` 的 `dtype_mm` 被误写成 `dtype=`）且在 pi0.5 配置下是死代码；
`prepare_pi05.py` 的批处理/断点续跑/manifest 逻辑与 `prepare.py` 逐行一致；被我动过的
SmolVLA 侧重构，所有既有调用方都还能正确解析。

**离线验证过的**（本机没有 jax/torch/pytest，以下都是纯 numpy/stdlib 能跑的部分）：
`utils/cache.py` 自带的 6 个测试全过；`build_records` 的鸭子类型元数据（纯 int 和
numpy array 两种）、样本数、train/val 不重叠、确定性、错误路径、超短 episode 容错；
`normalize.py` 对着**真实下载的 trossen norm_stats.json** 验证解析、14→20 补维、
quantile/z-score 往返精度（~5e-7）、二维广播、更宽数组的 unapply、过宽统计量被拒绝；
`_is_local_path` 对 7 种路径形态判断正确、`gs://` URL 在字符串拼接下不被破坏；
以及**整个 YAML 配置对着真实 pick_tube_01 元数据的交叉校验**（rename_map 源键存在、
camera_map 槽位合法且目标 rename 后存在、触觉键存在且没被喂给 pi0.5、维度都在
action_dim 内、窗口能整除、丢尾巴后最短 episode 还够用、4 个数据集共用同一 rename_map、
没有残留的 REPLACE_ME）——全部通过。

**norm stats 从哪来：已经定了，按你的要求直接用官方的。**
[configs/train_frs_pick_tube_pi05.yaml](configs/train_frs_pick_tube_pi05.yaml) 里
`norm_stats.asset_id: trossen`——pi05_base 的 `assets/` 目录列出来是 arx/arx_mobile/droid/
fibocom_mobile/franka/trossen/trossen_mobile/ur5e_dual，选 trossen 是因为它和 arx/ur5e_dual
一样是双臂平台，且是 Physical Intelligence 自己最常用的双臂平台（ALOHA 用的就是 Trossen
机械臂），跟 pick_tube 的"左手/右手"任务设定更接近；droid/franka 是单臂，先排除。
**维度不完全匹配**：trossen 的 state/actions 统计量是 14 维，pick_tube 实际是 20 维——毕竟不是
pick_tube 自己机器人的统计量。`modalities_eval/pi05_utils.py:_match_norm_stats_dim` 会自动把
多出来的 6 维补成"不做归一化"（mean=0/std=1，运行时会打印 WARNING），先把链路跑通；这几维
的数值不会被正确归一化，pi0.5 拿 state 编码进 prompt 时如果这几维超出 `[-1,1]` 会被直接截断到
边界（不会崩溃，但不严谨）。真要认真训练的话应该换成用 pick_tube 自己的数据算一份统计量
（用 `tools/compute_pi05_norm_stats.py`，就是 openpi 的 `scripts/compute_norm_stats.py`）。

**相机映射：已经查过真实数据集 + 你确认了左右手对应关系，不再有不确定性了。**
查了 [KaiyueChen/pick_tube_01](https://huggingface.co/datasets/KaiyueChen/pick_tube_01) 的
`meta/info.json`/`meta/tasks.jsonl`（四个 pick_tube_0X 结构一致）：`robot_type: "bimanual"`，
只有 `camera0`/`camera1` 两路 RGB（没有第三路/外部机位），task 原文是"Use the left hand to
pick up the green tube, and then use the right hand to pick up the blue tube."（双臂操作），
触觉 key 是 `tactile_left_0/right_0`（一个夹爪两个指垫）+ `tactile_left_1/right_1`
（另一个夹爪两个指垫）——"0"/"1" 和 `camera0`/`camera1` 编号对得上，说明这两路相机是
两条手臂各自的腕部相机，不是"主视角 + 单个腕部"的组合。所以
`configs/train_frs_pick_tube_pi05.yaml` 把两路都填进 `left_wrist_0_rgb`/`right_wrist_0_rgb`，
`base_0_rgb` 留空（自动补黑图 + mask=False）。左右手对应关系你已经确认：**camera0 = 左手**。
数据集自带的 `rename_map` 把编号整体加一（`camera0`→`camera1`、`camera1`→`camera2`），所以
`camera_map` 里写的是 rename 之后的名字：`observation.images.camera1`（= 原始 camera0）
对应 `left_wrist_0_rgb`，`observation.images.camera2`（= 原始 camera1）对应
`right_wrist_0_rgb`。

顺带确认了其他几个事实：`fps=30`、`observation.state`/`actions` 都是 20 维（和
`train_smolvla_jax.yaml` 里 `state_dim`/`action_dim: 20` 对得上）、`total_tasks: 1`（单任务，
上面那句 prompt 就是唯一的语言指令）、这几个数据集在 HF hub 上是 LeRobot `v2.1` 格式
（`codebase_version`），但 `configs/train_frs_pick_tube_pi05.yaml` 里 dataset root 写的是
`/workspace/lerobot_v30/...`——说明现有 SmolVLA 管线在使用前已经把它们转成了 v3.0，pi0.5
这边直接复用同一份转换好的本地数据，不需要另外处理版本转换。

## 关键 openpi API / 事实参考（省得以后重新翻源码）

（vendor 自 openpi `main` 分支 commit `15a9616a00943ada6c20a0f158e3adb39df2ccac`，2026-06-16）

- 官方 pi0.5 base checkpoint：`gs://openpi-assets/checkpoints/pi05_base`
  （`params/` 是 orbax PyTree checkpoint，`assets/` 存 norm stats）。openpi 仓库里没有现成的
  "直接对 pi05_base 做 zero-shot 推理"配置，只有一堆以它作初始权重的 finetune 配置
  （`pi05_aloha`/`pi05_droid`/`pi05_libero`，见 openpi `src/openpi/training/config.py`）。
- Flow-matching 约定和 SmolVLA 一致：t=1 是噪声，t=0 是数据，`dt = -1/num_steps`
  （`pi0.py:Pi0.sample_actions`）。
- pi0.5 vs pi0 的两个模型级差异（`pi0_config.py:Pi0Config.pi05` 文档字符串）：
  state 走离散语言 token 而不是连续输入；action expert 用 adaRMSNorm 注入 flow-matching
  timestep 而不是把 timestep 和 action 拼一起过 MLP。
- `Pi0.embed_prefix`/`embed_suffix` + `PaliGemma.llm`（gemma.py 里的双专家 transformer，
  PaliGemma 语言/视觉专家 + action expert）是核心计算图；`siglip.py` 是视觉塔（`variant="So400m/14"`），
  不经过 `vit.py`（openpi 里那是另一套没被 pi0.5 用到的 ViT 实现）。
- checkpoint 加载模式：`model.restore_params(checkpoint_dir/"params", dtype=jnp.bfloat16)` +
  `Pi0Config(...).load(params)`——照抄自 openpi 自己的
  `policies.policy_config.create_trained_policy`。

## 交叉核对：和 ~/VLA/VB-VLA（同一个团队的另一个项目）的架构差异

`~/VLA/VB-VLA` 是你们团队自己 fork 过的 openpi（`policy/src/openpi`），跑的是双臂机器人的
pi0.5 部署/训练。核对了一下发现两件事：

1. **camera0 = 左手，确认无误**：`policy/src/openpi/policies/vb_policy_vitac.py` 里写死
   `left_image = data["observation.images.camera0"]`，和上面确认的一致，不用改。
2. **触觉接入方式不一样，但已经确认按现状不改**：VB-VLA 的 `vb_policy_vitac.py`/
   `pi05_chaoyi_vitac` 这套是把 4 路触觉图像**直接当额外相机喂进 pi0.5 本身**
   （`image_keys = (left_image, right_image, tactile_left_0, tactile_right_0,
   tactile_left_1, tactile_right_1)`，六路图都过 pi0.5 自己的 SigLIP），配套给
   `Pi0Config` 加了 `image_keys`/`state_dim` 两个 vanilla openpi 没有的字段。
   这和这个分支现在的设计（触觉完全不进 pi0.5，只喂给下游 FRS 网络，pi0.5 只处理
   `left_wrist_0_rgb`/`right_wrist_0_rgb` 两路 RGB）是两条不同的路子。**已经确认维持现状，
   不跟进这个"触觉直喂 pi0.5"的模式**——如果以后要改成那样，`pi05_jax/pi0_config.py`/
   `pi0.py`/`model.py` 需要照 VB-VLA 的 fork 加 `image_keys`/`state_dim` 字段并在
   `inputs_spec`/`preprocess_observation` 调用处传下去，`modalities_eval/pi05_utils.py`
   的 `camera_map`/`IMAGE_KEYS` 假设也要跟着改（不再是固定 3 路 ALOHA 命名）。
   另外搜了一圈 VB-VLA 本地和部署脚本里的 checkpoint 路径，没有找到任何 pick_tube 专用的
   已训练 pi0.5 checkpoint（`pi05_chaoyi_vitac` 里的 "chaoyi" 只是人名，对应的是
   `liuchaoyi/...` 数据，不是 pick_tube）——真要走 VT-pi0.5 这条路，得先在训练服务器上
   实际跑一次微调，不是接个现成 checkpoint。

## 管线完整性盘点（2026-08-08）：还缺什么

把整条链路四步都对着 pi0.5 配置核了一遍：

| 步骤 | 工具 | 状态 |
| --- | --- | --- |
| 一键编排 | `scripts/start_frs_pi05_train.sh` | **本次新增**（原来只有 SmolVLA 版，而且会直接失败） |
| 1. 触觉 embedding | `tools/precompute_tactile_embeddings.py` | 现有工具直接可用，**已验证**不用改 |
| 2. action_cache | `tools/prepare_frs_pi05_cache.py` | 本次新增 |
| 3. FRS 训练 | `tools/train_frs.py` | 现有工具直接可用，**已验证**不用改 |
| 4. FRS 评估 | `train_pi05_frs/evaluate.py` | 现有工具直接可用（完全不依赖 base 模型） |

第 1、3 步能直接复用是核对过的：它们读的每个配置键 pi0.5 配置里都有；原来位于
`smolvla_jax` 的通用数据源解析和触觉 cache 已抽到 `lerobot/datasets/`，所以 pi0.5
运行时不再 import SmolVLA。第 4 步 `evaluate.py` 根本不 import 任何 base 模型代码，只读 cache。

原来的 `scripts/start_frs_train.sh` 对 pi0.5 配置会**在第 55 行直接退出**（它要求
`checkpoint_merge.adapter` 非空，pi0.5 没有这个概念），即使绕过也会去跑
`merge_smolvla_peft_to_jax.py` 和 SmolVLA 版的 `prepare_frs_caches.py`。新脚本去掉了
PEFT 合并这一步（pi0.5 直接用官方 checkpoint），并且在跑那两个很慢的阶段之前先做一次
`load_pi0()` 冒烟检查——这是 `pi05_jax/README.md` 验证清单的第 2 项，早失败比晚失败好。

**所以现在真正还缺的只有两类：**

1. **一行代码都没在真机上跑过**（最主要的缺口）。必须上有 GPU 的 Linux 训练服务器，按
   `src/lerobot/policies/pi05_jax/README.md` 的清单验证；新脚本已经把其中第 2 项
   （checkpoint 能否加载）内建成了前置检查。清单第 3 项（`denoise_step` 和 upstream
   `sample_actions` 数值是否一致）仍然需要手动做一次——那是这次唯一新写的模型级逻辑。
2. **没有移植的周边**（都不影响 FRS 训练本身，按需再做）。这些原本只有 SmolVLA 版，
   已随 SmolVLA 一起从本分支删除（见下面 2026-08-12 的清理记录，`git log` 里可找回）：
   - 真机部署客户端（原 `deploy_smolvla/`）。
   - loglike / action-error / t-SNE 分析脚本（原 `modalities_eval/` 下几个）。
     真要做的话 `modalities_eval/pi05_utils.py` 的 `Pi05EvalModel` 够复用。

## 明确不做的事

- 不 `pip install openpi`、不给它开单独环境——包名冲突的解法是 vendor 代码，不是隔离环境
  （之前 `openpi_bridge/` 那版方案已经废弃删除）。
- 不维护 SmolVLA 那条线——这个分支的目标就是换 base，不是两条线都要维护。相关代码已于
  2026-08-12 全部删除（见下）。
- 不在这台本地 macOS 机器上装 jax/flax/跑推理、不下载 pi0.5 checkpoint——这些必须在训练服务器
  （Linux + NVIDIA）上做，环境搭建见 [README.md](README.md) 的「快速开始」。

## 2026-08-12：删除 SmolVLA 整条线

这个分支现在只维护 pi0.5。删掉的（约 11000 行，`git log` 里都在）：
`src/lerobot/policies/smolvla_jax/`、`deploy_smolvla/`、`prepare.py`、`utils/source_model.py`、
`utils/utils.py`、`modalities_eval/utils.py` + 4 个分析脚本 + `modalities_eval/test/`、
`tools/` 下 6 个 SmolVLA 工具、4 个 SmolVLA config、`scripts/start_frs_train.sh` 和
`scripts/start_vtsmolvla_train.sh`、`tests/jax/` 下 9 个测试（只留 `test_tactile_cache.py`，
它测的是 `lerobot/datasets/tactile_cache.py`）。顺带删了没人引用的 `finalize_cache.py`、
`filter_cache.py`（`utils/cache.py:filter_cache_by_mse` 本体和它的测试都保留）、
`download_ckpt.py`，以及已被 README 取代的 `doc.md` / `train_for_agent.md`。

**顺带修掉一个真 bug**：`src/lerobot/policies/__init__.py` 原本
`from .smolvla_jax import JaxSmolVLA, ...`。因为 `import lerobot.policies.pi05_jax` 会先执行
父包的 `__init__.py`，**每一次 pi0.5 运行都在连带 import 整个 SmolVLA 模型栈**——而这个分支的
依赖是按 openpi 钉死的，本文件上面自己写着「smolvla_jax 大概率跑不起来」。一旦它 import 失败，
pi0.5 会在什么都没做之前就死。原来的 `tests/pi05/test_independence.py` 只 grep 几个具体运行时
文件，正好漏掉了这条包初始化路径；已改写成直接断言 `lerobot/policies/__init__.py` 不含任何
import，并覆盖全部共享阶段。

清理依赖：`websockets`/`msgpack`（原 `deploy_smolvla/` 用）、`scikit-learn`（原 t-SNE 脚本用）、
`contourpy`（原 loglike 画图用）现在都没有使用者了，从 `pyproject.toml` 移除并重新 `uv lock`
（159 个包，原 164 个）。`[tool.setuptools.packages.find]` 里的 `deploy*` glob 也一并去掉。
