# FRS base model 切换：SmolVLA → pi0.5（JAX，vendored from openpi）

分支：`pi05-frs-jax`（从 `eric` 切出）。目标：FRS（`tactile_flow_steering`）不再使用 SmolVLA，
全部按 pi0.5（Physical Intelligence 的 openpi，JAX 原生实现）的要求来——包括主环境的
jax/flax/transformers/orbax 版本，不用兼顾 SmolVLA 还能不能跑。这个文件记录已确认的架构决策、
关键发现和还没做完的事，避免跨 session 丢上下文。

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
`>=2.0.0,<2.3.0` 不变——这是已知的、还没验证的风险点：`jax==0.5.3`/`flax==0.10.2`
到底能不能在 numpy>=2 下正常跑，要在训练服务器上 `uv sync` 才能确认。

## 现状：整条链路的代码都写完了，一行都没跑过

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
    导出，其他地方的 import 不用改）和 `utils/flow_matching.py`（`deterministic_noise`/
    `inversion_mse`，从 `utils/source_model.py` 挪出来，同样保留重新导出）。
- 修了一个真实 bug（不是 TODO，是实际改对了）：`prepare_pi05.py`/
  `tools/prepare_frs_pi05_cache.py` 一开始把 `gs://...` checkpoint URL 直接传给
  `pathlib.Path(...)`，Python 的 `Path` 会把 URL 里的 `//` 归一化成单个 `/`
  （`Path("gs://a/b")` 变成 `"gs:/a/b"`），下载会失败。现在用 `_is_local_path`
  （基于 `urllib.parse.urlparse`）先判断是不是 URL，是的话全程留在字符串形态，
  只有确认是本地路径才包一层 `Path`。

**完全没跑过、必须在训练服务器上验证的（`src/lerobot/policies/pi05_jax/README.md` 里也写了）：**

1. `uv sync` 能不能正常解析出一套依赖（尤其是 numpy 版本风险，见上面"架构决策"）。
2. `load_pi0("gs://openpi-assets/checkpoints/pi05_base")` 加载出来的参数形状能不能对上
   `Pi0Config(pi05=True)`（对不上 `BaseModelConfig.load` 会直接报错，容易发现）。
3. **`build_prefix_cache`/`denoise_step` 拆分对不对**——这是这次唯一真正新写的模型级逻辑
   （其余都是原样 vendor），最值得单独验证：用同一份输入分别跑 upstream 原版
   `sample_actions`（t:1→0）和"手动逐步调 denoise_step"，两边应该给出完全一样的结果。
4. `Pi05EvalModel.prepare_sample` 拼出来的 `Observation`/`Pi0PrefixCache` 在
   `nnx.split`/`nnx.merge` + `jax.jit`（`utils/pi05_source_model.py`）下能不能正常跑通一次
   完整的 `sample_and_reverse`，shape 对不对。

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
（参考 openpi 的 `scripts/compute_norm_stats.py`，本仓库没 vendor 这个脚本）。

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

## 明确不做的事

- 不 `pip install openpi`、不给它开单独环境——包名冲突的解法是 vendor 代码，不是隔离环境
  （之前 `openpi_bridge/` 那版方案已经废弃删除）。
- 不管 `smolvla_jax/` 能不能跑、不管 `configs/train_smolvla_jax.yaml` /
  `configs/train_vtsmolvla_jax.yaml` 还有没有用——这个分支的目标就是换 base，不是两条线都要维护。
- 不在这台本地 macOS 机器上装 jax/flax/跑推理、不下载 pi0.5 checkpoint——这些必须在训练服务器
  （Linux + NVIDIA）上做，参考 [train_for_agent.md](train_for_agent.md) 的环境搭建流程。
