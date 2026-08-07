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

## 现状：模型代码已经搬进来了，checkpoint 加载/推理链路完全没跑过

已经做的（都在 [src/lerobot/policies/pi05_jax/](src/lerobot/policies/pi05_jax/)，
细节和取舍见该目录的 [README.md](src/lerobot/policies/pi05_jax/README.md)）：

- 从 openpi vendor 了 pi0.5 需要的模型代码：`model.py`（去掉了 PyTorch checkpoint 加载分支）、
  `pi0.py`、`pi0_config.py`、`gemma.py`、`siglip.py`、`lora.py`、`tokenizer.py`（只留
  `PaligemmaTokenizer`）、`array_typing.py`、`nnx_utils.py`、`image_tools.py`、`download.py`。
- 新写了 `sharding.py`（单机推理用的 no-op 版 `openpi.training.sharding`，不是 vendor 来的）
  和 `checkpoint.py`（`load_pi0()`，加载原生 JAX/orbax pi0.5 checkpoint）。
- 在 `pi0.py` 里新增了 `Pi0PrefixCache` / `Pi0.build_prefix_cache` / `Pi0.denoise_step`——
  这是 upstream 没有的：FRS 需要在任意 `(x, t)` 上算 flow-matching 速度场来做反向 ODE 积分
  （`utils/integration.py` 的 euler/fireflow，SmolVLA 那边对应 `utils/source_model.py` 的
  `sample_and_reverse`），而 upstream 的 `sample_actions` 只有一个写死方向的 t:1→0 前向循环，
  单步逻辑内联在闭包里拿不出来。这三个新增是把那段闭包逻辑原样拆出来，`sample_actions`
  本身一个字节没动。
- 主 `pyproject.toml` 依赖版本按 pi0.5 的要求改了（见上面"架构决策"）。
- 删掉了之前"独立环境（openpi_bridge/）"那版方案的所有文件——想清楚包名冲突之后，
  vendor 代码进来是更合适的路线，独立环境那套不再需要。

**完全没做、且没法在这台机器上验证的（`src/lerobot/policies/pi05_jax/README.md` 里也写了）：**

1. **一行代码都没跑过。** 这台开发机是 macOS、没装 jax/flax/GPU（`jax[cuda12-local]`
   只能在 Linux+NVIDIA 上装），vendor 过程中只做了 `python -m py_compile`（纯语法检查），
   没有导入过、更没有拿真实 checkpoint 跑过。上训练服务器后必须先验证：
   - `uv sync` 能不能正常解析出一套依赖（尤其是上面提到的 numpy 版本风险）；
   - `load_pi0("gs://openpi-assets/checkpoints/pi05_base")` 加载出来的参数形状能不能对上
     `Pi0Config(pi05=True)`（对不上 `BaseModelConfig.load` 会直接报错，容易发现）；
   - 新增的 `build_prefix_cache`/`denoise_step` 拆分对不对——用同一份输入分别跑 upstream
     原版 `sample_actions`（t:1→0）和"手动逐步调 denoise_step"，两边应该给出完全一样的结果。
     这是这次唯一真正新写的逻辑（其余都是原样 vendor），所以是最值得单独验证的一处。
2. **pick_tube 数据集样本 → pi0.5 `Observation` 的映射还没写。** 包括：哪个相机对应
   `base_0_rgb`/`left_wrist_0_rgb`/`right_wrist_0_rgb`、prompt 文本用什么；尤其要注意
   pi0.5（`pi05=True`）和 pi0 不一样——**state 不是连续输入，而是离散化后拼进 tokenized
   prompt**（`PaligemmaTokenizer.tokenize(prompt, state=state)`，见 `tokenizer.py` 顶部注释和
   `pi0.py:embed_suffix` 里 `if not self.pi05: state_token = ...`），漏掉这一步会让 state
   完全不起作用。
3. **归一化没做。** openpi 官方 `Policy` 会从 checkpoint 的 `assets/` 目录加载 norm stats
   （`openpi.training.checkpoints.load_norm_stats`，没有 vendor 进来）做
   `Normalize`/`Unnormalize`。FRS 这边需要等价的东西，否则 state/action 的数值范围对不上
   `pi05_base` 训练时的假设。
4. **反向积分的胶水代码没写。** 需要把 `build_prefix_cache`/`denoise_step` 接到
   `utils/integration.py` 的 euler/fireflow 求解器上，产出 action_cache——对应 SmolVLA 那边
   `utils/source_model.py` 的 `sample_and_reverse`/`reverse_integrate_actions`。
5. **没有对应 `prepare.py`/`tools/prepare_frs_caches.py` 的 pi0.5 版工具**，把上面几步串起来、
   按 `utils/cache.py` 定义的格式落盘（这样 `tools/train_frs.py` 就能直接读，不用改）。

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

## 明确不做的事

- 不 `pip install openpi`、不给它开单独环境——包名冲突的解法是 vendor 代码，不是隔离环境
  （之前 `openpi_bridge/` 那版方案已经废弃删除）。
- 不管 `smolvla_jax/` 能不能跑、不管 `configs/train_smolvla_jax.yaml` /
  `configs/train_vtsmolvla_jax.yaml` 还有没有用——这个分支的目标就是换 base，不是两条线都要维护。
- 不在这台本地 macOS 机器上装 jax/flax/跑推理、不下载 pi0.5 checkpoint——这些必须在训练服务器
  （Linux + NVIDIA）上做，参考 [train_for_agent.md](train_for_agent.md) 的环境搭建流程。
