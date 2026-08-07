# openpi_bridge

独立的 uv 子项目：只用来跑 pi0.5（[openpi](https://github.com/Physical-Intelligence/openpi)，
JAX 原生实现）推理，为 FRS 产出 action_cache。**故意**不并入主仓库的 `pyproject.toml` ——
为什么这么做、还差什么，见 [../pi05_frs_plan.md](../pi05_frs_plan.md)。

## 这是什么 / 不是什么

- 是：一个独立的 venv，装 openpi + numpy，跑一次性脚本，把 pi0.5 在 pick_tube 数据集上的
  预测动作 + 反向积分 latent 存成和 `tools/prepare_frs_caches.py` 输出格式完全一致的
  action_cache（`utils/cache.py` 定义的 memmap + `manifest.json`）。
- 不是：不需要跟主环境的 jax/flax/torch 版本兼容，不会被主仓库的训练代码 import。
  产出的 action_cache 落盘后，主环境的 `tools/train_frs.py` 直接读，不关心是谁生成的。

## 安装（要在能装 `jax[cuda12]` 的 Linux + NVIDIA 机器上跑，这台 Mac 装不了）

```bash
cd openpi_bridge
uv sync
```

`uv sync` 目前还没有实际跑过验证 —— openpi 自己是一个 uv workspace（`packages/openpi-client`
是 workspace member），从外部把它当普通 git 依赖装，第一次跑大概率要调整
（比如指定 `subdirectory=`，或者干脆 `git clone` + `uv pip install -e .` 而不是走
`[tool.uv.sources]`）。遇到问题按报错调整 `pyproject.toml`，不要死磕当前写法。

## 使用

```bash
uv run python prepare_pi05_cache.py --config ../configs/train_frs_pick_tube_pi05.yaml
```

`prepare_pi05_cache.py` 现在还是骨架，核心的 pi0.5 推理逻辑标了 `TODO` / `NotImplementedError`，
具体缺什么见文件内注释和 `../pi05_frs_plan.md`。
