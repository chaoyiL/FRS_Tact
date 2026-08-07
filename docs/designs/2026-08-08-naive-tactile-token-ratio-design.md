# Naive tactile token ratio baseline 设计

日期：2026-08-08
状态：用户已批准核心方案，等待书面 spec 复核

## 目标

在现有 JAX VT-SmolVLA 中，以最简单、零新增参数、最容易复现的方式提高 tactile token 在 conditioning prefix 中的占比，形成论文 baseline：

- 原始：每路 tactile view 1 token；四路共 4 token。
- 约 16%：每路 8 tokens；四路共 32 token。
- 约 32%：每路 21 tokens；四路共 84 token。

“总 token”固定定义为 conditioning prefix：两路 RGB、tactile、最大 48 个 language slots 和 1 个 state token；不包含 flow/action suffix。

这里报告的是固定张量中的 allocated prefix slot 占比，便于所有样本和实验使用同一口径。language padding 虽会被 attention mask 屏蔽，但仍占张量长度；若另报有效 token 占比，应使用 `4K / (128 + 4K + L_valid + 1)`，并同时给出 `L_valid`，不能与本设计的 15.31%/32.18% 混用。

## 已批准的方法

保持现有冻结 ResNet18、缓存格式和共享 `tactile_proj(512 -> 960)` 不变。每路投影后的 token 按 key-major 顺序连续复制 K 次，然后直接拼入原 prefix：

```text
[RGB1 x64, RGB2 x64, tactile_key0 xK, tactile_key1 xK,
 tactile_key2 xK, tactile_key3 xK, language x48, state x1]
```

数学形式：

```text
base = tactile_proj(rms_norm(resnet_embedding))  # [B,4,960]
expanded = repeat_interleave(base, K, axis=1)   # [B,4K,960]
expanded_mask = repeat_interleave(mask, K)      # [B,4K]
```

不新增 modality/type embedding、copy-slot embedding 或 learned position embedding。复制 token 继续使用模型现有的连续 RoPE position。

## 配额

非触觉 prefix slots 固定为 `2*64 + 48 + 1 = 177`。

| 配置 | K | tactile tokens | prefix slots | tactile/prefix |
|---|---:|---:|---:|---:|
| 原始 | 1 | 4 | 181 | 2.21% |
| tactile16 | 8 | 32 | 209 | 15.31% |
| tactile32 | 21 | 84 | 261 | 32.18% |

选择 K=8 和 K=21 是为了保持四路传感器完全对称。追求绝对 16.00%/32.00% 会要求给不同传感器分配不同 token 数，可能引入左右手或传感器偏置，因此不采用。

## 配置契约

新增：

```yaml
model:
  tactile_token_repeat_factor: 1
```

- 类型：正整数。
- 默认值：1。
- `tactile_num_tokens` 保留原语义，仍等于 tactile keys/缓存 streams 数量 4。
- effective transformer tactile tokens 为：

```text
tactile_num_tokens * tactile_token_repeat_factor
```

- 旧 checkpoint/config 缺新字段时自动按 1 读取。
- factor 为 0、负数、非整数时启动阶段 fail closed。
- 新字段写入 effective `config.json` 和训练 resume signature。
- 旧 resume metadata 缺字段时 canonicalize 为 1；K 不同的 checkpoint 禁止 strict optimizer resume，但允许作为权重 warm-start 并开启新 run。

## 数据、缓存和部署兼容性

- raw tactile 仍为 `[B,4,H,W,C]`。
- tactile cache v1 仍为 `[frames,4,512]`，无需重建。
- expansion 只发生在共享 projection 之后，因此 live image 和 cached embedding 路径保持一致。
- policy inference、training、validation rollout 和 `modalities_eval` 都复用模型内部 expansion。
- `modalities_eval` 仍传入 `[B,4]` sensor mask；模型内部同步扩展为 `[B,4K]`。tactile ablation 将四路原始 mask 全部置 False，扩展后自然屏蔽全部 4K tokens。
- 不设置 `prefix_length`；当前动态 prefix/mask/KV/RoPE 路径支持增长，避免引入固定长度保存问题。

## 论文实验配置

保留原 `configs/train_vtsmolvla_jax.yaml` 作为 K=1。新增两份明确配置：

- `configs/train_vtsmolvla_jax_tactile16.yaml`：K=8，独立 output、wandb name/tag。
- `configs/train_vtsmolvla_jax_tactile32.yaml`：K=21，独立 output、wandb name/tag。

除 factor、output identity 和实验标签外，两份配置必须与 K=1 保持一致。自动测试/校验应检查不存在其他超参漂移。

正式论文比较应让 K=1/8/21 从相同 base initialization、相同 v3 数据与 split、normalizer、seed、batch/effective batch、steps、optimizer/scheduler、LoRA/module modes、augmentation 和 modality dropout 设置开始。已有 `vtsmolvla_01_3w` 可用于兼容 smoke，但不能把不同 continuation 当成严格公平 baseline。

## 计算开销

相对 K=1 的最大 prefix self-attention 元素量近似：

- K=8：`209^2 / 181^2 = 1.33x`。
- K=21：`261^2 / 181^2 = 2.08x`。

训练报告应同时记录 step time、tokens/s、峰值显存和是否因显存调整 batch。若调整实际 batch，必须保持相同 effective batch，或在论文中单列说明。

## 测试与验收

### Config/checkpoint

- 旧视觉 config 和旧 VT config 缺字段时为 K=1。
- K=1/8/21 effective config round-trip。
- 0、负数、非整数拒绝。
- 新字段写入 checkpoint config。
- resume metadata 缺字段迁移为 1；不同 K strict resume 拒绝。

### Tensor contract

- S=4、K=1 输出逐值和 shape 与旧实现一致。
- K=8 输出 `[B,32,960]`、mask `[B,32]`。
- K=21 输出 `[B,84,960]`、mask `[B,84]`。
- 顺序严格为 key-major/token-minor。
- raw image 与 cached embedding 两路 expansion 一致。
- tactile ablation 后全部 expanded masks 为 False。

### Model smoke

- K=1/8/21 train loss finite。
- K=1/8/21 jitted sample finite。
- checkpoint save/load 后 factor 与输出 shape 保持。
- 在可用 GPU 上记录三个 K 的峰值显存和单步耗时。

## 结果解释边界

复制不会增加新的触觉观测信息。它改变的是 token multiplicity、连续 RoPE positions 和 attention mass；理想化相同 logits 下，K 个副本近似给该组增加 `log(K)` attention bias。论文中将其称为 `naive repeat-and-concat tactile token allocation baseline`，不得描述为 richer tactile representation。

## 非目标

- 不从 ResNet spatial feature map生成新 token。
- 不加入 temporal tactile history。
- 不把 tactile 当普通 SmolVLM image slots。
- 不增加可训练 token expander、slot embedding 或 modality embedding。
- 不修改 tactile cache v1 或 FRS 的四路 sensor-stream contract。
