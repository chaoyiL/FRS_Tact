# 双手 FRS 组合端点损失实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在完整保留现有 `gated` 损失和旧 checkpoint 语义的前提下，新增可由 YAML 选择的 `bimanual_gated` 组合端点损失。

**Architecture:** 新模式先从四个稳定触觉 token 生成 `[B,2]` 左右腕 Gate，再把 GT/VLA 的左右 10 维动作拼成每个样本唯一的 20 维 endpoint，只执行一次 FM。一次可微 FireFlow decode 由按手计算的 decode、low-safety、rank、repair 复用；旧 `gated` 分支和函数保持原行为。

**Tech Stack:** Python 3.11、JAX、Flax NNX、NumPy、PyYAML、pytest/unittest。

## Global Constraints

- 新 loss mode 固定命名为 `bimanual_gated`；旧 `gt`、`predicted`、`gated` 均保留。
- 当前双手契约固定为 20D：左手 `0:10`，右手 `10:20`；不得用任意偶数维自动二分。
- 左腕触觉 token 为索引 `(0,1)`，右腕为 `(2,3)`，对应 `_0` 与 `_1` 两组传感器。
- Decoder 输入和参数树不增加 Gate，继续使用 decoder input version 2。
- 新模式只训练一条 composite-endpoint FM；旧模式继续训练 GT/VLA 两条 FM。
- Decode、low-safety、rank、repair 均保留并按手计算；当前 `repair_weight: 0.0` 仍表示不产生梯度。
- 新模式禁止配置 `gate_lambda`；旧 `gated` 继续支持它。
- 实现遵循 TDD：每个任务先写失败测试，再实现，再运行回归测试。

---

## 文件结构

- 创建 `train_smolvla_frs/utils/bimanual_schema.py`：集中保存 mode/version、20D 切片、触觉分组与 metadata 校验。
- 修改 `train_smolvla_frs/utils/data.py`：提供按腕触觉变化与 conditioner 查询接口。
- 修改 `train_smolvla_frs/utils/model.py`：新增 composite endpoint、按手 MSE/辅助损失和 `train_step` 新分支；旧分支不重写。
- 修改 `train_smolvla_frs/utils/metrics.py`：新增按手验证数组与聚合指标。
- 修改 `train_smolvla_frs/train_frs.py`：接入新 mode、YAML 校验、日志、history 和 checkpoint metadata。
- 修改 `train_smolvla_frs/utils/history_plot.py`：绘制 `composite_fm` 并兼容旧 CSV。
- 保持 `train_smolvla_frs/configs/train_frs.yaml` 不变，并创建 `train_smolvla_frs/configs/train_frs_bimanual_gated.yaml` 作为新实验入口。
- 修改 `deploy_smolvla/frs_runtime.py`：允许并校验新 objective metadata，但不改变 decoder 输入。
- 修改 `train_smolvla_frs/README.md`：记录两种 gated 方案及切换方式。
- 修改 `tests/train_frs/test_data.py`、`tests/train_frs/test_model.py`、`tests/train_frs/test_evaluate.py`、`tests/train_frs/test_history_plot.py` 和 `tests/flow_decoder/test_frs_safety.py`：覆盖新行为与旧模式回归。

---

### Task 1: 双手 Schema 与按腕 Gate 数据接口

**Files:**
- Create: `train_smolvla_frs/utils/bimanual_schema.py`
- Modify: `train_smolvla_frs/utils/data.py:97-129,319-354,752-779`
- Test: `tests/train_frs/test_data.py:92-110,180-200`
- Test: `tests/train_frs/test_package_layout.py:1-190`

**Interfaces:**
- Produces: `tactile_change_per_wrist_from_tokens(current_tokens, baseline_tokens, *, wrist_token_indices=((0,1),(2,3))) -> np.ndarray[B,2]`
- Produces: conditioner method `tactile_change_per_wrist_for_cache_indices(cache_indices, current_tokens) -> np.ndarray[B,2]`
- Produces: `BIMANUAL_LOSS_MODE`, `BIMANUAL_OBJECTIVE_VERSION`, `LEFT_ACTION_SLICE`, `RIGHT_ACTION_SLICE`, `LEFT_WRIST_TOKEN_INDICES`, `RIGHT_WRIST_TOKEN_INDICES` 和 `validate_bimanual_objective_metadata(metadata)`。
- Existing scalar `tactile_change_from_tokens` and `gate_weights_for_cache_indices` remain unchanged.

- [ ] **Step 1: 写失败测试**

```python
def test_tactile_change_per_wrist_keeps_zero_and_one_groups_separate():
    baseline = np.zeros((1, 4, 2), dtype=np.float32)
    baseline[..., 0] = 1.0
    current = baseline.copy()
    current[:, 2:, 0] = -1.0
    change = tactile_change_per_wrist_from_tokens(current, baseline)
    np.testing.assert_allclose(change, [[0.0, 2.0]])

def test_tactile_change_per_wrist_rejects_non_four_token_input():
    with pytest.raises(ValueError, match="four tactile tokens"):
        tactile_change_per_wrist_from_tokens(
            np.zeros((1, 3, 2), np.float32), np.zeros((1, 3, 2), np.float32)
        )
```

- [ ] **Step 2: 验证测试先失败**

Run: `uv run --no-sync pytest -q tests/train_frs/test_data.py -k per_wrist`

Expected: collection/import failure because `tactile_change_per_wrist_from_tokens` does not exist.

- [ ] **Step 3: 实现最小按腕计算**

```python
WRIST_TOKEN_INDICES = (LEFT_WRIST_TOKEN_INDICES, RIGHT_WRIST_TOKEN_INDICES)

def tactile_change_per_wrist_from_tokens(current_tokens, baseline_tokens, *, wrist_token_indices=WRIST_TOKEN_INDICES):
    current = np.asarray(current_tokens, dtype=np.float32)
    baseline = np.asarray(baseline_tokens, dtype=np.float32)
    if current.shape != baseline.shape or current.ndim != 3 or current.shape[1] != 4:
        raise ValueError(f"Expected matching [B, four tactile tokens, D], got {current.shape} and {baseline.shape}.")
    current_n = _l2_normalize(current)
    baseline_n = _l2_normalize(baseline)
    per_token = 1.0 - np.sum(current_n * baseline_n, axis=-1)
    return np.stack([np.mean(per_token[:, indices], axis=1) for indices in wrist_token_indices], axis=1).astype(np.float32)
```

两个 conditioner 方法复用其现有 episode-baseline 查找逻辑：旧方法继续调用 `tactile_change_from_tokens`，新方法对同一组 `current/baseline` 数组调用 `tactile_change_per_wrist_from_tokens`；旧方法不能改名或改变返回 shape。

- [ ] **Step 4: 运行数据测试**

Run: `uv run --no-sync pytest -q tests/train_frs/test_data.py`

Expected: PASS，原 scalar Gate 测试和新 per-wrist 测试全部通过。

- [ ] **Step 5: 提交**

```bash
git add train_smolvla_frs/utils/bimanual_schema.py train_smolvla_frs/utils/data.py tests/train_frs/test_data.py tests/train_frs/test_package_layout.py
git commit -m "feat: add per-wrist FRS gate labels"
```

---

### Task 2: Composite endpoint 与单次 FM

**Files:**
- Modify: `train_smolvla_frs/utils/model.py:19-29,450-470,827-1079,1082-1264`
- Test: `tests/train_frs/test_model.py:240-290,657-780,843-970`

**Interfaces:**
- Consumes: `BIMANUAL_ACTION_DIM = 20`、`LEFT_ACTION_SLICE`、`RIGHT_ACTION_SLICE` from `utils/bimanual_schema.py`
- Produces: `bimanual_composite_endpoint(gt_action, predicted_action, gate_weights, *, low_gate_threshold, high_gate_threshold) -> tuple[target, effective_gates]`
- Produces: `bimanual_loss_components_per_sample(model, x_base, gt_action, predicted_action, t, tactile_seq, gate_weights, **loss_options) -> dict[str, Array]`
- `train_step` 在 `loss_mode="bimanual_gated"` 时消费 `[B,2]` Gate。
- Existing `gated_loss_components_per_sample` remains byte-for-byte behavior-compatible.

- [ ] **Step 1: 写 endpoint 与旧模式回归失败测试**

```python
def test_bimanual_composite_endpoint_selects_each_hand_independently():
    gt = jnp.arange(40, dtype=jnp.float32).reshape(1, 2, 20)
    vla = -gt
    target, effective = bimanual_composite_endpoint(
        gt, vla, jnp.asarray([[1.0, 0.0]]),
        low_gate_threshold=0.3, high_gate_threshold=0.7,
    )
    np.testing.assert_allclose(target[..., :10], gt[..., :10])
    np.testing.assert_allclose(target[..., 10:], vla[..., 10:])
    np.testing.assert_allclose(effective, [[1.0, 0.0]])

def test_bimanual_composite_endpoint_rejects_nonfinite_gate():
    actions = jnp.zeros((1, 2, 20), dtype=jnp.float32)
    with pytest.raises(ValueError, match="finite"):
        bimanual_composite_endpoint(
            actions, actions, jnp.asarray([[jnp.nan, 0.0]]),
            low_gate_threshold=0.3, high_gate_threshold=0.7,
        )

```

已有 `test_gated_loss_components_sum_to_total` 原样保留，继续断言旧函数只返回 `gt_fm/vla_fm/low_safety/decode/rank/repair` 六项。

- [ ] **Step 2: 验证 endpoint 测试失败、旧测试仍通过**

Run: `uv run --no-sync pytest -q tests/train_frs/test_model.py -k 'composite_endpoint or gated_loss_components_sum'`

Expected: endpoint import failure；现有 gated 测试 PASS。

- [ ] **Step 3: 实现 composite endpoint**

```python
def bimanual_composite_endpoint(gt_action, predicted_action, gate_weights, *, low_gate_threshold=0.3, high_gate_threshold=0.7):
    if gt_action.shape != predicted_action.shape or gt_action.shape[-1] != BIMANUAL_ACTION_DIM:
        raise ValueError("bimanual composite endpoint requires matching 20D actions")
    if gate_weights.shape != (gt_action.shape[0], 2):
        raise ValueError(f"bimanual gate_weights must have shape {(gt_action.shape[0], 2)}, got {gate_weights.shape}")
    effective = three_region_effective_gate_weights(
        gate_weights, low_gate_threshold=low_gate_threshold, high_gate_threshold=high_gate_threshold
    )
    per_dim = jnp.concatenate(
        [jnp.repeat(effective[:, :1], 10, axis=1), jnp.repeat(effective[:, 1:], 10, axis=1)], axis=1
    )[:, None, :]
    return per_dim * gt_action + (1.0 - per_dim) * predicted_action, effective
```

- [ ] **Step 4: 写单次 FM 失败测试**

使用现有小模型 fixture，将 `model.__call__` 的可观察计数替身设为返回零速度；断言 `bimanual_loss_components_per_sample` 的 `composite_fm` 等于 `mean(square(target-x_base))`，并且不返回非零 `gt_fm`/`vla_fm`。

- [ ] **Step 5: 实现新分支最小 FM**

```python
target, effective = bimanual_composite_endpoint(
    gt_action,
    predicted_action,
    gate_weights,
    low_gate_threshold=rank_low_gate_threshold,
    high_gate_threshold=rank_high_gate_threshold,
)
flow = flow_matching_loss_per_sample(model, x_base, target, t, tactile_seq, state=state, state_keep_mask=state_keep_mask)
components = {
    "gt_fm": jnp.zeros_like(flow),
    "vla_fm": jnp.zeros_like(flow),
    "composite_fm": flow,
    "low_safety": jnp.zeros_like(flow),
    "decode": jnp.zeros_like(flow),
    "rank": jnp.zeros_like(flow),
    "repair": jnp.zeros_like(flow),
}
```

保留现有 `LOSS_COMPONENT_NAMES` 六项不变，新增 `TRAIN_LOSS_COMPONENT_NAMES = ("gt_fm", "vla_fm", "composite_fm", "low_safety", "decode", "rank", "repair")`。`gt`、`predicted`、旧 `gated` 的 `train_step` component 字典显式增加 `composite_fm=0`，但旧的 `gated_loss_components_per_sample` 仍只返回原六项；新 `bimanual_gated` 分支只调用新函数。

- [ ] **Step 6: 运行模型核心测试**

Run: `uv run --no-sync pytest -q tests/train_frs/test_model.py -k 'composite or gated_loss or train_step'`

Expected: PASS；旧 gated 数值断言不变，新 composite FM 通过。

- [ ] **Step 7: 提交**

```bash
git add train_smolvla_frs/utils/model.py tests/train_frs/test_model.py
git commit -m "feat: add bimanual composite FRS flow loss"
```

---

### Task 3: 按手 Decode、Safety、Rank、Repair

**Files:**
- Modify: `train_smolvla_frs/utils/model.py:563-824,827-1079`
- Test: `tests/train_frs/test_model.py:598-780,843-970`

**Interfaces:**
- Produces: `bimanual_mse_per_sample(decoded, endpoint) -> Array[B,2]`
- Produces: `_average_active_wrist_terms(left_term, left_active, right_term, right_active) -> Array[B]`
- Completes: `bimanual_loss_components_per_sample` seven-term dictionary.

- [ ] **Step 1: 写按手辅助项失败测试**

构造 `decoded=[GT_L,VLA_R]`、`gates=[[1,0]]`，断言：

```python
assert components["decode"][0] == pytest.approx(0.0)
assert components["rank"][0] == pytest.approx(0.0)
assert components["repair"][0] == pytest.approx(0.0)
```

再只破坏右手 decoded，断言 `decode>0`，但左手 rank/repair 数值不改变；只破坏左手时作镜像断言。增加 batch 中仅左腕 high-active 的测试，断言 rank 不会因右腕空组除以 2。

- [ ] **Step 2: 验证测试失败**

Run: `uv run --no-sync pytest -q tests/train_frs/test_model.py -k bimanual_aux`

Expected: FAIL，因为新辅助项仍为零。

- [ ] **Step 3: 实现按手 MSE 与 direct decode**

```python
def bimanual_mse_per_sample(left, right):
    squared = jnp.square(left - right)
    return jnp.stack(
        [jnp.mean(squared[..., :10], axis=(1, 2)), jnp.mean(squared[..., 10:], axis=(1, 2))], axis=1
    )

mse_gt = bimanual_mse_per_sample(decoded, gt_action)
mse_vla = bimanual_mse_per_sample(decoded, predicted_action)
decode_term = float(aux_decode_weight) * jnp.mean(
    effective * mse_gt + (1.0 - effective) * mse_vla, axis=1
)
```

- [ ] **Step 4: 实现 thresholded 辅助项**

对每个 wrist 分别调用现有 `_active_group_normalized_per_sample` 或 source-aware 对应 helper：

```python
low_penalty = jax.nn.relu(jnp.minimum(mse_gt, mse_vla) - low_gate_safety_margin)
rank_penalty = jax.nn.relu(mse_gt - mse_vla + rank_margin)
baseline = bimanual_mse_per_sample(predicted_action, gt_action)
repair_penalty = jax.nn.relu(mse_gt - baseline + repair_margin)
low_strength = (1.0 - raw_gates) * (raw_gates <= rank_low_gate_threshold)
high_strength = raw_gates * (raw_gates >= rank_high_gate_threshold)
```

每个 penalty 的两列独立归一化，再用活跃 wrist 数求平均。`worst_source_cvar` 对左右列分别调用 `high_gate_worst_source_cvar_loss`，最后只平均 active wrist scalar。

- [ ] **Step 5: 运行辅助项与完整模型测试**

Run: `uv run --no-sync pytest -q tests/train_frs/test_model.py`

Expected: PASS；旧六项公式和新七项公式都满足 total 等于分项和。

- [ ] **Step 6: 提交**

```bash
git add train_smolvla_frs/utils/model.py tests/train_frs/test_model.py
git commit -m "feat: add per-wrist FRS auxiliary losses"
```

---

### Task 4: 训练配置、日志与兼容性

**Files:**
- Modify: `train_smolvla_frs/train_frs.py:25,324-590,693-785,980-1238,1620-1975,2048-2206`
- Create: `train_smolvla_frs/configs/train_frs_bimanual_gated.yaml`
- Modify: `train_smolvla_frs/utils/history_plot.py`
- Modify: `train_smolvla_frs/README.md:44-73`
- Test: `tests/train_frs/test_model.py:190-290`
- Test: `tests/train_frs/test_history_plot.py`
- Test: `tests/train_frs/test_package_layout.py`

**Interfaces:**
- Extends: `LossMode = Literal["gt", "predicted", "gated", "bimanual_gated"]`
- History adds `train_loss_composite_fm`, `train_gate_w_left`, `train_gate_w_right`.
- Checkpoint `extra_metadata` adds `loss_objective_version=2`, `action_slices={"left":[0,10],"right":[10,20]}`, `wrist_token_indices={"left":[0,1],"right":[2,3]}`.

- [ ] **Step 1: 写 config 与旧模式兼容失败测试**

```python
def test_bimanual_mode_rejects_gate_lambda():
    config = minimal_config(loss_mode="bimanual_gated")
    config["frs_training"]["gate_lambda"] = 0.25
    with pytest.raises(ValueError, match="gate_lambda.*bimanual_gated"):
        train_from_config(config)

def test_legacy_gated_mode_still_accepts_gate_lambda():
    assert parse_loss_settings(minimal_config(loss_mode="gated", gate_lambda=0.25)).gate_lambda == 0.25
```

- [ ] **Step 2: 扩展 mode 与训练 batch Gate**

`loss_mode == "gated"` 继续调用 scalar 方法；新模式调用：

```python
change_per_wrist = conditioner.tactile_change_per_wrist_for_cache_indices(indices, current_tokens)
gate_w = gate_weights_from_change(change_per_wrist, tau=gate_tau, temperature=gate_temperature)
batch_gate_w_left = float(np.mean(gate_w[:, 0]))
batch_gate_w_right = float(np.mean(gate_w[:, 1]))
```

所有需要 episode baseline 的判断改为 `loss_mode in ("gated", "bimanual_gated")`。复制当前正式配置生成 `train_frs_bimanual_gated.yaml`，只把 `loss_mode` 改为 `bimanual_gated`、删除 `gate_lambda` 并使用新的 output 目录；原 `train_frs.yaml` 保持不变。

- [ ] **Step 3: 扩展 component/history/checkpoint metadata**

`component_losses` 增加 `composite_fm`，每个旧 mode 记录 0。CSV 和 plot 使用字段：

```python
"train_loss_composite_fm",
"train_gate_w_left",
"train_gate_w_right",
```

Resume 校验：`bimanual_gated` 必须完全匹配 objective version、动作切片、触觉索引；旧 `gated` metadata 路径保持不变。

- [ ] **Step 4: 更新 README 与历史图测试**

README 明确给出：

```yaml
loss_mode: gated           # 旧 scalar-gate 双 FM
loss_mode: bimanual_gated  # 新 per-wrist composite endpoint FM
```

历史图测试分别构造旧 CSV（无 composite 字段）和新 CSV，断言二者都能输出 PNG。

- [ ] **Step 5: 运行训练入口测试**

Run: `uv run --no-sync pytest -q tests/train_frs/test_model.py tests/train_frs/test_history_plot.py tests/train_frs/test_package_layout.py`

Expected: PASS；旧 mode 与新 mode 均被 CLI/YAML 接受，旧 history 仍可绘制。

- [ ] **Step 6: 提交**

```bash
git add train_smolvla_frs/train_frs.py train_smolvla_frs/configs/train_frs_bimanual_gated.yaml train_smolvla_frs/utils/history_plot.py train_smolvla_frs/README.md tests/train_frs/test_model.py tests/train_frs/test_history_plot.py tests/train_frs/test_package_layout.py
git commit -m "feat: wire bimanual FRS loss mode"
```

---

### Task 5: 按手验证指标与 checkpoint 选择

**Files:**
- Modify: `train_smolvla_frs/utils/metrics.py:28-80,129-270,287-514`
- Modify: `train_smolvla_frs/evaluate.py`
- Modify: `train_smolvla_frs/train_frs.py:95-190,1270-1560`
- Test: `tests/train_frs/test_evaluate.py`
- Test: `tests/flow_decoder/test_frs_safety.py`

**Interfaces:**
- Validation result adds `sample_gate_w_left/right`, `sample_mse_gt_left/right`, `sample_mse_vla_left/right`, and per-wrist low/rank/repair aggregates.
- `checkpoint_selection_key` in new mode compares feasibility of both wrists and then the worse wrist；old mode key stays unchanged.

- [ ] **Step 1: 写 per-wrist metrics 失败测试**

构造两个样本：第一个仅左腕 high，第二个仅右腕 high；令其中一只手 GT 改善、另一只手退化。断言聚合结果分别报告两只手，不允许完整 20D 平均抵消。

- [ ] **Step 2: 实现验证数组**

新模式使用 `gate_weights[B,2]`，并从动作误差 `[B,H,20]` 计算：

```python
mse_gt_left = np.mean(np.square(prediction[..., :10] - gt_action[..., :10]), axis=(1, 2))
mse_gt_right = np.mean(np.square(prediction[..., 10:] - gt_action[..., 10:]), axis=(1, 2))
mse_vla_left = np.mean(np.square(prediction[..., :10] - predicted_action[..., :10]), axis=(1, 2))
mse_vla_right = np.mean(np.square(prediction[..., 10:] - predicted_action[..., 10:]), axis=(1, 2))
```

保留现有完整动作指标用于跨版本总体比较，新增字段用于安全判断。

- [ ] **Step 3: 实现 new-mode checkpoint key**

新 key 首项为左右腕所有约束 violation 之和，随后依次比较：较差 wrist 的 high-gate gain、较差 wrist 的 rank satisfaction、较差 wrist 的 low preservation、总体 GT MSE。`loss_mode="gated"` 继续走现有 key，不改变已有测试期望。

- [ ] **Step 4: 运行评估与安全测试**

Run: `uv run --no-sync pytest -q tests/train_frs/test_evaluate.py tests/flow_decoder/test_frs_safety.py`

Expected: PASS；单手退化不能被另一手改善掩盖，旧 gated checkpoint key 测试不变。

- [ ] **Step 5: 提交**

```bash
git add train_smolvla_frs/utils/metrics.py train_smolvla_frs/evaluate.py train_smolvla_frs/train_frs.py tests/train_frs/test_evaluate.py tests/flow_decoder/test_frs_safety.py
git commit -m "feat: add per-wrist FRS validation metrics"
```

---

### Task 6: 全量回归、静态检查与文档核对

**Files:**
- Modify: `deploy_smolvla/frs_runtime.py:539-570`
- Test: `tests/jax/test_frs_deployment.py:2230-2280`
- Modify other files only when a verification failure identifies a defect in files already listed above.

**Interfaces:**
- Produces a verified `bimanual_gated` training path without changing legacy inference/checkpoint parameter trees，并让部署契约接受且严格校验新 objective metadata。

- [ ] **Step 1: 写部署 metadata 失败测试并实现 validator 接入**

参数化旧 `gated` 与新 `bimanual_gated` checkpoint metadata，断言二者均可通过；把新 metadata 的 action slice 或 wrist token index 改错，断言 `_validate_contract` 抛出包含具体字段名的 `ValueError`。实现只调用 `validate_bimanual_objective_metadata`，不向 decoder 或 `decode_actions` 增加 Gate 参数。

- [ ] **Step 2: 运行 FRS 全套测试**

Run: `uv run --no-sync pytest -q tests/train_frs tests/flow_decoder`

Expected: PASS。

- [ ] **Step 3: 运行 JAX 部署兼容测试**

Run: `uv run --no-sync pytest -q tests/jax/test_frs_deployment.py tests/jax/test_tactile_integration.py`

Expected: PASS；decoder input version 2、参数树和部署 decode 均未改变。

- [ ] **Step 4: 运行格式和差异检查**

Run: `ruff check train_smolvla_frs tests/train_frs tests/flow_decoder`

Expected: PASS。

Run: `git diff --check`

Expected: 无输出，退出码 0。

- [ ] **Step 5: 核对新 YAML 可解析且旧 YAML 未变**

Run: `uv run --no-sync python -c "import yaml; old=yaml.safe_load(open('train_smolvla_frs/configs/train_frs.yaml')); new=yaml.safe_load(open('train_smolvla_frs/configs/train_frs_bimanual_gated.yaml')); assert old['frs_training']['loss_mode']=='gated'; assert 'gate_lambda' in old['frs_training']; assert new['frs_training']['loss_mode']=='bimanual_gated'; assert 'gate_lambda' not in new['frs_training']"`

Expected: 无输出，退出码 0。

- [ ] **Step 6: 最终提交**

```bash
git add train_smolvla_frs tests/train_frs tests/flow_decoder tests/jax
git commit -m "test: verify bimanual FRS loss compatibility"
```
