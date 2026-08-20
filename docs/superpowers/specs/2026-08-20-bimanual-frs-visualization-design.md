# 双手 FRS 训练与评估可视化设计

## 目标

在不改变 `bimanual_gated` 训练目标、checkpoint 选择规则和旧版输出的前提下，新增一套直观的双手可视化，同时回答四个问题：

1. 训练与验证是否正常收敛；
2. 高 Gate 手是否从 VLA 向 GT 修正；
3. 低 Gate 手是否保持原始 VLA，而不是被另一只手带偏；
4. 左高右低、左低右高、双高和双低四种场景是否都有足够验证样本。

## 输出文件

训练输出目录继续保留现有 `training_curves.png`，并新增：

- `training_overview.png`：整体收敛、解码误差和约束总览；
- `bimanual_behavior.png`：四种左右手 Gate 组合下的逐手行为；
- `gate_diagnostics.png`：左右手 Gate、触觉变化和联合分布；
- `bimanual_action_examples.png`：混合 Gate 场景中的代表性与失败动作案例。

新图只在 `loss_mode: bimanual_gated` 时生成。旧 loss mode 和旧 `history.csv` 必须继续由现有绘图器正常处理。

## 数据定义

固定动作 schema 不变：左手为 `[0, 10)`，右手为 `[10, 20)`，夹爪分别为维度 9 和 19。Gate 阈值沿用训练配置中的 `rank_low_gate_threshold` 与 `rank_high_gate_threshold`，默认分别为 0.3 和 0.7。

验证样本按 `(w_left, w_right)` 分为：

- `low_low`：左低、右低；
- `high_low`：左高、右低；
- `low_high`：左低、右高；
- `high_high`：左高、右高。

任一手处于中 Gate 区间的样本不强行并入四象限，只进入 Gate 分布与样本数统计。

对每个四象限、每只手记录：

- `n`；
- `mse_gt = MSE(FRS, GT)`；
- `mse_vla = MSE(FRS, VLA)`；
- `mse_vla_gt = MSE(VLA, GT)`；
- `gt_gain = mse_vla_gt - mse_gt`；
- `relative_gt_error = mse_gt / max(mse_vla_gt, 1e-8)`；
- `vla_preserve_ratio = mse_vla / max(mse_vla_gt, 1e-8)`；
- `rank_satisfied_frac`。

比值采用“均值之比”，不采用逐样本比值的均值，避免单个接近零的 VLA→GT baseline 主导结果。原始 MSE 同时保留，防止只看比值造成误判。

## 图一：整体训练总览

`training_overview.png` 使用六个纵向面板：

1. epoch 平均 `total/composite_fm/decode/rank/low_safety/repair`；
2. `train_composite_fm` 与 `val_composite_fm`；
3. 全 20D 的 `MSE(FRS,GT)`、`MSE(FRS,VLA)` 和冻结的 `MSE(VLA,GT)` baseline；
4. 整体 `gt_gain` 与 `relative_gt_error`，分别标出 0 和 1 的参考线；
5. 左右手 `rank_satisfied_high_frac`、`low_safe_frac` 和 checkpoint feasible 状态，并标出配置门槛；
6. 左右手 Gate mean/p10/p50/p90 与 high/mid/low 样本数。

Loss 面板允许使用对称 log 或普通线性坐标，但不得隐藏 0 值分量。验证只在 `eval_every` epoch 出现，其他 epoch 不做插值。

## 图二：双手行为主图

`bimanual_behavior.png` 使用 4 行 × 2 列：行对应 `low_low/high_low/low_high/high_high`，列对应左手和右手。

每个子图按 epoch 绘制两个无量纲指标：

- `relative_gt_error`：越小越接近 GT，1 表示与原始 VLA 到 GT 的误差相同；
- `vla_preserve_ratio`：越小越保持 VLA，0 表示完全复现 VLA。

高 Gate 手突出显示 `relative_gt_error`，低 Gate 手突出显示 `vla_preserve_ratio`；非目标曲线保留为浅色参考。标题明确写出当前手在该象限的期望行为，并显示最新验证的 `n`、原始三项 MSE、GT gain 与 rank 满足率。

当 `n=0` 时显示无样本；当 `0<n<20` 时仍绘图但加醒目的“样本不足”标记，不把它用于可靠结论。20 只作为可视化可信度提示，不改变 checkpoint 选择。

## 图三：Gate 诊断

`gate_diagnostics.png` 使用 2×2 布局：

1. 左右手 Gate 的 median 及 p10–p90 区间；
2. 左右手 tactile change 的 median 及 p10–p90 区间；
3. 左右手 low/mid/high 样本数随 epoch 的变化；
4. 最新验证集的 3×3 联合 Gate 区域热力图，横轴为右手、纵轴为左手，每格同时显示样本数和百分比。

Gate 与 tactile change 使用不同子图，不使用双 Y 轴叠加。

## 图四：动作案例

`bimanual_action_examples.png` 只关注最关键的 `high_low` 和 `low_high` 两组。每组选择：

- 一个接近该组中位数的代表样本；
- 一个无触觉手 `mse_vla` 最大的漂移样本。

每个样本一行，包含：

1. 左手每个 horizon step 到 GT/VLA 的 L2 距离；
2. 右手每个 horizon step 到 GT/VLA 的 L2 距离；
3. 全 20D 的 `FRS - VLA` 热力图，并在维度 9/19 处标出夹爪；
4. 左右夹爪的 GT/VLA/FRS 三条轨迹。

图标题写明 cache index、episode、`w_left/w_right` 和两只手的三项 MSE。若某个混合象限没有样本，则保留占位说明，不用其他 Gate 区域冒充。

训练内验证已经完成 decode；实现应保留本轮验证预测用于绘图，不得为了四张图再次运行完整 ODE decode。每次验证覆盖稳定文件名，避免 30 epoch 产生大量图片；数值历史仍完整保存在 CSV。

## 数据流与兼容性

`evaluate_split` 继续产生整体和逐手 per-sample 指标，并新增一个独立的四象限聚合函数。训练循环把四象限标量写入 `history.csv`，把最新验证的必要样本数组传给可视化模块。历史绘图读取缺失字段时跳过曲线或显示 pending，因而旧 CSV 仍可重画。

现有 `training_curves.png`、standalone evaluate JSON/CSV 和旧图片名称不删除、不改语义。平均 Gate `(w_left+w_right)/2` 最多作为 legacy 对照，不能用于双手 checkpoint 选择或双手主图分层。

## 错误处理

- 非有限 Gate 或误差在聚合入口直接报错；
- 空象限返回 `n=0` 和 NaN 指标，由绘图器显示占位；
- baseline 近零时仍写原始 MSE，比值分母使用 `1e-8`；
- 单数据集和多数据集都生成前三张图；动作案例只有在能保持正确 source/cache 映射时生成，否则明确跳过并记录 warning；
- 绘图失败不得破坏 checkpoint 保存或训练历史写入。

## 测试与验收

测试覆盖：

1. 四象限边界、逐手切片和比值公式；
2. 单手高/另一手低时，两只手指标不互相污染；
3. 空象限与少样本标记；
4. 新旧 history CSV 均可绘图；
5. 四张图片存在且非空，面板数量和关键标签正确；
6. 动作案例同时读取左右手维度，并包含夹爪 9/19；
7. 绘图路径不触发第二次完整 decode；
8. `loss_mode: gated/gt/predicted` 的历史和图保持兼容。

本次工作不修改 loss、Gate 计算、optimizer、checkpoint 选择门槛或部署推理。
