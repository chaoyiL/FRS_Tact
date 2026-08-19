# 双手 FRS 组合端点损失设计

## 目标

为 20 维双手动作块训练一个触觉条件化的 FRS 速度场，并解决以下问题：当只有一只手出现明显触觉接触时，另一只未接触手不能在积分后偏离冻结 SmolVLA 的原始动作。

本设计为每个样本构造唯一且一致的双手联合端点：

- 左手 10 个动作维度由左腕 Gate 独立决定趋向 GT 还是 VLA；
- 右手 10 个动作维度由右腕 Gate 独立决定趋向 GT 还是 VLA；
- 从缓存的 `x_base` 到该联合端点只训练一条 Flow Matching 轨迹；
- 保留 decode、low-safety、rank 和 repair，并将它们改为按手计算。

本次修改不把 Gate 数值加入 decoder 输入。Decoder 仍从现有的触觉历史和 state 中自行判断 steering 状态，从而保持 decoder input version 2 和当前部署接口不变。

## 固定数据契约

本设计只适用于当前 VB3/pick-tube 的 20 维动作契约：

| 切片 | 含义 |
| --- | --- |
| `0:3` | 左手平移 |
| `3:9` | 左手 rotation-6D |
| `9` | 左夹爪 |
| `10:13` | 右手平移 |
| `13:19` | 右手 rotation-6D |
| `19` | 右夹爪 |

启用双手组合端点损失时，实现必须校验 `action_dim == 20`，不能对任意数据集直接使用 `action_dim // 2` 推断左右手边界。

触觉字段名中的 `left/right` 表示同一手腕上的两个传感器表面，不代表机器人左手或右手。正确的 Gate 分组为：

- 左腕：`tactile_left_0`、`tactile_right_0`；
- 右腕：`tactile_left_1`、`tactile_right_1`。

## 左右手 Gate

对于手腕 `h` 的两个触觉 token，分别与同一 episode 第一帧对应 token 比较：

\[
s_h=\frac12\sum_{k\in h}\left(1-\cos(e^{current}_k,e^{baseline}_k)\right).
\]

使用现有 sigmoid 标定得到原始 Gate：

\[
w_h=\sigma\left(\frac{s_h-\tau}{T}\right).
\]

再对每只手独立应用当前三段式映射：

\[
g_h=\operatorname{clip}\left(\frac{w_h-l}{u-l},0,1\right),
\qquad l=0.3,\ u=0.7.
\]

为了兼容现有日志，可以继续计算全局 Gate，但它只用于报告：

\[
w=\frac{w_L+w_R}{2}.
\]

`w_h` 和 `g_h` 都不接收梯度。第一版左右腕共用相同的 `tau` 和 temperature，同时在训练与验证指标中输出两只手各自的 Gate 分布。是否需要分别重新标定，后续根据真实数据统计决定。

## 组合端点与 Flow Matching

记 `G` 为 GT 动作块，`P` 为冻结 SmolVLA 动作块。按照左右手动作切片，为每个样本构造唯一端点 `Y`：

\[
Y_L=g_LG_L+(1-g_L)P_L,
\]

\[
Y_R=g_RG_R+(1-g_R)P_R,
\]

\[
Y=\operatorname{concat}(Y_L,Y_R).
\]

FRS 不再分别训练完整 GT 和完整 VLA 两条轨迹，而是只训练以下一致的组合轨迹：

\[
x_t=(1-t)x_{base}+tY,
\]

\[
v^*=Y-x_{base},
\]

\[
L_{FM}=\operatorname{mean}_{H,D}\left(v_\theta(x_t,t,c)-v^*\right)^2.
\]

这一项替换现有 gated 模式中的 `gt_fm` 和 `vla_fm` 两次计算。训练历史新增 `train_loss_composite_fm`。为了兼容旧绘图代码，可以暂时保留原来的两个 CSV 字段并明确写入 0，但不能把新的组合损失误导性地拆分到两列中。`train_loss_total` 和 `train_flow_loss` 保持当前含义。

由于不再存在独立的 VLA FM 分支，`gate_lambda` 不再参与计算。双手模式下如果配置仍包含 `gate_lambda`，应直接报错；仓库提供的双手 YAML 需要删除该字段，避免静默忽略。

## Decode 与辅助损失

使用配置指定的可微积分器，从 `x_base` 只 decode 一次：

\[
\hat A=\operatorname{Decode}_\theta(x_{base},c).
\]

沿 horizon 和每只手的 10 个动作维度分别计算：

\[
d^G_h=MSE_h(\hat A,G),\quad
d^P_h=MSE_h(\hat A,P),\quad
b_h=MSE_h(P,G).
\]

带阈值的分组辅助项（low-safety、rank、repair）采用活跃组归一化。左右手先各自归一化，再对实际活跃的手求平均。空活跃组返回 0，并且不进入活跃手数量的分母。因此，如果一个 batch 只有一只手包含高 Gate 样本，该手的损失不会被额外除以 2。Direct decode 对每个样本的两只手始终有定义，使用普通的双手平均。

### Direct decode

对两只手选择出的端点保留直接 decode 监督：

\[
L_{decode}=\lambda_{decode}\frac12\sum_{h\in\{L,R\}}
\left[g_hd^G_h+(1-g_h)d^P_h\right].
\]

与旧的“仅高 Gate 对 GT 做 decode”不同，新设计会直接把无触觉手锚定到 VLA 端点。所有辅助项复用同一个 decoded tensor，不重复积分。

### Low-gate safety

使用原始 Gate，在每只手内部保留当前的最近端点安全 hinge：

\[
L_{low,h}=\lambda_{low}\operatorname{WMean}_{(1-w_h)\mathbf1[w_h\le l]}
\left[\operatorname{ReLU}(\min(d^G_h,d^P_h)-\delta_{low})\right].
\]

该项与无触觉手的 decode-to-VLA 监督存在部分功能重叠，但按照本次要求，第一轮实验仍保留它，以便与现有消融结果比较。

### Rank

保留 rank，但改为每只手独立计算，防止一只手的改善掩盖另一只手的失败：

\[
L_{rank,h}=\lambda_{rank}\operatorname{WMean}_{w_h\mathbf1[w_h\ge u]}
\left[\operatorname{ReLU}(d^G_h-d^P_h+m_{rank})\right].
\]

现有 `balanced_mean` 与 `worst_source_cvar` 聚合方式继续保留，但输入改为每只手的局部 penalty。CVaR 模式下，每个“数据源/手腕”组合视为独立活跃组。

### Repair

保留 repair，并改为每只手独立与冻结 VLA baseline 比较：

\[
L_{repair,h}=\lambda_{repair}\operatorname{WMean}_{w_h\mathbf1[w_h\ge u]}
\left[\operatorname{ReLU}(d^G_h-b_h+m_{repair})\right].
\]

当前 YAML 中 `repair_weight` 为 0，因此该项继续实现和记录，但在修改配置之前不会产生梯度。

## 总损失

双手 gated 目标为：

\[
L=L_{FM}+L_{decode}+L_{low}+L_{rank}+L_{repair}.
\]

其中 `low`、`rank` 和 `repair` 是上述左右手分项经过活跃手归一化后的结果。

普通 batch reduction 和可选的 dataset-balanced reduction 保持当前语义。所有 MSE 继续在归一化模型动作空间中计算，并覆盖整个 action horizon。

## 训练与部署行为

FRS 参数树和 decoder 输入均不改变。使用新损失训练的 checkpoint 通过现有 metadata 校验后，可以继续由 decoder-input-v2 部署代码加载。部署执行网络时不需要显式提供 `w_L` 或 `w_R`。

由于两只手仍共享模型参数，这个训练损失提供的是软约束，而不是运行时硬保证。当前部署中的 inactive-arm XYZ protection 继续作为独立的最后安全保护。部署端按手 residual blend 不在本次修改范围内。

与现有 gated 目标相比，计算量变化如下：

- Flow Matching 从两次速度场计算（完整 GT、完整 VLA）减少为一次组合端点计算；
- FireFlow decode 仍然只执行一次，并由 decode、safety、rank、repair 共同复用；
- 不增加额外的触觉 ResNet 前向。

## 配置与 Checkpoint Metadata

新增明确的双手组合端点 loss mode，不静默改变旧 checkpoint 的语义。运行配置和 checkpoint metadata 必须记录：

- loss mode/version；
- 左右手 action slices；
- 左右腕 tactile-token groups；
- Gate thresholds、tau 和 temperature；
- 各辅助项权重及聚合方式。

Resume 必须拒绝旧 scalar-gate 目标或 action/tactile 分组不同的 checkpoint。旧 checkpoint 仍可按原有 decoder 配置进行评估和部署。

## 指标与评估

训练历史和验证输出新增：

- `gate_w_left`、`gate_w_right` 及各自分位数；
- 左右手分别到 GT 和 VLA 的 MSE；
- 左右手冻结 VLA baseline 到 GT 的 MSE；
- 左右手 low-safety violation fraction；
- 左右手 high-gate rank satisfaction 与 GT gain；
- composite-endpoint FM 和 decode 各项。

Checkpoint 选择不能允许一只手的收益抵消另一只手的退化。可行性条件按手判断，selection key 先比较较差的一只手，再比较整体误差。

## 错误处理

出现以下情况时，训练必须尽早失败：

- 双手模式下 action dimension 不是 20；
- 配置的 action slices 重叠、存在空缺或超出 action dimension；
- 左右腕触觉分组不能各自解析为恰好两个 token；
- resume checkpoint 使用不兼容的 loss metadata；
- 任意 Gate、组合端点、loss 或 decoded action 出现非有限值。

## 测试方案

单元测试必须覆盖：

1. 左腕使用 `_0` 两个 token，右腕使用 `_1` 两个 token；
2. `w_L=1,w_R=0` 时端点严格等于 `[GT_L,VLA_R]`；
3. `w_L=0,w_R=1` 时端点严格等于 `[VLA_L,GT_R]`；
4. 两个 Gate 同为 0 或同为 1 时，分别严格退化为完整 VLA 和完整 GT；
5. soft Gate 只插值其所属的 10 维动作切片；
6. composite FM 只调用一次模型速度场，并以 `Y-x_base` 为监督；
7. decode、safety、rank、repair 不允许跨手混合误差；
8. 空活跃手分组返回有限的零损失；
9. source-balanced 和 worst-source-CVaR reduction 保持左右手隔离；
10. 旧 scalar-gate checkpoint 不能 resume 双手训练；
11. 训练历史和评估指标包含新增的按手字段；
12. decoder 参数结构和部署推理保持向后兼容。

集成测试使用“仅单手接触”样本，验证一次优化更新会让接触手趋向 GT、未接触手趋向 VLA。另运行一个短程损失消融，对比 FM-only 与 FM+decode 的端点漂移和训练耗时；在此之前不修改 decode 频率或积分步数。
