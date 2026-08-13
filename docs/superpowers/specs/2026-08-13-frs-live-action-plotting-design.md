# FRS 实时动作绘图设计

## 目标

在 `frs_steering_v1` 模式下实时绘制三类绝对动作轨迹，同时保持 legacy 绘图行为不变：

- 每个 chunk 的完整 VLA 预测动作，显示为蓝色虚线；
- 仅最终被服务端确认 `scheduled` 的 steer 动作，显示为橙色实线/点；
- 机器人测量反馈，显示为黑色实线。

`stale` 和 `rejected` steer 仍写入 JSONL 诊断，但不进入橙色执行轨迹。

## 架构

不修改 client/server wire protocol。权威绘图字段由 robot server 在已有内部 trace callback 中补充，随后由 `ActionTraceLogger` 持久化。绘图进程将 `frs_chunk` header 与同一 `chunk_id` 的 `frs_steer` records 聚合成现有绘图核心可消费的 chunk view。

服务端必须记录实际控制路径使用的数据，而不是在绘图器中猜测：

- chunk 开始：`execution_mode`、`observation_timestamp`、`control_dt`、RTC action timestamps，以及初始 Quest-frame reference waypoint；
- chunk ready：完整 robot-space `action_vla`；
- scheduled steer：`action_index`、最终 `scheduled_timestamp`、robot-space selected action，以及传给 `env.exec_actions` 的绝对 Quest-frame waypoint；
- stale/rejected steer：保留状态和诊断，不记录为执行 waypoint。

## 时间轴

RTC 模式直接使用服务端从权威 observation capture timestamp 生成的 action timestamps。

Block 模式维护实时估计时间轴：已经 scheduled 的索引使用真实 `scheduled_timestamp`；未执行后缀从最后一个已知时间按 `control_dt` 外推。每次新动作 scheduled 后重新计算后缀，因此完整 VLA chunk 始终可见，而橙色 steer 点始终落在真实执行时间上。

## 坐标与曲线

VLA 仍按 chunk 初始 reference waypoint 累积相对 20D action，生成完整绝对 14D waypoint 轨迹。Steer 曲线不从客户端相对 action重新推断，而直接使用服务端实际转换并提交的绝对 waypoint，避免逐步 steer 时参考机器人状态变化造成偏差。机器人反馈继续按动作时间戳插值。

`full_prediction.png` 绘制完整 VLA、scheduled steer 和反馈；`executed_vs_actual.png` 仅绘制 scheduled steer 与反馈。不同 chunk 之间保留 NaN 分隔，不画跨 chunk 的伪连接线。

## 实时性与容错

沿用独立 headless Matplotlib 进程和约 1 秒刷新周期。FRS header 或 steer record 到达都会标记视图为 dirty 并触发下一次刷新。聚合状态有界并沿用现有 live-history 压缩策略。

trace 记录、校验、聚合或绘图失败均不得影响机器人控制；缺字段或损坏记录只使对应曲线缺失，并将错误保存在现有 trace 诊断中。正常 shutdown 仍以 `.trace_complete` 触发最终渲染。

## 兼容性

- legacy `log_chunk` JSONL schema 和绘图路径保持不变；
- `frs_steering_v1` wire message schema 保持不变；
- 调度、安全、ACK、RTC protection/stale 和 block pacing 逻辑保持不变；
- PNG 文件名和 session 输出目录保持不变。

## 验收测试

- RTC FRS 聚合使用权威完整时间轴；
- block FRS 时间轴随 scheduled 动作实时校正并外推后缀；
- 只有 `scheduled` steer 进入橙色曲线，`stale/rejected` 被排除；
- steer 使用服务端实际绝对 waypoint；
- FRS records 增量到达时实时视图更新；
- chunk 分隔、反馈插值、最终渲染和 bounded history 保持正确；
- legacy 绘图测试不变并全部通过。
