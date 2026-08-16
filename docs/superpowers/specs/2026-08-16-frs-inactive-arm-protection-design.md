# FRS 非活动臂 XYZ 保护设计

## 目标

当某一只手的 VLA 物理空间 XYZ 三轴绝对值都小于可配置阈值时，FRS 最终输出的该臂 XYZ 必须回退为 VLA 预测，避免近静止动作被 FRS 放大为漂移。左右臂独立判断。

## 配置

在 `frs` 下新增：

```yaml
inactive_arm_xyz_threshold_m: 0.00025
```

阈值单位为米。`null` 关闭保护。启用值必须是有限正数；布尔值、字符串、零、负数、NaN 和无穷大均拒绝。

## 数据流

保护在 temporal ensemble 完成之后、最终反归一化之前执行：

1. 从缓存的 robot-space VLA chunk 读取当前 action index。
2. 分别检查左臂 `0:3` 和右臂 `10:13`。
3. 若 `max(abs(vla_xyz)) < threshold`，把待发送 normalized action 对应的 XYZ 替换为缓存的 normalized VLA XYZ。
4. 再走现有反归一化和夹爪增益流程。

这样 `selected_normalized` 与实际发送的 `selected_action` 保持一致。`decoded_normalized` 和 FRS diagnostics 保留原始 FRS 输出，以便诊断模型偏移。

## 边界和安全行为

- 判定严格使用 `<`；任一轴等于或超过阈值均不保护。
- 只替换该臂 XYZ，不替换旋转或夹爪。
- 左右臂独立触发。
- 启用保护时要求 action dimension 为 20，否则初始化时失败。
- 现有完整 FRS chunk 的 shape、finite、max-abs 和 delta-RMS 检查仍先执行；保护不能掩盖超限 FRS 输出。

## 测试

覆盖配置默认关闭、合法/非法阈值、20D 契约、左右臂独立触发、双臂触发、负值、严格阈值边界、只替换 XYZ、非恒等反归一化以及 temporal ensemble 后仍执行保护。
