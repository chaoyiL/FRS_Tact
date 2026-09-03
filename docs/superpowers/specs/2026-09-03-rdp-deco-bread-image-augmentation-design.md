# RDP DECO Bread 图像增强设计

## 目标

在 RDP single-right LDP 训练中保留现有 `RandomCrop(ratio=0.9)`，并加入与 DECO Bread 一致的双相机共享光度增强，提高模型对亮度、对比度、色彩和轻微失焦变化的鲁棒性。

增强只影响重新训练的 LDP。AT、触觉 PCA embedding、验证、离线评估和真机推理行为保持不变。

## 增强契约

对每个训练观测时刻的 `camera1` 和 `camera2` 共享一次随机分支及全部随机参数：

- 25%：保持原图。
- 75%：依次应用亮度、对比度和饱和度调整。
  - 亮度倍率：`[0.8, 1.2]`
  - 对比度倍率：`[0.85, 1.30]`
  - 饱和度倍率：`[0.80, 1.15]`
- 非原图分支再以 20% 概率应用高斯模糊：
  - kernel：从 `3`、`5` 中随机选择
  - sigma：`[0.1, 1.0]`
- 光度运算期间将像素裁剪到 `[0, 1]`，随后恢复 RDP encoder
  既有的 `[-1, 1]` 输入域。

这对应 `train_deco/bread_phase/augmentation.py` 的有效 Bread 分支。该配置中的 low-light 分支概率为 0，因此 exposure 和 gamma 参数不会参与训练，不在 RDP 中增加无效路径。

## 数据流与作用范围

LDP 训练中的视觉处理顺序为：

1. policy 先按 RDP 既有逻辑把 RGB 从 `[0, 1]` 归一化到 `[-1, 1]`。
2. encoder 读取两路 RGB，将其临时映射回 `[0, 1]` 并组成视图维度。
3. 对同一展平观测共享 DECO Bread 光度参数，随后映射回 `[-1, 1]`。
4. 分回各相机，执行现有 `Resize(224, 224)`。
5. 执行现有 `RandomCrop(ratio=0.9)`，得到 `201×201` 输入。
6. 执行现有 ImageNet normalization。
7. 分别送入各自的 ResNet18 编码器。

RDP 在调用视觉编码器前将 batch 和 observation time 展平。因此，两路相机在同一个观测时刻共享增强参数，而两个历史时刻分别采样。这保持实现局部且不改变 policy/encoder 接口。

`model.eval()` 下跳过整个光度增强。随机裁剪模块继续按现有逻辑在 eval 模式使用中心裁剪。

## 代码边界

- 在 `multi_image_obs_encoder.py` 中增加一个无参数、train-only 的 Bread 光度增强模块。
- `MultiImageObsEncoder.forward` 在逐路视觉预处理前统一调用该模块。
- Hydra 配置显式声明光度增强参数；现有 `random_transforms` 中的 `RandomCrop` 保持不变。
- 训练 encoder 对核心光度参数整批采样和广播，避免逐样本 GPU scalar 同步。
- 部署侧 encoder 接受但忽略该纯训练配置字段，确保新 checkpoint 可加载。
- 不复制 DECO 的命令行参数系统，不修改数据集文件，不增强触觉 embedding。

旧 checkpoint 使用其保存的旧配置加载，不新增推理期随机行为。新训练或从旧权重开始的新 run 使用新配置；不承诺在同一 run 中以 `resume=true` 静默改变已记录的训练契约。

## 验证

增加小型单元测试覆盖：

- eval 模式严格返回原值。
- 固定随机种子时结果可复现。
- 两路相机使用相同光度参数。
- 光度运算值保持在 `[0, 1]`，模块输出保持在 `[-1, 1]`；配置非法时快速失败。
- 使用真实 RDP image normalizer 验证 `[0,1] → [-1,1] → 增强 → [-1,1]` 数据流。
- 含 Bread 配置的新 checkpoint 可由部署侧 encoder 构造。
- single-right LDP 的最终 Hydra 配置同时包含 Bread 光度增强和 `RandomCrop(ratio=0.9)`。

运行相关测试，并打印一次最终 Hydra 配置确认实际训练入口启用增强。
