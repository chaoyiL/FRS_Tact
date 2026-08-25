# train_deco

独立的 DECO Stage 1 双目视觉策略训练与 TorchScript 导出包。源码从
`/home/hillbot/deco/deco_baseline` 中当前验证过的 `cloud_training` 链路提取，
不依赖 FRS_Tact 根目录的 SmolVLA、JAX 或 FRS 训练代码。

## 固定数据合同

- 图像：`observation.images.camera0`、`observation.images.camera1`
- state：20D，`7 + 7 + 6`
- action：每手 `TCP delta xyz + Rotation6D matrix columns + absolute gripper`
- 默认 chunk：32
- 数据频率：30Hz
- 当前主训练链：LeRobot v2.1 parquet

## 使用

```bash
bash train_deco/scripts/setup_env.sh
bash train_deco/scripts/prepare_data.sh --mode local --root /path/to/pick_tube_01
bash train_deco/scripts/train.sh --mode local-smoke
```

正式训练使用 `--mode local-train`，服务器 DDP 使用 `--mode server-train`。
所有脚本从 FRS_Tact 仓库根目录启动；虚拟环境、manifest 和输出默认位于
`train_deco/` 下。

## 暗光增强训练

训练脚本默认启用 `low-light-v1`：25% 保留原始光照、55% 使用暗光曝光或
Gamma、20% 使用温和亮度扰动。两路相机在同一样本中共享光照参数，避免破坏
双视角对应关系。暗光范围来自真机无标签图片的亮度统计；这些图片不会作为
带动作标签的训练或验证样本。

从头训练时使用新的 `RUN_ID`，并且不要设置 `RESUME_FROM`：

```bash
RUN_ID=deco_low_light_v1 \
  bash train_deco/scripts/train.sh --mode server-train
```

可使用环境变量覆盖参数，例如：

```bash
AUGMENTATION_EXPOSURE_MIN=0.58 \
AUGMENTATION_EXPOSURE_MAX=0.90 \
AUGMENTATION_GAMMA_MIN=1.10 \
AUGMENTATION_GAMMA_MAX=1.50 \
  bash train_deco/scripts/train.sh --mode server-train
```

所有增强参数都会保存到 checkpoint 的 `config.augmentation` 中；exact resume
会拒绝不同的增强配置。验证和 TorchScript 导出不执行随机图像增强。
采用的无标签真机亮度统计记录在
`train_deco/configs/low_light_reference.yaml`，该文件只说明增强范围的来源，
训练代码不会读取其中的真机图片路径或把这些图片混入监督数据。

ResNet34 初始化权重默认路径：

```text
train_deco/pretrained/resnet34-b627a593
```

可用 `BACKBONE_WEIGHTS=/absolute/path` 覆盖。恢复旧 checkpoint 时应使用本机
路径覆盖旧训练机中保存的 `/home/ljl/...` 路径。

训练输出包含：

- `deco_stage1_latest.pt` / `deco_stage1_best.pt`：恢复训练用 checkpoint；
- `deco_stage1_latest.ts` / `deco_stage1_best.ts`：部署产物；
- 对应 `.ts.json`：输入输出合同和 TorchScript SHA256。

部署端只需要 `.ts` 和 `.ts.json`，见 `deploy_deco/`。
