# train_deco

独立的 DECO Stage 1 双目视觉策略训练与 TorchScript 导出包。源码从
`/home/hillbot/deco/deco_baseline` 中当前验证过的 `cloud_training` 链路提取，
不依赖 FRS_Tact 根目录的 SmolVLA、JAX 或 FRS 训练代码。

## 状态与动作合同

- 图像：`observation.images.camera0`、`observation.images.camera1`
- `dual-arm-20x20`：state 为 `7 + 7 + 6`，action 为双臂各 10D
- `single-right-arm-7x10`：state 为右臂 7D，action 为右臂 10D
- 每臂 10D action 均为 `TCP delta xyz + Rotation6D matrix columns + absolute gripper`
- 默认 chunk：32
- 数据频率：30Hz
- 当前主训练链：LeRobot v2.1 parquet

## 使用

```bash
bash train_deco/scripts/setup_env.sh
bash train_deco/scripts/prepare_data.sh --mode local --root /path/to/pick_tube_01
bash train_deco/scripts/train.sh --mode local-smoke
```

7D state / 10D action 不能仅凭维度判断左右臂，必须在生成 manifest 时显式声明：

```bash
PYTHON_BIN=/path/to/train_deco/.venv/bin/python \
bash train_deco/scripts/prepare_data.sh \
  --mode server \
  --root /path/to/insert_01 \
  --output /path/to/insert_01.json \
  --dataset-id insert_01 \
  --state-action-profile single-right-arm-7x10
```

正式训练使用 `--mode local-train`，服务器 DDP 使用 `--mode server-train`。
所有脚本从 FRS_Tact 仓库根目录启动；虚拟环境、manifest 和输出默认位于
`train_deco/` 下。

## Stage 2 触觉图像训练

Stage 2 从一个已经训练好的 Stage 1 checkpoint 初始化，冻结全部 Stage 1
参数和触觉 ResNet18，只训练四路触觉 token 的 sensor embedding、cross-attention
K/V 与 gate，以及 PI Adapter。首次本机启动可从仓库根目录运行：

```bash
ALLOW_CPU=1 bash train_deco/setup_environment.sh

bash train_deco/scripts/train.sh \
  --mode local-stage2 \
  --manifest /absolute/path/to/pick_tube_01_06.json \
  --run-id pick_tube_stage2_local \
  --stage1-checkpoint /home/typhon/FRS_Tact/checkpoints/deco/image_aug/deco_stage1_latest.pt \
  --tactile-encoder-checkpoint /home/typhon/FRS_Tact/checkpoints/encoder/encoder_ckpt_0824
```

`--manifest` 必须指向 `prepare_data.sh` 生成的 LeRobot v2.1 单根或多根
manifest。`ALLOW_CPU=1` 只允许环境安装后的健康检查在没有 CUDA 的机器上
完成，不会把正式训练改成 CPU 训练。如果虚拟环境已经存在，也可只补转换依赖：

```bash
train_deco/.venv/bin/python -m pip install \
  'safetensors>=0.5,<1' \
  'jax[cpu]>=0.4.30,<0.6' \
  'flax>=0.10,<0.12'
```

`safetensors` 是加载已转换 encoder 的运行时依赖；JAX/Flax 只在转换时惰性
导入，并在导入前强制选择 CPU，不进入训练 forward。给定上面的 JAX checkpoint
目录后，第一次启动会自动转换并校验 encoder，然后写入内容寻址缓存：

```text
/home/typhon/FRS_Tact/checkpoints/deco/tactile_encoder_cache/v1/<source_sha256>/current.json
/home/typhon/FRS_Tact/checkpoints/deco/tactile_encoder_cache/v1/<source_sha256>/generations/<generation>/encoder.safetensors
/home/typhon/FRS_Tact/checkpoints/deco/tactile_encoder_cache/v1/<source_sha256>/generations/<generation>/encoder.json
```

可用 `--tactile-encoder-cache /absolute/cache/path` 改写缓存根目录。相同源内容的
后续启动直接复用已验证的 artifact；源文件内容变化会产生新的 SHA256 目录。
`server-stage2` 使用 DDP 时只有 global rank 0 执行解析/转换并原子更新
`current.json`，随后广播 immutable generation 路径。barrier 后每个 rank 都 strict-load
同一个缓存 artifact，并独立复核 source/artifact SHA256、转换 metadata 与 parity 记录。
非零 rank 不导入 JAX。
因此 rank 0 校验转换 metadata 和 SHA256，其他 rank 也会在加载前独立复核。

Stage 2 checkpoint 写入 `OUTPUT_DIR/RUN_ID/`：

- `deco_stage2_latest.pt`
- `deco_stage2_best.pt`
- `deco_stage2_epoch_<N>.pt`（到达 `SAVE_EVERY` 时）
- `deco_stage2_latest.ts` / `deco_stage2_best.ts`
- `deco_stage2_epoch_<N>.ts` 及各自的 `.ts.json` sidecar

Stage 2 模式默认训练 50 epochs；Stage 1 模式仍默认 100。每个 Stage 2
`.pt` 成功落盘后才自动导出对应 TorchScript。导出失败不会删除或回滚
checkpoint，而会记录明确的 `torchscript_export_failed` 事件。

`--stage1-checkpoint` 只用于 fresh Stage 2：它严格校验 action/observation/chunk/
camera、模型架构、task、action mode、objective、dataset id 与 normalization stats，
再加载并冻结 Stage 1 权重。训练启动前还会独立构造并 strict-load Stage 1
reference，用固定 RNG/noise 验证零 gate、零 adapter Stage 2 的确定性对齐；
不一致立即终止。精确恢复 Stage 2 时改用 `--resume`：

```bash
bash train_deco/scripts/train.sh \
  --mode local-stage2 \
  --resume /absolute/output/pick_tube_stage2_local/deco_stage2_latest.pt
```

`--resume` 在 dataset/model/optimizer/scheduler 构造前恢复 checkpoint 保存的
LR/scheduler/batch size/adapter rank 等所有训练状态参数，以及完整 model、
optimizer、scheduler、scaler、epoch/step、归一化统计、RNG 和 provenance。
命令行只保留 output/run-id/epochs/workers/log/save/keep、TorchScript 尺寸和
验证 seed 等运行时覆盖。因此上面的仅 `--resume` 命令可恢复非默认训练配置，
无需 Stage 1 或 encoder 源路径；它会拒绝 Stage 1 checkpoint。

### Stage 2 数据合同

首版只支持 `--dataset-format lerobot-v21`。每个数据根必须在元数据和 Parquet
中提供如下四个固定字段名；loader 总是按这组固定字段名读取，所以流顺序
不依赖 Parquet 的物理列顺序。每个字段可以是任意但彼此一致的 RGB HWC `[H, W, 3]`
图像；当前实际数据是 `[224, 224, 3]`：

1. `observation.images.tactile_left_0`
2. `observation.images.tactile_right_0`
3. `observation.images.tactile_left_1`
4. `observation.images.tactile_right_1`

dataset sample 中它们被解码为 unit-space float32 `[4, 3, H, W]`，batch
合同为 `[B, 4, 3, H, W]`；触觉预处理再把它们转换为
`[B, 4, 3, 224, 224]`。触觉图像只做保持长宽比的 224×224 黑边 letterbox，
不应用视觉 ImageNet normalization 或暗光增强。缺失字段、非 image 类型、非 RGB
或四路形状不一致会在训练前报出具体字段；metadata/Parquet 列的排列不构成错误。
选择旧的
`preprocessed` backend 会明确报错
`Stage2 currently requires --dataset-format lerobot-v21`，不会静默丢弃触觉输入。

每个 train-subset 与 unseen validation 集、每个 validation noise seed 都记录三组
loss 和 velocity MAE：正常触觉、临时将所有 tactile gate 置零的
`tactile_disabled`、以及按 batch 确定性 roll 的 `shuffled_tactile`。gate
在 forward 后恢复，shuffle 不消耗正常验证 RNG；best checkpoint 和 early
stopping 始终只使用正常的 unseen loss。

## 图像增强训练

训练脚本的新运行默认使用 `balanced-light-v2`：25% 保留原始光照、75% 使用
温和亮度、对比度、饱和度和模糊扰动；不使用暗光曝光或 Gamma 增强。两路相机在
同一样本中共享增强参数，避免破坏双视角对应关系。

`low-light-v1` 保留用于复现旧训练：25% 保留原始光照、55% 使用暗光曝光或
Gamma、20% 使用温和亮度扰动。其暗光范围来自真机无标签图片的亮度统计；这些图片
不会作为带动作标签的训练或验证样本。

以下是新的 Bread Stage1 运行命令。manifest 路径记录自原始 Bread checkpoint；
该文件必须存在于目标训练机器上，仓库不声明它在本机可用：

```bash
RUN_ID=bread-deco-stage1-balanced-light-v2 \
AUGMENTATION_PRESET=balanced-light-v2 \
RESUME_FROM= \
bash train_deco/scripts/train.sh \
  --mode server-train \
  --manifest /home/ljl/FRS_Tact/train_deco/data_manifests/bread_01_03.json
```

所有增强参数都会保存到 checkpoint 的 `config.augmentation` 中；exact resume
会拒绝不同的增强配置。要 exact resume 旧 Bread checkpoint，必须显式设置
`AUGMENTATION_PRESET=low-light-v1`。验证和 TorchScript 导出不执行随机图像增强。
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
