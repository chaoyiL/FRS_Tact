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
