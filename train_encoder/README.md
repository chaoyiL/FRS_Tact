# 触觉 encoder 多任务训练

先一键下载选中的 episode，再训练：

```bash
bash scripts/download_encoder_data.sh
uv run python -m train_encoder.train --config train_encoder/configs/train.yaml
```

默认训练同一个共享触觉 ResNet18 encoder，RGB CLIP 保持冻结。

| 任务 | 有效手侧 | 训练 episode | 验证 episode |
| --- | --- | ---: | ---: |
| KaiyueChen/pick_01 | right | 300 | 35 |
| KaiyueChen/insert_01 | right | 300 | 35 |
| KaiyueChen/bread_03 | left、right | 300 | 35 |
| KaiyueChen/two_tubes_01 | left、right | 300 | 35 |

`tasks` 中的 `dataset_repo_ids` 定义该任务的数据来源，多个 repo 合并后按任务预算无放回随机抽取。
训练、验证按完整 episode 隔离；`split_seed` 固定时选择可复现。来源不足 335 条或所选 episode
短到无法构成 future pair 时，启动报错，避免重复采样凑数或静默少用数据。
共使用 1,200 条训练轨迹和额外 140 条验证轨迹。

下载默认放在 `.cache/encoder_data/train300_val35/`。脚本只读取完整来源 metadata，
实际数据只下载抽中的 335 个 episode Parquet/任务。四个来源是 v2.1 内嵌图像格式，
每个 Parquet 已包含 RGB 和触觉图像，不需要额外下载 videos，也不转换或改写原始 episode ID。
训练通过 `dataset_manifest` 直接读取这些本地文件，首次加载会在同目录生成所需图像列的
Arrow 缓存（需要额外磁盘空间），不会连接 Hub 补下未选中的数据。

`manifest.json` 记录来源 commit、原始 episode ID、训练/验证成员、文件大小及帧数；
只有全部任务的文件大小、Parquet 帧数、episode 标签和必要图像列检查通过后才写入。
每个来源的 `selection.json` 记录固定抽样计划，中断后重复运行即可续传；改变 seed、预算或来源时，
请使用新的下载目录。下载只读 Hugging Face，不上传或修改远端仓库。

可先查看抽样数量和数据体积（仅下载 metadata）：

```bash
bash scripts/download_encoder_data.sh --dry-run
```

自定义目录：

```bash
bash scripts/download_encoder_data.sh --output-dir /path/to/encoder_data
uv run python -m train_encoder.train --dataset-manifest /path/to/encoder_data/manifest.json
```

脚本优先使用 `DATA_TOOL_PYTHON` 或项目已有环境；没有环境时通过 `uv` 准备仅供下载的轻量依赖。
私有数据可先在对应环境中运行 `hf auth login`，或设置 `HF_TOKEN`。

`sides` 对应固定键：left 使用 `camera0` 和 `tactile_{left,right}_0`，right 使用 `camera1`
和 `tactile_{left,right}_1`（原始 LeRobot 键带 `observation.images.` 前缀）。单手任务只展开有效手侧。
每条轨迹按 `frame_stride: 5` 取当前帧，预测 `future_offset: 1` 个原始帧后的 RGB。
300 是轨迹数；不同轨迹长度和有效手侧数会带来不同的训练样本数，batch 中不保证任务或左右手比例相同。

默认参数为 batch 32、40 个 epoch、学习率 `3e-4`、warmup 2 个 epoch、cosine 调度，
每 2 个 epoch 验证，连续 8 个 epoch 未改善时早停。这是多任务训练的起始配置，尚未通过真实训练调优。
`val_fraction` 只用于旧的顶层 `dataset_repo_ids` 配置；`tasks` 模式使用显式的训练、验证预算。

输出在 `checkpoints/encoder_four_tasks_train300/`，含 `best/`、`last/`、训练曲线及
`data_selection.json`。该 JSON 和 checkpoint metadata 保存各任务实际选中的全局 episode ID、
有序 repo 列表及记录摘要；全局 ID 按 repo 顺序累加，恢复与评估时应保持来源数据和顺序不变。
恢复训练可加 `--resume`，程序会检查任务配置、下载清单摘要和记录摘要是否与 checkpoint 一致。
数据移动后，独立评估可用 `--dataset-manifest /new/path/manifest.json` 指定新位置。

独立评估默认从 checkpoint 恢复来源、任务预算、手侧和抽样参数：

```bash
uv run python -m train_encoder.evaluate_retrieval \
  --checkpoint-dir checkpoints/encoder_four_tasks_train300/best \
  --output-dir outputs/encoder_four_tasks_retrieval \
  --retrieval-pool-size 32
```

`--dataset-repo-id` 仍可覆盖训练 YAML 的整个任务配置，切回原有的所有 episode、双手采样模式；
修改多任务数据来源和每个任务预算时，直接编辑 YAML 的 `tasks`。
