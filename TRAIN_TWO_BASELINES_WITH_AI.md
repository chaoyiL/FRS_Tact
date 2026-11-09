把下面资源链接填好，然后将本文件交给有当前仓库终端访问权限的 AI，发送：

> 请读取并执行 `TRAIN_TWO_BASELINES_WITH_AI.md`。自动准备所有资源、生成配置、依次完整训练两个 baseline，并完成离线评估。请直接执行，持续处理可修复的问题，最终交付权重、配置、日志和评估结果。

用户只需编辑下面的 URL 清单。其余路径、单臂／双臂配置、动作字段、缓存和训练调度由 AI 根据本仓库及资源元信息确定。前提是链接能提供完整且相互匹配的数据和模型资产，执行机器有可用 GPU、磁盘与资源访问权限。

```yaml
resources:
  # 两个 baseline 中 act_inter 使用的冻结触觉编码器，按现有实验使用 0824 版本。
  # 填 Hugging Face 模型仓库或完整 checkpoint 压缩包的 URL。
  tactile_encoder_url: "KaiyueChen/encoder_ckpt_0824"

  # Pi0.5 预处理需要的公共资源，默认地址来自仓库 tokenizer 实现。
  paligemma_tokenizer_url: "gs://big_vision/paligemma_tokenizer.model"

  experiments:
    # 每一项对应一份数据集和一个匹配的冻结 Pi0.5，分别训练两个 baseline。
    # 多个任务请复制这一整项并填写各自链接；当前 baseline 每个训练配置只接一份数据集。
    - dataset_url: ""
      pi05_checkpoint_url: ""
      # 填该 Pi0.5 实际使用的 norm_stats.json 文件 URL。
      # 如果已完整包含在 checkpoint 中，可保持 null，由 AI 从资产中解析。
      norm_stats_url: null
      # 如果 checkpoint 未包含模型档位、相机映射、归一化方式等来源信息，
      # 可补充原 Pi0.5 训练配置的文件 URL；已包含时保持 null。
      pi05_training_config_url: null
```

URL 可使用 Hugging Face 仓库页、指定 revision 的仓库页／文件链接，或可下载的资源压缩包链接。例如数据集仓库 URL 的形式是 `https://huggingface.co/datasets/OWNER/DATASET`，模型仓库是 `https://huggingface.co/OWNER/MODEL`。链接包含 revision 时必须尊重该 revision。不要在本文件填写访问 token；使用执行环境已有的凭据。

以下内容是给执行 AI 的完整任务要求。

1. **确认目标和输入，然后直接推进。**

   目标是对每个 experiment 完整运行：

   | baseline | 训练目录 | Flow 起点 | 修正网络条件 |
   |---|---|---|---|
   | act_inter | `train_baseline_act_inter` | 冻结 VLA 输出动作 | 触觉历史＋当前 state |
   | state | `train_baseline_state` | 冻结 VLA 反向积分的 structured noise | 当前 state |

   两者都需要视觉、state、专家动作和冻结 Pi0.5；act_inter 额外需要触觉数据与编码器。视觉仍用于 state baseline 的上游 VLA。专家动作是训练监督目标。

   先读取两个目录的 README、配置解析、pipeline 和实际启动脚本。使用当前仓库实现。用户授权本机依赖安装、上述资源下载／转换、生成运行配置和必要的下载／调度辅助文件、训练与离线评估。遵守执行平台的权限规则。

   检查必填 URL 是否为空；norm stats 和源模型配置优先从 checkpoint 解析，有补充链接时使用补充资源并校验一致性。只有确实缺少无法推断的来源信息、凭据或硬件条件时才报告具体阻塞；不要为路径命名、常规安装或正常启动再次要求确认。

2. **建立固定的本地运行目录和状态记录。**

   从本文件位置确定仓库根目录，不依赖某台机器上的 `/workspace` 或 `/home/typhon`。在仓库根目录建立 `.baseline_runs/<run_id>/`，包含：

   ```text
   resource_manifest.json       原 URL、解析后的 revision、下载位置和身份校验信息
   run_state.json               各阶段状态、退出码、进程／会话和日志位置
   inputs/                     原始下载与冻结资产
   datasets/                   准备好的 LeRobot v3 数据
   configs/                    每个 experiment 的两份最终训练 YAML
   cache/<experiment>/act_inter/
   cache/<experiment>/state/
   outputs/<experiment>/act_inter/
   outputs/<experiment>/state/
   logs/
   metrics/
   run_all.sh                  可重复执行的完整流程入口
   training_report.md
   ```

   缓存、输出和数据／权重目录互相分开。相同实验中两个 baseline 共享冻结输入资产，但动作缓存和训练输出独立。更换数据、revision、权重或统计量后建立新的运行标识。

   已有同来源、完整且校验通过的资源可复用，保留文件身份与时间戳，避免无意义重写导致续训来源校验失配。资源不完整时续传或重新下载到独立临时目录，验证后再发布。

3. **准备独立环境，验证 GPU。**

   检查 `uv`，缺失时按执行环境可用方式安装。先退出会与项目隔离规则冲突的已激活虚拟环境，再在仓库根目录执行：

   ```bash
   bash train_baseline_act_inter/scripts/setup_env.sh
   bash train_baseline_state/scripts/setup_env.sh
   ```

   两者使用各自 Python 3.12 `.venv`。分别用项目解释器检查 PyTorch CUDA 和 JAX GPU，并做小型设备计算；只看到 GPU 名称或安装成功不足以证明可训练。

   下载／转换数据时，复用已有数据工具环境；没有时按 `data_tools/pyproject.toml` 创建独立 Python 3.12 环境，例如：

   ```bash
   UV_PROJECT_ENVIRONMENT="$PWD/data_tools/.venv" \
     uv sync --project data_tools --python 3.12
   ```

   调用数据工具时明确选择这个解释器，并使仓库根目录在它的 `PYTHONPATH` 中，以使用仓库的 LeRobot 转换器。启动 baseline 时使用其启动脚本，让私有 `src` 优先，避免在同一进程中混用不同项目的 LeRobot。

4. **下载精确的 URL 资源并准备数据。**

   Hugging Face 仓库用 Hub 下载工具获取完整文件，保留大型权重的实际内容；仓库网页 HTML 不是下载产物。其他文件按真实格式下载、校验并解压到独立输入目录；解压路径必须限制在目标目录内。

   数据准备优先复用 `scripts/download_data.sh` 的流程。注意它当前固定下载 `KaiyueChen/<名称>`，`--dataset OWNER/NAME` 会截掉 OWNER；不能把其他命名空间的链接直接交给它。来源或 revision 不适配脚本时，下载准确的数据仓库，复用同一个仓库内转换器完成本地转换，必要时编写辅助调度代码。

   读取 `meta/info.json` 判断版本。已有合格 v3 数据直接使用；v2.1 数据在工作副本中转换成 v3，保留原始下载。转换入口为 `lerobot.datasets.v30.convert_dataset_v21_to_v30`，参数以当前代码／帮助为准，使用明确的本地 `--root` 并关闭 `--push-to-hub`。

   需要保留 act_inter 使用的触觉流，采用完整数据。读取实际 metadata 和 Parquet，确认视觉、触觉、state、专家动作、episode／frame 对齐及终止帧。自动选择实际动作键；本仓库转换结果通常为 `action`，不要照抄示例中的 `actions`。

   验证 profile：右臂为 7D state／10D action，双臂为 20D state／20D action。VLA 的原生动作宽度另行从模型确定。原始双臂数据不能仅靠改配置冒充右臂数据。检查 episode 划分后 train、validation、test 都有有效样本。

5. **解析模型资产和配套统计量。**

   找到实际包含 `params/` 的原生 Orbax Pi0.5 checkpoint 根目录，资源包多套一层目录时要正确定位。仅有 PyTorch 权重或独立 PEFT adapter 不满足当前 baseline 的加载接口；检查是否提供仓库已有的可靠转换／合并途径，无法获得完整原生权重时报告需要替换的资源链接。

   从原训练配置、checkpoint 元信息和参数形状确定实际 `model_action_dim`、`paligemma_variant`、`action_expert_variant`、相机映射和归一化方式，再用项目加载器验证。相机 rename 要按数据真实键和源 VLA 的训练含义生成，不照抄可能发生键冲突的映射。

   norm stats 必须来自同一个源 VLA 的训练配置。多个同维度候选无法唯一判定时，使用用户的 `norm_stats_url` 或源训练配置，不随意挑选。不要为 baseline 重新计算一套不同统计量。要求其 state／actions 统计覆盖对应物理维度，不能靠补齐 identity stats 掩盖不匹配。

   将选定统计文件放在稳定的 `<norm_stats_dir>/<asset_id>/norm_stats.json` 下，记录来源和内容哈希，自动写入配置。独立文件 URL 没有 asset 名称时可创建稳定名称，但保留原始来源信息。

   验证 0824 编码器的 `checkpoint.json`、参数文件和所需的 tactile ResNet 参数能被当前加载器读取。两路／四路触觉顺序按训练代码要求设置。

   预热 PaliGemma tokenizer。现有加载器从 `gs://big_vision/paligemma_tokenizer.model` 读取，并使用 `OPENPI_DATA_HOME`（默认 `~/.cache/openpi`）缓存；用户给了镜像链接时，将实际 tokenizer 文件准备在该加载器期望的缓存位置。不要只下载到一个运行时不会读取的目录。

6. **生成两份最终 YAML，保持实验定义。**

   单臂以各项目 `configs/*_task3.yaml` 为结构模板，双臂以各项目 `configs/train_baseline_*.yaml` 通用配置为模板。模板只是结构参考；数据、权重、统计量和所有绝对路径均从本次解析结果写入。

   两个 baseline 使用同一数据、源 VLA、归一化统计、相机映射、`frame_stride: 5`、`split_seed: 42`、train／validation／test 比例 `0.8/0.1/0.1` 和训练 seed `0`。维持模板的训练目标及网络结构，默认完整训练 50 epochs，学习率 `3e-4`、weight decay `1e-4`。

   act_inter 保留触觉配置、冻结 encoder、10 帧触觉历史和 stride 3，以 VLA 动作为起点。state 配置包含 `reverse_steps: 50`、`reverse_solver: slerpflow`，去除触觉相关配置，以 structured noise 为起点。两者保留各自既有 loss 与 gripper 行为，FireFlow 解码 10 步。

   明确写入 `dataset.root/repo_id/action_key/camera_map/rename_map`、源模型和 norm 资产、`cache.*_root`、`decoder.output`、动作与 state 维度。state 的 YAML 使用自己的 schema。初次训练 `decoder.resume: false`；只有相同实验存在兼容 `last.pt` 时才设置为 true。

   默认保持各模板 batch size；显存不足时可降低 cache、训练和评估 batch size，记录调整。不要通过截断正式样本、减少 epochs、修改 loss 或跳过反向积分来宣称完成训练。

7. **预检后直接执行全流程，缓存由总入口生成。**

   用生成配置的绝对路径依次执行对应命令；以下变量由运行脚本设置：

   ```bash
   bash train_baseline_act_inter/scripts/start_train.sh --config "$ACT_CONFIG" --check
   bash train_baseline_state/scripts/start_train.sh --config "$STATE_CONFIG" --check

   bash train_baseline_act_inter/scripts/start_train.sh --config "$ACT_CONFIG"
   bash train_baseline_state/scripts/start_train.sh --config "$STATE_CONFIG"
   ```

   `--check` 仅检查配置及输入路径可读性；前面的真实数据检查、模型加载和 GPU 验证仍需执行。单 GPU 默认顺序训练，避免并发加载 VLA 或两个训练进程。

   act_inter 总入口自动运行触觉特征缓存、VLA／GT／state 动作缓存、修正网络训练。state 总入口自动运行 VLA／反向积分／GT／state 缓存、修正网络训练。缓存格式不同，不能直接互换，也不能将 TacFRS 缓存直接当作 baseline 缓存。

   可修复下载、依赖、字段和路径问题后重试；保留诊断与日志。若要先冒烟，使用独立缓存和输出目录，保证采样覆盖训练与验证 episode；不要盲目使用很小的 `--max-samples`，它可能只截取到一个 split。正式运行移除样本／步数上限。

   将流程写入本次运行目录的 `run_all.sh`，然后自动执行；逐阶段保存日志和真实退出码，脚本失败必须向上传递。持续监督长任务，按执行环境支持的方式保留会话、进程与日志，记录进度和恢复方式。启动进程本身不代表训练完成。

8. **完成训练后执行离线评估。**

   两个训练进程都需正常结束，产生能被对应加载器读取的 `best.pt` 和 `last.pt`，checkpoint 记录的 epoch 达到配置目标。随后分别使用同一运行配置和缓存评估 `best.pt`：

   ```bash
   train_baseline_act_inter/.venv/bin/python -m train_baseline_act_inter.evaluate \
     --config "$ACT_CONFIG" --checkpoint "$ACT_OUTPUT/best.pt" \
     --split test --output "$ACT_METRICS_JSON"

   train_baseline_state/.venv/bin/python -m train_baseline_state.evaluate \
     --config "$STATE_CONFIG" --checkpoint "$STATE_OUTPUT/best.pt" \
     --split test --output "$STATE_METRICS_JSON"
   ```

   检查指标有限、test split 非空，报告原 VLA／修正网络误差、relative MSE reduction 和 delta RMS。训练与评估结果按实际数值报告；修正器未优于 VLA 时也保留结果。

9. **交付可核查的结果。**

   在 `training_report.md` 汇总每个 experiment 的原始 URL／revision、实际资源路径、最终配置、环境和 GPU、样本与 split 数量、缓存路径、训练 epochs／steps、batch 调整、best／last 权重、日志、评估指标及复跑命令。

   只有全部 experiment 的两个 baseline 均完成正式训练和评估，才将总状态标为 completed。若仍在运行或被外部条件阻塞，如实标为 running／blocked，给出当前阶段、具体原因、日志和可继续执行的命令。

   默认交付本机训练产物；额外的 Git 推送、权重上传和机器人部署由用户另行指定。
