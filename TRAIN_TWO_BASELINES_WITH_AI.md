把下面资源链接、Hugging Face 主页链接和 token 填好，然后将本文件交给有当前仓库终端访问权限的 AI，发送：

> 请读取并执行 `TRAIN_TWO_BASELINES_WITH_AI.md`。自动准备资源、生成配置、依次训练两个 baseline，每个训练完成后立即上传 checkpoint 到指定 Hugging Face 主页下，两个都上传成功后关闭当前 RunPod。发现训练失败或提前终止时，立即上传本次运行中已保存的最佳可用 checkpoint，然后关闭当前 RunPod；上传失败也关闭。请直接执行并交付上传链接和运行记录。

用户只需编辑下面的资源清单和两个上传字段。其余路径、单臂／双臂配置、动作字段、缓存和训练调度由 AI 根据本仓库及资源元信息确定。执行机器为 RunPod Pod，需要可用 GPU、持久存储、资源访问权限及当前 Pod 的停止权限。本指南完成条件为训练、上传和停机，无需训练后的独立离线评估。

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

publishing:
  # 填个人或组织主页，例如 https://huggingface.co/KaiyueChen。
  # AI 会在该主页对应的 namespace 下为两个 baseline 分别创建模型仓库。
  huggingface_home_url: ""
  # 填有该 namespace 模型仓库写入权限的 Hugging Face token。
  # 也可以保持空字符串，由执行环境的 HF_TOKEN 提供。
  huggingface_token: ""
```

资源 URL 可使用 Hugging Face 仓库页、指定 revision 的仓库页／文件链接，或可下载的资源压缩包链接。例如数据集仓库 URL 的形式是 `https://huggingface.co/datasets/OWNER/DATASET`，模型仓库是 `https://huggingface.co/OWNER/MODEL`。链接包含 revision 时必须尊重该 revision。

Hugging Face token 仅用于认证，读取后通过进程内参数或受限环境变量传给上传工具；日志、最终训练配置、报告和上传包均须排除 token。本文件填写真实 token 后的副本也须排除 Git 提交和 Hugging Face 上传。RunPod 使用容器环境提供的 `RUNPOD_POD_ID` 和 `RUNPOD_API_KEY`，后者与 Hugging Face token 是不同凭据；AI 启动训练前须确认这两项及对应停止权限可用。

以下内容是给执行 AI 的完整任务要求。

1. **确认目标和输入，然后直接推进。**

   目标是对每个 experiment 完整运行：

   | baseline | 训练目录 | Flow 起点 | 修正网络条件 |
   |---|---|---|---|
   | act_inter | `train_baseline_act_inter` | 冻结 VLA 输出动作 | 触觉历史＋当前 state |
   | state | `train_baseline_state` | 冻结 VLA 反向积分的 structured noise | 当前 state |

   两者都需要视觉、state、专家动作和冻结 Pi0.5；act_inter 额外需要触觉数据与编码器。视觉仍用于 state baseline 的上游 VLA。专家动作是训练监督目标。

   先读取两个目录的 README、配置解析、pipeline 和实际启动脚本。使用当前仓库实现。用户授权本机依赖安装、资源下载／转换、生成运行配置和下载／调度辅助文件、训练、上传到填写的 Hugging Face namespace，以及完成或失败后的当前 RunPod Stop。执行这些已授权步骤无需再要求用户确认；遵守执行平台的权限规则。

   检查必填 URL、上传主页和 token，解析目标 namespace，用 `HfApi.whoami()` 验证身份，再通过目标仓库的授权检查或创建操作确认写入权限。norm stats 和源模型配置优先从 checkpoint 解析，有补充链接时使用补充资源并校验一致性。先核对当前 Pod ID 和停机认证，并建立第 7 条要求的收尾监督器，再启动准备／训练流程。缺少无法推断的输入或停机权限时明确报告，避免启动无法按要求收尾的训练；已进入运行流程后的失败按第 8～10 条收尾。

2. **建立固定的本地运行目录和状态记录。**

   从本文件位置确定仓库根目录，不依赖某台机器上的 `/home/typhon`。将运行目录放在当前 Pod 已挂载的持久卷中，默认 `/workspace/baseline_runs/<run_id>/`；先确认该挂载实际可持久保存。仓库位于持久卷时也可使用仓库的 `.baseline_runs/<run_id>/`。运行目录包含：

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
   upload_staging/             只包含待上传模型产物的暂存包
   upload_manifest.json        两个目标 repo、选择的 checkpoint、哈希和上传状态
   run_all.sh                  可重复执行的完整流程入口
   finalize.py                 失败保全、上传和最终停止 Pod 的幂等收尾器
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

   在正式运行前根据 GPU 检查决定 batch size，记录相对模板的调整。正式运行中出现 OOM、非有限 loss 或其他训练失败时立即进入上传／停机收尾，不重新启动训练。不要通过截断正式样本、减少 epochs、修改 loss 或跳过反向积分来宣称完成训练。保留训练过程中用于选择 `best.pt` 的原有验证。

7. **预检后直接执行全流程，缓存由总入口生成。**

   用生成配置的绝对路径执行预检；以下变量由运行脚本设置：

   ```bash
   bash train_baseline_act_inter/scripts/start_train.sh --config "$ACT_CONFIG" --check
   bash train_baseline_state/scripts/start_train.sh --config "$STATE_CONFIG" --check
   ```

   先运行 act_inter：

   ```bash
   bash train_baseline_act_inter/scripts/start_train.sh --config "$ACT_CONFIG"
   ```

   它正常完成且第 8 条上传验证成功后，再运行 state 并立即上传：

   ```bash
   bash train_baseline_state/scripts/start_train.sh --config "$STATE_CONFIG"
   ```

   `--check` 仅检查配置及输入路径可读性；前面的真实数据检查、模型加载和 GPU 验证仍需执行。单 GPU 默认顺序训练，避免并发加载 VLA 或两个训练进程。

   act_inter 总入口自动运行触觉特征缓存、VLA／GT／state 动作缓存、修正网络训练。state 总入口自动运行 VLA／反向积分／GT／state 缓存、修正网络训练。缓存格式不同，不能直接互换，也不能将 TacFRS 缓存直接当作 baseline 缓存。

   准备阶段可修复下载、依赖、字段和路径问题；一旦正式总入口出现非零退出、提前终止、OOM、NaN 或子进程消失，立即停止后续训练任务，按第 9 条保全上传并关闭 Pod。若要先冒烟，使用独立缓存和输出目录，保证采样覆盖训练与验证 episode；不要盲目使用很小的 `--max-samples`，它可能只截取到一个 split。正式运行移除样本／步数上限。

   将流程写入运行目录的 `run_all.sh`，然后自动执行。使用独立父监督器运行训练子进程，逐阶段保存日志、真实退出码和预期 epoch／step，识别提前返回 0 的未完成训练。监督器与训练子进程分开管理进程组，捕获 `SIGINT`／`SIGTERM`，接到信号时停止训练并限时等待退出，避免读取正在写入的文件。

   成功路径、异常、信号和上传失败路径必须汇入同一个幂等收尾器。用 `try/finally` 或等效机制保证“尝试保全上传”之后始终调用停机；单纯依靠 `set -e` 会跳过错误后的上传／停止命令，不能作为完整监督方案。上传子进程也必须有总超时，退出码或超时均不能跳过 Stop。持续监督长任务，保留会话、进程与日志。

8. **训练完成后立即上传 checkpoint。**

   正常完成要求训练进程退出码为 0、`last.pt` 记录的 epoch 达到配置目标，且 `best.pt`／`last.pt` 可由对应加载器读取；`best.pt` 的选中 epoch 可以早于最后一轮。完成后直接打包上传，省去训练后的独立离线诊断／评估。

   从 `huggingface_home_url` 解析个人或组织 namespace；上传地址限定在这个 namespace 下。每个 experiment 为两个 baseline 创建独立模型仓库，例如 `<namespace>/<experiment>-baseline-act-inter-<run_id>` 和 `<namespace>/<experiment>-baseline-state-<run_id>`。名称自动生成并写入 `upload_manifest.json`，相同运行可以续传，其他运行使用独立名称。新建仓库默认 private，已有仓库沿用其可见性。

   采用当前安装版本支持的 `huggingface_hub.HfApi`：显式传入 token，`create_repo(repo_id=目标仓库, repo_type="model", private=True, exist_ok=True)` 创建目标，`upload_folder()` 上传独立暂存包。只打包已验证的 `best.pt`、可用的 `last.pt`、最终训练 YAML、所用 norm stats、源模型／编码器来源信息，以及不含凭据的训练状态、日志和 README。不要将整个工作目录或本任务文档作为上传包。官方接口见 [Hugging Face 上传指南](https://huggingface.co/docs/huggingface_hub/guides/upload)。

   上传必须等待远端 commit 完成，记录 repo URL、commit SHA 和各 checkpoint 的 SHA-256。使用该 commit 的远端文件信息或回读结果核对大小与哈希，不能仅凭文件上传进度达到 100% 判定成功。每个目标的上传和回读校验合计限时 5 分钟，包含工具内部重试；由父监督器实施总超时。确定上传失败或超时后立即进入停机流程，禁止无限重传或继续训练下一个 baseline。

9. **训练失败／终止时，上传本次最佳可用 checkpoint 后关闭。**

   一旦发现正式训练失败或提前终止，停止后续任务，收集当前 run 的已有产物。先选择本次运行最新保存且有效的 `best.pt`；有多个有效的 best 备份时，使用训练期间记录的验证指标 `decoder_smooth_l1` 选择最优，指标相同取 step 较新的。不要把其他 run、测试／冒烟运行或来源不兼容的文件混入。

   “可用”至少要求：文件写入已完成，使用本 baseline 的 `load_decoder_checkpoint(..., map_location="cpu")` 安全加载、结构及 `source_contract` 匹配，模型参数和已有选择指标有限。仅做加载与完整性检查，不重新跑离线诊断，也不为了挑 checkpoint 再训练。

   如果没有有效 best，但有有效 `last.pt`，将其作为回退结果，注明 `last_fallback`、原始 epoch／step 和失败原因。部署加载器要求精简的 best schema：用对应 checkpoint 模块的 `save_best_checkpoint()` 从已加载模型及原始 payload 导出便携副本，保留原来的 metrics 和 source contract；不要仅把含优化器等续训字段的 `last.pt` 改名。可同时保留原始 `last.pt` 供恢复。

   尝试上传已有的最佳可用产物与失败记录；此前已经验证上传成功的结果不必重复上传。尚未启动的另一个 baseline 标为 `not_started`。如果失败发生在首次有效保存前，记录 `no_valid_checkpoint`，尽力上传不含凭据的失败记录，仍执行停机。上传失败或超时记录 `upload_failed` 后同样关闭 Pod，不等待用户回复、不启动下一轮训练。

10. **最后始终执行当前 RunPod 的 Stop。**

   正常路径在两个 checkpoint 都验证上传成功后关闭；多个 experiment 时，所有计划结果上传成功后关闭。失败路径在第 9 条的限时保全上传结束后关闭，上传失败也关闭。终态顺序为：保存本地报告和上传状态、刷新文件到持久卷、记录 `stop_requested`、调用 RunPod Stop。最终报告写明完成／失败、权重 epoch／step、已成功的远端链接和 commit，以及未上传产物的持久卷路径。不得把失败收尾标为训练成功。

   关闭目标只使用已核对的当前 `RUNPOD_POD_ID`。优先使用已配置认证的当前 CLI；先按 `runpodctl --help` 核对版本，当前官方命令是：

   ```bash
   runpodctl pod stop "$RUNPOD_POD_ID"
   ```

   CLI 不可用时，用 RunPod REST Stop API；下面示例从环境读取 RunPod 凭据，不使用 Hugging Face token：

   ```bash
   python - <<'PY'
   import os
   import re
   import urllib.request

   pod_id = os.environ["RUNPOD_POD_ID"]
   api_key = os.environ["RUNPOD_API_KEY"]
   if not re.fullmatch(r"[A-Za-z0-9_-]+", pod_id) or not api_key:
       raise RuntimeError("Missing or invalid current RunPod identity / credential")
   request = urllib.request.Request(
       f"https://rest.runpod.io/v1/pods/{pod_id}/stop",
       data=b"",
       headers={"Authorization": f"Bearer {api_key}"},
       method="POST",
   )
   with urllib.request.urlopen(request, timeout=20) as response:
       if response.status != 200:
           raise RuntimeError(f"RunPod Stop returned HTTP {response.status}")
       print(f"RunPod Stop accepted for {pod_id}", flush=True)
   PY
   ```

   Stop 失败时，在父监督器仍存活的情况下最多重试 3 次，单次请求限时 20 秒，并尝试上述另一种可用方式。只能依据 API／CLI 的实际结果记录接受或失败；停机可能立即断开当前会话，不能因断线就伪称已核实 stopped。若控制面确实不可访问或认证不可用，保存失败原因与 Pod ID，明确报告仍需停止，不能谎报已关闭。

   本指南中的关闭采用 **Stop**，释放 GPU 并保留持久卷；容器临时盘会清除，保留的存储仍可能计费。Pod ID／API key 环境变量、CLI 和 REST 方法依据 [RunPod 环境变量](https://docs.runpod.io/pods/templates/environment-variables)、[管理 Pod](https://docs.runpod.io/pods/manage-pods) 和 [REST Stop API](https://docs.runpod.io/api-reference/pods/POST/pods/podId/stop)。

   上述监督器处理训练子进程失败及可捕获信号。整个 Pod 被强制回收、主机断电，或监督器本身被 `SIGKILL` 时，容器内程序无法继续上传／停机；需要保证覆盖这些情形时，应在 Pod 外部署监督器。不要把本地 `finally` 当成对这类中断的保证。
